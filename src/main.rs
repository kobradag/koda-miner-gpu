#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;
use std::fs;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use clap::{App, FromArgMatches, IntoApp};
use kobra_miner::PluginManager;
use log::{error, info};
use crate::cli::Opt;
use crate::client::grpc::KobradHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;

mod cli;
mod client;
mod kobrad_messages;
mod miner;
mod pow;
mod target;
mod watch;

// List of allowed dynamic libraries (DLLs or shared objects) to load as plugins
const WHITELIST: [&str; 4] = ["libkobracuda", "libkobraopencl", "kobracuda", "kobraopencl"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
}

// Type alias for common error handling
pub type Error = Box<dyn StdError + Send + Sync + 'static>;
type Hash = Uint256;

// Number of blocks per mining cycle
const BLOCK_CYCLE: u16 = 100;
// Number of blocks in the cycle dedicated to dev fee mining (2 out of 100)
const DEVFEE_CYCLE: u16 = 2;

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    // Disable "Quick Edit Mode" in Windows consoles to prevent the miner from freezing
    let console = win32console::console::WinConsole::input();
    let mut mode = console.get_mode()?;
    mode = (mode & !win32console::console::ConsoleMode::ENABLE_QUICK_EDIT_MODE)
        | win32console::console::ConsoleMode::ENABLE_EXTENDED_FLAGS;
    console.set_mode(mode)?;
    Ok(())
}

// Filters and returns a list of plugins that match the allowed file extensions and names
fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                // Check if the file is a dynamic library (DLL or shared object)
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| {
                // Check if the file name matches an entry in the whitelist
                WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap())
            })
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

// Determines which type of client (Stratum or gRPC) to create based on the provided address
async fn get_client(
    kobra_address: String,
    mining_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
) -> Result<Box<dyn Client + 'static>, Error> {
    if kobra_address.starts_with("stratum+tcp://") {
        let (_schema, address) = kobra_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string().clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
        )
        .await?)
    } else if kobra_address.starts_with("grpc://") {
        Ok(KobradHandler::connect(
            kobra_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
        )
        .await?)
    } else {
        // Return an error if the address schema is not recognized
        Err("Did not recognize pool/grpc address schema".into())
    }
}

// Main logic for starting the mining client
async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
) -> Result<(), Error> {
    let mut client = get_client(
        opt.kobra_address.clone(),
        opt.mining_address.clone(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
    )
    .await?;

    // Set up the dev fee to mine a small portion of the time for development support
    client.add_devfund(String::from("kobra:qql0qdtl7q2kypx69lp72hc2fas46x22enqsfvfpw7g34l5uyz5264xjnfqdx"), 2);
    
    // Register the client with the mining pool
    client.register().await?;
    
    // Create a miner manager with the client and plugin setup
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    
    // Start listening for new blocks to mine
    client.listen(&mut miner_manager).await?;
    
    // Clean up the miner manager when done
    drop(miner_manager);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    // Adjust the Windows console settings if running on Windows
    adjust_console().unwrap_or_else(|e| {
        eprintln!("WARNING: Failed to protect console ({}). Any selection in console will freeze the miner.", e)
    });

    // Determine the current executable's directory
    let mut path = current_exe().unwrap_or_default();
    path.pop(); // Get the parent directory
    let plugins = filter_plugins(path.to_str().unwrap_or("."));
    
    // Load the plugins and initialize command-line arguments
    let (app, mut plugin_manager): (App, PluginManager) = kobra_miner::load_plugins(Opt::into_app(), &plugins)?;
    let matches = app.get_matches();
    
    // Process plugin options and configure command-line arguments
    let worker_count = plugin_manager.process_options(&matches)?;
    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    opt.process()?;
    
    // Initialize logging
    env_logger::builder().filter_level(opt.log_level()).parse_default_env().init();
    
    // Display initial startup information
    info!("=================================================================================");
    info!("                 Kobra-Miner GPU {}", env!("CARGO_PKG_VERSION"));
    info!(" Mining for: {}", opt.mining_address);
    info!("=================================================================================");
    info!("Found plugins: {:?}", plugins);
    info!("Plugins found {} workers", worker_count);
    
    // Check if any workers are defined; exit if none are found
    if worker_count == 0 && opt.num_threads.unwrap_or(0) == 0 {
        error!("No workers specified");
        return Err("No workers specified".into());
    }

    let block_template_ctr = Arc::new(AtomicU16::new(0));
    let devfund_percent = 200; // Percentage of mining time dedicated to the dev fee

    if devfund_percent > 0 {
        info!(
            "Devfund enabled, mining {}.{}% of the time to devfund address: {}",
            devfund_percent / 100,
            devfund_percent % 100,
            opt.devfund_address
        );
    }

    loop {
        // Update the block template counter to determine which block is being mined
        update_block_template_ctr(&block_template_ctr);

        let current_block = block_template_ctr.load(Ordering::SeqCst);
        let is_devfee_block = current_block < DEVFEE_CYCLE;  // Check if this block should be mined as part of the dev fee

        if is_devfee_block {
            info!("Mining to DevFee address on block {}", current_block);
        } else {
            info!("Regular mining on block {}", current_block);
        }

        // Start the mining client and handle errors or reconnections
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error {:?}", e),
        }
        
        info!("Client closed, reconnecting");
        sleep(Duration::from_millis(100));  // Wait briefly before reconnecting
    }
}

// Updates the block template counter, resetting it when it reaches the cycle limit
fn update_block_template_ctr(block_template_ctr: &Arc<AtomicU16>) {
    let current_value = block_template_ctr.fetch_add(1, Ordering::SeqCst);
    if current_value >= BLOCK_CYCLE {
        block_template_ctr.store(0, Ordering::SeqCst);  // Reset counter when it reaches the limit
    }
}
