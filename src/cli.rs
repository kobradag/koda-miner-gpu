use clap::Parser;
use log::LevelFilter;

use crate::Error;

#[derive(Parser, Debug)]
#[clap(name = "kobra-miner", version, about = "A Kobra high performance CPU miner", term_width = 0)]
pub struct Opt {
    #[clap(short, long, help = "Enable debug logging level")]
    pub debug: bool,
    #[clap(short = 'a', long = "mining-address", help = "The Kobra address for the miner reward")]
    pub mining_address: String,
    #[clap(short = 's', long = "kobra-address", default_value = "127.0.0.1", help = "The IP of the kobra instance")]
    pub kobra_address: String,

    #[clap(short, long, help = "Kobrad port [default: Mainnet = 44448, Testnet = 44450]")]
    port: Option<u16>,

    #[clap(long, help = "Use testnet instead of mainnet [default: false]")]
    testnet: bool,
    #[clap(short = 't', long = "threads", help = "Amount of CPU miner threads to launch [default: 0]")]
    pub num_threads: Option<u16>,
    #[clap(
        long = "mine-when-not-synced",
        help = "Mine even when kobra says it is not synced",
        long_help = "Mine even when kobra says it is not synced, only useful when passing `--allow-submit-block-when-not-synced` to kobra  [default: false]"
    )]
    pub mine_when_not_synced: bool,

    #[clap(skip)]
    pub devfund_address: String,
}

impl Opt {
    pub fn process(&mut self) -> Result<(), Error> {
        //self.gpus = None;
        if self.kobra_address.is_empty() {
            self.kobra_address = "127.0.0.1".to_string();
        }

        if !self.kobra_address.contains("://") {
            let port_str = self.port().to_string();
            let (kobra, port) = match self.kobra_address.contains(':') {
                true => self.kobra_address.split_once(':').expect("We checked for `:`"),
                false => (self.kobra_address.as_str(), port_str.as_str()),
            };
            self.kobra_address = format!("grpc://{}:{}", kobra, port);
        }
        log::info!("kobra address: {}", self.kobra_address);

        if self.num_threads.is_none() {
            self.num_threads = Some(0);
        }

        let miner_network = self.mining_address.split(':').next();
        self.devfund_address = String::from("kobra:qql0qdtl7q2kypx69lp72hc2fas46x22enqsfvfpw7g34l5uyz5264xjnfqdx");
        let devfund_network = self.devfund_address.split(':').next();
        if miner_network.is_some() && devfund_network.is_some() && miner_network != devfund_network {
            log::info!(
                "Mining address ({}) and devfund ({}) are not from the same network. Disabling devfund.",
                miner_network.unwrap(),
                devfund_network.unwrap()
            )
        }
        Ok(())
    }

    fn port(&mut self) -> u16 {
        *self.port.get_or_insert(if self.testnet { 44450 } else { 44448 })
    }

    pub fn log_level(&self) -> LevelFilter {
        if self.debug {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        }
    }
}
