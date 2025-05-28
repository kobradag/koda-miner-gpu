####################################################################################
###
### kobra-miner
### https://github.com/kobradag/koda-miner-gpu/releases
###
### Hive integration: Merlin
###
####################################################################################

#!/usr/bin/env bash
[[ -e /hive/custom ]] && . /hive/custom/kobra/h-manifest.conf
[[ -e /hive/miners/custom ]] && . /hive/miners/custom/kobra/h-manifest.conf
conf=""
conf+=" --mining-address $CUSTOM_TEMPLATE" -s $CUSTOM_URL


[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
