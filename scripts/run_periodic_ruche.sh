#!/bin/bash

echo "[RUCHE PERIODIC SCRIPT]"
echo "1) Setting up ruche environement"
echo "2) Update local state metadata"
echo "3) Collect extracted features to the /dump folder"
echo "4) Wipe processed videos"
echo "5) Update local state"


source base_env.sh
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_metadata.sh
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_collect_outputs.sh
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_wipe_remote.sh
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_metadata.sh
