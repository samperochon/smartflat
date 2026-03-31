# 1) Update state, collect and distribute outputs

echo '1) Update local metadata 2) Wipe processed videos' #Collect outputs 3) Distribute local outputs 4) Print compute log'

source base_env.sh
echo "Done setting up environement"

source $SMARTFLAT_ROOT/api/scripts/periodic/submit_metadata.sh
echo 'Done.'
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_wipe_remote.sh
echo 'Done.'

#source $SMARTFLAT_ROOT/api/scripts/periodic/submit_collect_outputs.sh
#echo 'Done.'
#source $SMARTFLAT_ROOT/api/scripts/periodic/submit_distribute_outputs.sh
#echo 'Finished'
#source $SMARTFLAT_ROOT/api/scripts/periodic/submit_print_compute_log.sh 
