#!/bin/bash
echo "Local machine data transfer"
echo "1) Collect (i) the metadata state accross vm and (ii) extracted fatures to store in the smartflat harddrive"
echo "2) Distribute the newly computed features to the dataset"
echo "3) Update and propagate local metadata state"
echo "4) Create backup of the smartflat persistent_dataframe to the Mac"
echo "5) Send backup of the smartflat features (dump) to @pomme and @cheetah"
echo "6) Bring and update the local frozen-metrics-logs to the smartflat dataset"
echo "7) Rapatriate the results of the change-point-detection algorithms and clustering"
echo "8) Bring back of the light version of the dataset to the local machine"

# 1) Collect from the remotes

# a. The metadata states
echo "1) Collect (i) the metadata state accross vm and (ii) extracted fatures to store in the smartflat harddrive"
#scp cheetah:/diskA/sam_data/data-gold-final/dataframes/persistent_metadata/cheetah_dataset_df.csv /Volumes/Smartflat/data-gold/dataframes/persistent_metadata
scp ruche:/gpfs/workdir/perochons/data-gold-final/dataframes/persistent_metadata/ruche_dataset_df.csv /Volumes/Smartflat/data-gold-final/dataframes/persistent_metadata
#scp pomme:/home/perochon/data-gold-final/dataframes/persistent_metadata/pomme_dataset_df.csv /Volumes/Smartflat/data-gold/dataframes/persistent_metadata

# b. the extracted features
#rsync -ahuvz --progress cheetah:/diskA/sam_data/data-gold-final/dump /Volumes/Smartflat/data-gold
rsync -ahuvz --progress ruche:/gpfs/workdir/perochons/data-gold-final/dump /Volumes/Smartflat/data-gold-final
rsync -ahuvz --progress ruche:/gpfs/workdir/perochons/data-gold-light/dump /Volumes/Smartflat/data-gold-final
rsync -ahuvz --progress ruche:/gpfs/workdir/perochons/data-gold-final/dump /Volumes/Smartflat/data-gold-final


# 2) Distribute the newly computed features to the dataset (by default the input dump and output folders are the default ones of get_data_root()
echo "2) Distribute the newly computed features to the dataset"
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_distribute_outputs.sh

# 3) Update and propagate local metadata state
echo "3) Update and propagate local metadata state"

source $SMARTFLAT_ROOT/api/scripts/periodic/submit_metadata.sh

scp /Volumes/Smartflat/data-gold-final/dataframes/persistent_metadata/gold_dataset_df.csv pomme:/home/perochon/data-gold-final/dataframes/persistent_metadata/
scp /Volumes/Smartflat/data-gold-final/dataframes/persistent_metadata/gold_dataset_df.csv ruche:/gpfs/workdir/perochons/data-gold-final/dataframes/persistent_metadata/
scp /Volumes/Smartflat/data-gold-final/dataframes/persistent_metadata/gold_dataset_df.csv cheetah:/diskA/sam_data/data-gold-final/dataframes/persistent_metadata


# 4) Create backup of the smartflat persistent_dataframe to the Mac
echo "4) Create backup of the smartflat persistent_dataframe to the Mac"
rsync -ahuvzL --progress /Volumes/Smartflat/data-gold-final/dataframes/persistent_metadata /Users/samperochon/Borelli/data/dataframes/


# 5) Send backup of the smartflat features (dump) to @pomme and @cheetah
echo "5) Send backup of the smartflat features (dump) to @pomme and @cheetah"
rsync -ahuvz --progress  /Volumes/Smartflat/data-gold-final/dump cheetah:/diskA/sam_data/data-gold-final
rsync -ahuvz --progress  /Volumes/Smartflat/data-gold-final/dump pomme:/home/perochon/data-gold-final/

# 6) Bring and update the local frozen-metrics-logs to the smartflat dataset
# Note: Possibility to use the script api/scripts/consolidation/rename_compute_files_with_machine_name.sh
scp -r ruche:'/gpfs/workdir/perochons/data-gold-final/dataframes/frozen-metrics-logs/*' /Volumes/Smartflat/data-gold/dataframes/frozen-metrics-logs/ruche/

# 6) Temporary -  Synchronize the persistent_metadata (gold) from the smartflat hard-drive to the virtual machines
#rsync -ahuvz --progress --chmod 777 /Volumes/Smartflat/data-gold-final pomme:/home/perochon/
/
#TODO ? :scp pomme:/home/perochon/data-gold/dataframes/persistent_metadata/pomme_dataset_df.csv  /Users/samperochon/Borelli/data/dataframes/



# 7) Rapatriate the results of the change-point-detection algorithms and clustering
echo "7) Rapatriate the results of the change-point-detection algorithms and clustering"

rsync -ahuvz --progress cheetah:/diskA/sam_data/data-gold-final/experiments/change-point-detection-experiment /Volumes/Smartflat/data-gold-final/experiments
rsync -ahuvz --progress cheetah:/diskA/sam_data/data-gold-final/experiments/change-point-detection-deployment /Volumes/Smartflat/data-gold-final/experiments
rsync -ahuvz --progress cheetah:/diskA/sam_data/data-gold-final/experiments/change-point-detection-deployment-calibrated /Volumes/Smartflat/data-gold-final/experiments

rsync -ahuvz --progress cheetah:/diskA/sam_data/data-gold-final/outputs/change-point-detection-experiment /Volumes/Smartflat/data-gold-final/outputs
rsync -ahuvz --progress cheetah:/diskA/sam_data/data-gold-final/outputs/change-point-detection-deployment /Volumes/Smartflat/data-gold-final/outputs
rsync -ahuvz --progress cheetah:/diskA/sam_data/data-gold-final/outputs/change-point-detection-deployment-calibrated /Volumes/Smartflat/data-gold-final/outputs


#echo "8) Bring back of the light version of the dataset to the local machine"
#rsync -ahuvz --progress ruche:/gpfs/workdir/perochons/data-gold-light /Volumes/Smartflat/

# TODO: add total transfer of the dataset to a remote machine (include finalized dataset-gold and /persistent-metadata)
#cd /Volumes/Smartflat/data-gold/dataframes/persistent_metadata && scp * ruche:/gpfs/workdir/perochons/data-gold/dataframes/persistent_metadata/
#cd /Volumes/Smartflat/data-gold/dataframes/persistent_metadata && scp * cheetah:/diskA/sam_data/data-gold/dataframes/persistent_metadata/


# 2) Back-up experiments for local usage etc
#rsync -ahuvz --progress cheetah:/diskA/sam_data/data/experiments/\*deployment\* /Volumes/Smartflat/data/experiments/

# 3) Backup outputs (/!\ need to be created in the local machines)

# Backup Dataset outputs to cheetah
#rsync -ahuvz --progress /Volumes/Smartflat/data/dump/'*' cheetah:/diskA/sam_data/data/dump/
#rsync -ahuvz --progress /Volumes/Smartflat/data/dump/'*' pomme:/home/perochon/data/dump/
