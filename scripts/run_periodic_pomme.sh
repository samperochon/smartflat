#!/bin/bash
echo "Pomme machine data transfer"
echo "1) Collect (i) the ruche metadata and (ii) dump features"
echo "2) Distribute the newly computed features to the dataset"
echo "3) Update metadata state"


# 1) Collect from the remotes

# a. The metadata states

scp ruche:/gpfs/workdir/perochons/data-gold-final/dataframes/persistent_metadata/ruche_dataset_df.csv /home/perochon/data-gold-final/dataframes/persistent_metadata

# b. the extracted features
rsync -ahuvz --progress ruche:/gpfs/workdir/perochons/data-gold-final/dump /home/perochon/data-gold-final

# 2) Distribute the newly computed features to the dataset (by default the input dump and output folders are the default ones of get_data_root()
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_distribute_outputs.sh


# 3) Update and propagate local metadata state

source $SMARTFLAT_ROOT/api/scripts/periodic/submit_metadata.sh