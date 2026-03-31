#!/bin/bash
echo "Cheetah machine data transfer"
# echo "1) Collect (i) the metadata state accross vm and (ii) extracted fatures to store in the smartflat harddrive"
# echo "2) Distribute the newly computed features to the dataset"
echo "1) Get dump from ruche"
#echo "2) Get dataframe updates from ruche"
echo "3) Update metadata state"
echo "# 4) Distribute outputs to the dataset"
echo "3) Update metadata state"



# 1) Get dump from ruche
echo "1) Get dump from ruche"
rsync -ahuvzL --progress ruche:ruche:/gpfs/workdir/perochons/data-gold-final/dump /diskA/sam_data/data-features/

# 2) Get dataframe updates from ruche 
#echo "2) Get dataframe updates from ruche"
# scp  ruche:/gpfs/workdir/perochons/data-gold-final/dataframes/persistent_metadata/smartflat_video_metadata.csv  /diskA/sam_data/data-features/dataframes/persistent_metadata


# 3) Update and propagate local metadata state
echo "3) Update and propagate local metadata state"
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_metadata.sh


# 4) Distribute outputs to the dataset 
echo "# 4) Distribute outputs to the dataset"
python $SMARTFLAT_ROOT/api/features/consolidation/main_distribute_outputs.py -p


# 5) Update and propagate local metadata state
echo "3) Update and propagate local metadata state"
source $SMARTFLAT_ROOT/api/scripts/periodic/submit_metadata.sh

