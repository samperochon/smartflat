#!/bin/bash

# Module load
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment code
source activate smartflat 

# Run script
python $SMARTFLAT_ROOT/api/features/consolidation/main_init_folder_structure.py --path_metadata /gpfs/workdir/perochons/data-gold-final/dataframes/persistent_metadata/feature_extraction_dataset_df.csv
