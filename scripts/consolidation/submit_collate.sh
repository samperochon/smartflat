#!/bin/bash

# Set conda environment
source activate smartflat 

# Run script
python $SMARTFLAT_ROOT/api/features/consolidation/main_collate.py --process

$SMARTFLAT_ROOT/api/features/consolidation/main_collate_clean.py --process
