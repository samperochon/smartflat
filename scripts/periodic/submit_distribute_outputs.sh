#!/bin/bash

# Set conda environment
conda activate temporal_segmentation 

# Run script
python $SMARTFLAT_ROOT/api/features/consolidation/main_distribute_outputs.py #-p