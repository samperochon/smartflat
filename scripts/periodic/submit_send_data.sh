#!/bin/bash

# Set conda environment
conda activate temporal_segmentation 

# Run script
python $SMARTFLAT_ROOT/api/features/consolidation/main_send_data.py --remote_name ruche --process #'cheetah' or 'ruche'



