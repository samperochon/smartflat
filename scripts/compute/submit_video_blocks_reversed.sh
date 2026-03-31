#!/bin/bash

# Set conda environment
conda activate videomae 

# Run script
python $SMARTFLAT_ROOT/api/features/video/main.py --cuda '1' --reversed
