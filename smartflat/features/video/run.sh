#!/bin/bash

# Set the name of your Conda environment
CONDA_ENV_NAME="videomae"

# Activate the Conda environment
source activate "$CONDA_ENV_NAME"


# Run your Python script:
python /home/perochon/temporal_segmentation/contrib/VideoMAEv2/main.py

# Deactivate the Conda environment (optional)
conda deactivate


