#!/bin/bash
#SBATCH --job-name=speech_inference
#SBATCH --output=%x.o
#SBATCH --error=%x.err
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --time=24:00:00
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --partition=gpua100

#gpua100

# Module load
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/11.8.0/gcc-11.2.0
module load ffmpeg/4.3.2/gcc-11.2.0

# Activate anaconda environment codecl
source activate whisperx

# Temporary
#python -m pytorch_lightning.utilities.upgrade_checkpoint .cache/torch/whisperx-vad-segmentation.bin

# Train the network
time python3 $SMARTFLAT_ROOT/api/features/audio/main.py 
