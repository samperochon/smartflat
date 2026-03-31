#!/bin/bash
#SBATCH --job-name=hands_inference
#SBATCH --output=%x.o
#SBATCH --error=%x.err
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu_long

# Module load
module load anaconda3/2020.02/gcc-9.2.0

[ ! -d output ] && mkdir output
[ ! -d output ] && mkdir output/hands



# Activate anaconda environment codecl
source activate mediapipe

# Train the network
time python3 $SMARTFLAT_ROOT/api/features/hands/main.py 
