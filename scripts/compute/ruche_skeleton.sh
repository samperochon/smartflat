#!/bin/bash
#SBATCH --job-name=skeleton_inference
#SBATCH --output=%x.o
#SBATCH --error=%x.err
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --time=72:00:00
#SBATCH --ntasks=80
#SBATCH --gres=gpu:0
#SBATCH --partition=mem

# Module load
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment codecl
source activate mediapipe

# Train the network
time python3 $SMARTFLAT_ROOT/api/features/skeleton/main.py 
