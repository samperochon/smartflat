#!/bin/bash
#SBATCH --job-name=video_3
#SBATCH --output=%x.o
#SBATCH --error=%x.err
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100

# Module load
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199


# Activate anaconda environment codecl
source activate smartflat

# Train the network
time python3 $SMARTFLAT_ROOT/api/features/video/main.py --chunk_idx 3 --num_chunks 6
