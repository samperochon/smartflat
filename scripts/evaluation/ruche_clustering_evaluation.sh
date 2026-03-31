#!/bin/bash
#SBATCH --job-name=clustering_evaluation
#SBATCH --output=%x.o
#SBATCH --error=%x.err
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --time=4:00:00
#SBATCH --ntasks=1000
#SBATCH --partition=cpu_med

# Module load
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment code
source activate smartflat

# Run evaluation
time python3 $SMARTFLAT_ROOT/api/engine/clustering_evaluation.py --config_name ClusteringAllConfig 

# #SBATCH --gres=gpu: