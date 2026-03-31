#!/bin/bash
# First part: Slurm directives to reserve nodes, memory and time
#SBATCH --job-name=resolution_reduction_extra_cpus
#SBATCH --output=%x.o
#SBATCH --error=%x.err
#SBATCH --mail-type=END
#SBATCH --time=72:00:00
#SBATCH --ntasks=160
#SBATCH --partition=cpu_long

# Second part: Unix script to execute the code
# Load necessary modules
module load python/3.9.10/gcc-11.2.0
module load miniconda3/4.10.3/gcc-13.2.0 #module load miniconda3/23.5.2/gcc-13.2.0
module load ffmpeg/4.3.2/gcc-11.2.0

# Third part: code to run on the cluster machine 

input_dir=/gpfs/workdir/perochons/data-gold-final  #$1
output_dir=/gpfs/workdir/perochons/data-gold-light #$2 

# Activate conda environment
source activate smartflat 
echo "Environment setup completed" >> ~/setup_log.txt

# Run conversion
time python3 $SMARTFLAT_ROOT/api/features/consolidation/main_conversion.py "$input_dir" "$output_dir"