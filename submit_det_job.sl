#!/bin/bash
#SBATCH --job-name=ViT_NeSI
#SBATCH --account=nesi99999         # <--- CHANGE THIS to your project code
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=vit_output_%j.log

# 1. Load the Miniconda module
module purge
module load Miniconda3

# 2. Activate conda environment using the subshell workaround
# This ensures the shell functions are available for 'conda activate'
source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate  # Workaround for activation issues in some Slurm environments

# 3. Target your specific environment path
conda activate /nesi/project/nesi99999/thomas.li/pytorch_ml

# module load Python/3.10.5-gimkl-2022a
# module load CUDA/11.7.0

# Run the script
srun python det_train.py
