#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12G
#SBATCH --time=4:00:00
#SBATCH --job-name="test"
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/S-%x.%j.out

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source .venv/bin/activate

python3 src/train.py