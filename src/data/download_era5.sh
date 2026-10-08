#!/bin/bash
# SLURM wrapper around download_era5.py; the download itself needs no GPU,
# adjust the directives below to your cluster. Arguments are passed through.
#SBATCH --job-name=download_era5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX3090
#SBATCH --time=72:00:00

[ -f runs/env.sh ] && source runs/env.sh
python src/data/download_era5.py "$@"