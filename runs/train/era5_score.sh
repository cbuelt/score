#!/bin/bash
#SBATCH --job-name=mvg-era5-debug
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --time=72:00:00

set -euo pipefail


# Machine-specific environment (conda activation, data locations); see runs/env.sh.example.
[ -f runs/env.sh ] && source runs/env.sh

echo "Job:         $SLURM_JOB_ID on $(hostname)"
echo "Python:      $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ERA5 surface temperature (U-Cast backbone), SCORE with the two-stage fit.
# Any extra args passed to sbatch are forwarded to Hydra, e.g.
#   sbatch runs/train/era5_score.sh seed=1
python src/train/train.py experiments=train/era5_score "$@"
