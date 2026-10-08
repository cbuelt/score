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

# ERA5 surface temperature (U-Cast backbone), single-stage baseline: LorD fitted
# with the NLL. The other baselines use the same experiment with a different
# method, e.g.
#   python src/train/train.py experiments=train/era5 method=diag method.loss=crps
# Any extra args passed to sbatch are forwarded to Hydra, e.g.
#   sbatch runs/train/era5.sh method.rank=4 seed=1
python src/train/train.py experiments=train/era5 method=lord method.loss=nll "$@"
