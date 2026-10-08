#!/bin/bash
#SBATCH --job-name=ettm2-uni
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00

# ETTm2, univariate (features=S, C = 1).
# Same methods, losses, ranks and seeds as runs/train/euppbench.sh except Chol,
# which is not run on ETTm; swept over both reported horizons (T96/192).
#
# Both sweeps are taken from the environment (as in runs/benchmark.sh), so the
# job can be split when it does not fit the walltime:
#   HORIZONS=192 sbatch runs/train/ettm2_uni.sh
#   SEEDS=2019,2020,2021,2022,2023 sbatch runs/train/ettm2_uni.sh
# Any extra args are forwarded to Hydra verbatim.
#
# Deliberately no `set -e`: the runs are independent, and one that dies (an
# OOM, or a method that refuses the event dimension) must not take the rest of
# the grid with it. Check the log for which `python` lines actually finished.

# Machine-specific environment (conda activation, data locations); see runs/env.sh.example.
[ -f runs/env.sh ] && source runs/env.sh

echo "Job:         ${SLURM_JOB_ID:-none} on $(hostname)"
echo "Python:      $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXP=train/time_series/univariate/ettm2
EXP_SCORE=train/time_series/univariate/ettm2_score
HORIZONS=${HORIZONS:-96,192}
SEEDS=${SEEDS:-1,2,3,4,5}


# SCORE (two-stage fit)
python src/train/train.py --multirun experiments="$EXP_SCORE" datamodule.pred_len="$HORIZONS" seed="$SEEDS" "$@"
# SB (sample-based generative baseline, fair kernel score)
python src/train/train.py --multirun experiments="$EXP" method=sample_based datamodule.pred_len="$HORIZONS" seed="$SEEDS" "$@"
# Det
python src/train/train.py --multirun experiments="$EXP" method=deterministic method.loss=mse datamodule.pred_len="$HORIZONS" seed="$SEEDS" "$@"
# LorD (r = 1, 2, 4, 8, 16)
python src/train/train.py --multirun experiments="$EXP" method=lord method.loss=nll,kernel_score method.rank=1,2,4,8,16 datamodule.pred_len="$HORIZONS" seed="$SEEDS" "$@"
# Diag
python src/train/train.py --multirun experiments="$EXP" method=diag method.loss=nll,kernel_score datamodule.pred_len="$HORIZONS" seed="$SEEDS" "$@"
