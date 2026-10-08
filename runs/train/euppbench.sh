#!/bin/bash
#SBATCH --job-name=euppbench
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00

# EUPPBench station post-processing (d = 115 stations, 72 h lead time), all
# methods over five seeds. Any extra args are forwarded to Hydra verbatim.

# Machine-specific environment (conda activation, data locations); see runs/env.sh.example.
[ -f runs/env.sh ] && source runs/env.sh


# SCORE (two-stage fit, graph Fourier transform with Matern filter)
python src/train/train.py --multirun experiments=train/euppbench_score seed=1,2,3,4,5 "$@"
# Chol
python src/train/train.py --multirun experiments=train/euppbench method=cholesky method.loss=nll,kernel_score seed=1,2,3,4,5 "$@"
# SB (sample-based generative baseline, fair kernel score)
python src/train/train.py --multirun experiments=train/euppbench method=sample_based seed=1,2,3,4,5 "$@"
# Det
python src/train/train.py --multirun experiments=train/euppbench method=deterministic method.loss=mse seed=1,2,3,4,5 "$@"
# LorD (r = 1, 2, 4, 8, 16)
python src/train/train.py --multirun experiments=train/euppbench method=lord method.loss=nll,kernel_score method.rank=1,2,4,8,16 seed=1,2,3,4,5 "$@"
python src/train/train.py --multirun experiments=train/euppbench method=lord method.loss=kernel_score method.rank=8,16 seed=1,2,3,4,5 "$@"
# Diag
python src/train/train.py --multirun experiments=train/euppbench method=diag method.loss=nll,kernel_score seed=1,2,3,4,5 "$@"


