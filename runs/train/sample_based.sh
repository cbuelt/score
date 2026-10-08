#!/bin/bash
#SBATCH --job-name=mvg-generative
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
set -euo pipefail

# SB, the sample-based generative baseline, fitted with the fair Gaussian kernel
# score.
#
# The backbone maps (input, latent) -> one sample; M = n_train_samples latents
# per case give the ensemble the score is computed from, and the fair
# (1/(M(M-1))) spread term is what keeps a small M from being rewarded for
# under-dispersion. This is the nonparametric reference: no distributional
# assumption, and correspondingly no log score -- NLL is reported as inf and
# renders as an empty cell in the tables.
#
# The default objective is the kernel score, at the same gamma ladder the
# Gaussian families use, so SB is fitted on the metric the tables report
# rather than on a second objective. For the energy-score variant the
# generative-forecasting literature uses, append `method.loss=energy_score` --
# it gets its own run slug, and the two must not share a results table.
#
# Cost model: the trunk runs once per case and only the head is replayed per
# member, so a step costs roughly (trunk + M * head) rather than M * model.
# Where the head is the expensive part -- depth, at full resolution -- lower
# trainer/datamodule batch size rather than M.
#
# This script covers the station and ETT tasks. The gridded ones are heavier
# and are launched separately, e.g.
#   python src/train/train.py experiments=train/era5 method=sample_based
#   python src/train/train.py experiments=train/depth method=sample_based
#
# Any extra args are forwarded to Hydra, e.g.
#   sbatch runs/train/sample_based.sh trainer.max_epochs=1 seed=1

# Machine-specific environment (conda activation, data locations); see runs/env.sh.example.
[ -f runs/env.sh ] && source runs/env.sh

echo "Job:         ${SLURM_JOB_ID:-none} on $(hostname)"
echo "Python:      $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Station post-processing (multigraph). Targets have gaps; the ensemble
# estimators mask them exactly, unlike the sliced Gaussian estimator the
# closed-form families use.
python src/train/train.py -m \
    experiments=train/euppbench \
    method=sample_based \
    seed=1,2,3,4,42 "$@"

# Univariate and multivariate ETT horizons (PatchTST).
python src/train/train.py -m \
    experiments=train/time_series/univariate/etth1,train/time_series/univariate/etth2,train/time_series/univariate/ettm1,train/time_series/univariate/ettm2 \
    method=sample_based \
    seed=1,2,3,4,42 "$@"

python src/train/train.py -m \
    experiments=train/time_series/multivariate/etth1,train/time_series/multivariate/etth2,train/time_series/multivariate/ettm1,train/time_series/multivariate/ettm2 \
    method=sample_based \
    seed=1,2,3,4,42 "$@"
