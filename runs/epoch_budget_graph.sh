#!/bin/bash
#SBATCH --job-name=mvg-epoch-budget-graph
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
set -euo pipefail

# Epoch-budget ablation, station-network (`graph`) task only.
#
#   sbatch runs/epoch_budget_graph.sh          # the whole ladder, resume-safe
#
# This wrapper exists for one reason: **the card**. `runs/epoch_budget.sh` asks
# for `--gres=gpu:GF2080Ti:1`, which is right for the four univariate ETT sets
# (well under 1 GB) and fatal for this one. Every euppbench method peaks at
# ~15.9 GB allocated at `batch_size: 8`, against the 2080 Ti's 11 GB, so `graph`
# OOMs on its first cell -- and because runs/epoch_budget.sh runs `set -e`, that
# OOM takes the whole job down with it rather than skipping the task.
#
# `RTX3090` (24 GB) is the pin. `--gres=gpu:RTXA6000:1` also works; the card
# only moves `wall_s`, never a score, and every row records its own `gpu`.
#
# SBATCH directives cannot read the environment (they are parsed before the shell
# runs), which is why this is a separate file rather than a variable in
# runs/epoch_budget.sh. Everything else is delegated: this script only sets the
# defaults and execs that one, so the ladder, the arms, the frac=0.1 exclusion
# and the resume logic all live in a single place.
#
# Cost: ~15 GPU-h per seed on a 24 GB card for the full 9-rung ladder over both
# arms, so ~75 h for five seeds. That does not fit the walltime above from
# scratch -- it does not need to, see below.
#
# ---------------------------------------------------------------------------
# Resume, and why the default is the whole ladder
#
# `src/evaluation/epoch_budget.py` skips any cell already in
# logs/epoch_budget/scores.jsonl, and it checks *before* importing Lightning and
# building the model, so a skipped cell costs a few seconds rather than a
# training run. Re-running the full ladder is therefore the safe default: it
# trains exactly what is missing (all 90 cells = 2 arms x 9 rungs x 5 seeds when
# complete). Never pass `epoch_budget.overwrite=true` to "make sure" -- that
# retrains all 90 cells.
#
# To target a gap directly instead of walking the ladder, e.g.:
#
#   FRACS=1.0 ARMS=ours SEEDS=3,4,5 sbatch runs/epoch_budget_graph.sh
#
# and to run a ladder from scratch, split it by seed rather than raising the
# walltime -- the rows are keyed per cell, so five jobs cannot collide:
#
#   for s in 1 2 3 4 5; do
#     SEEDS=$s sbatch --time=24:00:00 runs/epoch_budget_graph.sh
#   done
#
# Note `SEEDS` is a comma-separated Hydra multirun sweep (`1,2,3,4,5`), not a
# space-separated shell list -- it is passed straight to `seed=`. `FRACS` and
# `ARMS` are space-separated. An explicit `FRACS` overrides the per-task ladder,
# so do **not** pass `0.1` here: euppbench refuses it (E = 4 -> a 3+1 split, and
# a 1-epoch stage 2 never anneals after warmup clamping) and the job aborts
# instead of dropping the rung.
# ---------------------------------------------------------------------------

# Same `.project-root` walk as runs/epoch_budget.sh: under sbatch, `$0` is
# Slurm's copy in the spool directory, so a relative `runs/...` path would
# resolve there and not in the repo.
root=$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" && pwd)
while [ ! -e "$root/.project-root" ] && [ "$root" != "/" ]; do
  root=$(dirname "$root")
done
if [ ! -e "$root/.project-root" ]; then
  echo "cannot find the repo root (.project-root) from ${SLURM_SUBMIT_DIR:-$(dirname "$0")}" >&2
  exit 1
fi
cd "$root"

echo "Job:    ${SLURM_JOB_ID:-none} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Guard the reason this wrapper exists: if the scheduler handed us a card that
# cannot hold ~15.9 GB, say so now instead of OOMing on the first cell and
# taking the job down 20 minutes later with nothing written.
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ -n "${total_mib:-}" ] && [ "$total_mib" -lt 20000 ]; then
  echo "This GPU has ${total_mib} MiB; the euppbench arms need ~15.9 GB plus" >&2
  echo "headroom. Resubmit with --gres=gpu:RTX3090:1 or --gres=gpu:RTXA6000:1." >&2
  exit 1
fi

# Delegated so the ladder and the resume logic stay in one place. PYTHONNOUSERSITE
# keeps ~/.local out of sys.path, so user-site packages cannot shadow the
# environment's.
export PYTHONNOUSERSITE=1
exec bash runs/epoch_budget.sh graph
