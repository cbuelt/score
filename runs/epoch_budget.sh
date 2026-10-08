#!/bin/bash
#SBATCH --job-name=mvg-epoch-budget
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GF2080Ti:1
#SBATCH --time=72:00:00
set -euo pipefail

# Epoch-budget ablation: is SCORE's gain due to the correlation core, or to the
# extra epochs its two-stage fit spends?
#
# At each rung of the ladder, two arms are trained at the *same* total epoch
# budget E and scored on the test set:
#
#   marginal  a diagonal Gaussian (Diag) on the marginal CRPS for the full E
#             epochs (`method=diag method.loss=crps`, which is the objective
#             SCORE's stage 1 optimizes, and the reference model of Feik et al.
#             on EUPPBench)
#   ours      SCORE: the same backbone, two-stage fit, 3E/4 on stage 1
#             (marginals) + E/4 on stage 2 (core)
#
# Submitted like runs/benchmark.sh -- one self-contained sbatch script, tasks as
# positional arguments, everything else through the environment:
#
#   sbatch runs/epoch_budget.sh                            # all default tasks
#   sbatch runs/epoch_budget.sh etth1_uni etth2_uni ettm1_uni   # the ETT ladders
#   FRACS="0.2 0.4" sbatch runs/epoch_budget.sh graph      # half the ladder
#   ARMS=ours sbatch runs/epoch_budget.sh etth1_uni        # one arm
#   SEEDS=1 bash runs/epoch_budget.sh etth1_uni            # one seed, smoke test
#
# Submit it from the repo root if your sbatch setup writes logs to a relative
# `--output` path (e.g. `logs/epoch_budget/`): sbatch refuses the job if that
# directory does not exist. The script itself locates the repo by the
# `.project-root` marker (see below).
#
# Splitting the job. All default tasks take ~58 GPU-h (see cost below), which
# fits 72 h with little margin, and EUPPBench is 52 of those hours. Seeds are
# disjoint sets of cells and each row is one short O_APPEND write to the shared
# jsonl, so the graph ladder parallelises safely across jobs:
#
#   sbatch runs/epoch_budget.sh etth1_uni etth2_uni ettm1_uni
#   for s in 1 2 3 4 5; do SEEDS=$s sbatch --time=24:00:00 runs/epoch_budget.sh graph; done
#
# The default task order puts the cheap ETT ladders first and EUPPBench last, so
# a job that hits the walltime loses graph rungs (resumable, below) rather than
# the ladders that were nearly free.
#
# --- the ladder -----------------------------------------------------------
# The rung is passed as a *fraction* of the experiment's own
# `trainer.max_epochs`, never as an epoch count, so the ladder stays anchored to
# the main-experiment configuration: at frac=1.0 the two-stage split is exactly
# the one the experiment config carries (euppbench 40 -> 30+10, ETT 100 -> 75+25),
# which makes that rung a reproducibility check on the whole pipeline. At the
# default 0.1 spacing:
#
#   ETT univariate  10 20 30 40 50 60 70 80 90 100
#                    splits 8+2 15+5 23+7 30+10 38+12 45+15 53+17 60+20 68+22 75+25
#   euppbench        8 12 16 20 24 28 32 36 40      (frac=0.1 is ETT-only)
#                    splits 6+2 9+3 12+4 15+5 18+6 21+7 24+8 27+9 30+10
#
# `frac=0.1` is refused on euppbench: 0.1 of 40 epochs is 4, i.e. 3+1, and a
# 1-epoch stage 2 is pure warmup (subtractive, clamped to `stage2 - 1`) so its LR
# never anneals. `resolve_budget` raises rather than train a rung on a different
# recipe, which would abort this script -- hence TASK_FRACS below gives graph its
# own ladder starting at 0.2.
# Five of the ten ETT rungs are not multiples of 4, so their 3:1 split is rounded
# by an epoch (80:20 at E=10, under a point from E=50 up); both arms of a rung
# always get the same total E, and the realised split is recorded per row.
#
# --- resuming and overwriting ---------------------------------------------
# Cells already in logs/epoch_budget/scores.jsonl are skipped, so a job killed by
# the scheduler is resumed by resubmitting it unchanged -- never with
# `epoch_budget.overwrite=true`, which would make a requeued job retrain every
# cell it touches and start the 24 h EUPPBench jobs from zero. To start from a
# clean record instead, archive the file once, before submitting:
#
#   mv logs/epoch_budget/scores.jsonl logs/epoch_budget/scores.jsonl.superseded-$(date +%Y%m%d-%H%M%S)
#
# Mixing ladders needs no archiving: the rungs two spacings share (0.2/0.4/0.6/
# 0.8/1.0) resolve to the same budgets and splits, so those cells are skipped.
#
# --- cost -----------------------------------------------------------------
# Approximate, on an RTX 4090 (both arms, five seeds; 100 cells per ETT data set,
# 90 for graph):
#
#   etth1_uni / etth2_uni   ~1.1 GPU-h each
#   ettm1_uni / ettm2_uni   ~4.3 GPU-h each
#   graph                   ~10-15 GPU-h PER SEED (90 cells) on a 24 GB card
#
# The GF2080Ti this script asks for is ~2.6x slower than a 4090, so budget ~11 h
# for an ETTm ladder here. EUPPBench does not fit that card at all: every
# euppbench method peaks at ~15.9 GB allocated (stage 2 alone at 9.0) against
# its 11 GB, so `graph` OOMs on its first cell and `set -e` takes the job with
# it. Use runs/epoch_budget_graph.sh (24 GB card) for graph, rather than dropping
# `datamodule.batch_size` -- which would change the recipe the 1.0 rung is
# supposed to reproduce.
#
# ETTm is 69,680 rows against ETTh's 17,420 and costs ~3.9x per cell -- the ratio
# climbs with the rung (3.2x at frac=0.2, ~4.1x from 0.6 up) as the fixed
# per-epoch overhead stops dominating.
#
# W&B is off by default (configs/epoch_budget.yaml): these few hundred runs carry
# the same tags as the main runs, and the score tables select W&B runs by those
# tags, so logging them would add spurious seeds at the wrong budgets to every
# table. The scores come back from `train()` directly instead.

# Repo root. Under sbatch, `$0` is Slurm's *copy* of this script in the spool
# directory, so `dirname $0/..` lands in the spool, not the repo -- the relative
# `src/...` path below then does not exist. Slurm starts the job in the
# submitting directory instead, which it also exports as SLURM_SUBMIT_DIR; walk
# up from that (or from this file, when run as a plain `bash
# runs/epoch_budget.sh`) to the `.project-root` marker, the same anchor
# src/evaluation/epoch_budget.py bootstraps from. That way the script also works
# when submitted from a subdirectory.
root=$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" && pwd)
while [ ! -e "$root/.project-root" ] && [ "$root" != "/" ]; do
  root=$(dirname "$root")
done
if [ ! -e "$root/.project-root" ]; then
  echo "cannot find the repo root (.project-root) from ${SLURM_SUBMIT_DIR:-$(dirname "$0")}" >&2
  exit 1
fi
cd "$root"

# Machine-specific environment (conda activation, data locations); see runs/env.sh.example.
[ -f runs/env.sh ] && source runs/env.sh

echo "Job:         ${SLURM_JOB_ID:-none} on $(hostname)"
echo "Python:      $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PY=python
ABLATE="$PY src/evaluation/epoch_budget.py"

# Single-stage experiment (the `marginal` arm) and the SCORE experiment (`ours`).
# Same task nicknames as runs/benchmark.sh, so the cost rows and the score rows
# of one task are named alike.
declare -A EXP=(
  [graph]=train/euppbench
  [etth1_uni]=train/time_series/univariate/etth1
  [etth2_uni]=train/time_series/univariate/etth2
  [ettm1_uni]=train/time_series/univariate/ettm1
  [ettm2_uni]=train/time_series/univariate/ettm2
)
declare -A EXP_SCORE=(
  [graph]=train/euppbench_score
  [etth1_uni]=train/time_series/univariate/etth1_score
  [etth2_uni]=train/time_series/univariate/etth2_score
  [ettm1_uni]=train/time_series/univariate/ettm1_score
  [ettm2_uni]=train/time_series/univariate/ettm2_score
)

# Rungs, as fractions of each experiment's configured budget. 1.0 is the
# main-experiment configuration and is what ties the ladder to the reported
# numbers.
FRACS=${FRACS:-}
DEFAULT_FRACS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
# Per-task ladder, for a task that cannot train the default one: euppbench refuses
# frac=0.1 outright (see above), so passing it would abort the job rather than
# skip the rung. An explicit `FRACS=...` wins over this table -- unlike
# benchmark.sh's TASK_HORIZONS, because here the override is how half a ladder
# gets resumed in a second job.
declare -A TASK_FRACS=(
  [graph]="0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
)

# Five seeds, as in runs/train/euppbench.sh. Five makes the spread a sample
# standard deviation worth quoting rather than a range.
SEEDS=${SEEDS:-"1,2,3,4,5"}

# Both arms by default; a single arm is useful for resuming half a sweep.
ARMS=${ARMS:-"marginal ours"}

# Extra Hydra overrides, forwarded verbatim to every cell, e.g.
#   EXTRA="datamodule.num_workers=8" sbatch runs/epoch_budget.sh graph
# An env var rather than "$@" because the positional arguments of this script are
# task names: forwarding them would hand Hydra `graph` as an override.
EXTRA=${EXTRA:-}

TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  # All four univariate ETT sets, then the station network. Unlike the runtime
  # benchmark, where a second set of the same shape re-measures the same step
  # cost, this study reads a *score* against the budget, one ladder per data
  # set: whether SCORE's joint-score gain survives at every rung is a claim
  # about the task, not about the tensor shapes, so every data set is a separate
  # test of it.
  TASKS=(etth1_uni etth2_uni ettm1_uni ettm2_uni graph)
fi

echo "tasks: ${TASKS[*]} | seeds: $SEEDS | arms: $ARMS"

for task in "${TASKS[@]}"; do
  if [ -z "${EXP[$task]:-}" ]; then
    echo "unknown task '$task' (expected one of: ${!EXP[*]})" >&2
    exit 1
  fi
  fracs=${FRACS:-${TASK_FRACS[$task]:-$DEFAULT_FRACS}}
  echo "--- $task, rungs: $fracs"

  for f in $fracs; do
    for arm in $ARMS; do
      echo "===== $task arm=$arm frac=$f seeds=$SEEDS ====="
      case "$arm" in
        marginal)
          # Single stage: `trainer.max_epochs` is the knob the driver sets. The
          # CRPS loss is explicit because the plain time-series experiments do
          # not override /method at all (they inherit `deterministic` from
          # configs/train.yaml), and because the main Diag runs use
          # nll/kernel_score -- this arm has to match stage 1's objective, not
          # the reported Diag rows.
          $ABLATE -m experiments="${EXP[$task]}" \
              method=diag method.loss=crps \
              epoch_budget.arm=marginal epoch_budget.frac="$f" \
              seed="$SEEDS" $EXTRA
          ;;
        ours)
          # SCORE two-stage fit: the per-stage ceilings are the knob;
          # trainer.max_epochs is inert here (see train._build_stage_trainer).
          $ABLATE -m experiments="${EXP_SCORE[$task]}" \
              epoch_budget.arm=ours epoch_budget.frac="$f" \
              seed="$SEEDS" $EXTRA
          ;;
        *)
          echo "unknown arm '$arm' (expected 'marginal' or 'ours')" >&2
          exit 1
          ;;
      esac
    done
  done
done

echo "===== done ====="
echo "Rows are in logs/epoch_budget/scores.jsonl."
echo "Open notebooks/epoch_budget.ipynb to aggregate them and draw the figure."
