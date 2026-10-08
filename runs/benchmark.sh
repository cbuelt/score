#!/bin/bash
#SBATCH --job-name=mvg-benchmark
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail

# Per-step training cost of each predictive method, at one pinned batch size per
# task. Every method is fitted with the *kernel score*, SB included
# (method/sample_based.yaml defaults to it; its runs are published under the
# slug `generative_kernel_score`).
#
#   sbatch runs/benchmark.sh                       # the time-series tasks, see TASKS
#   sbatch runs/benchmark.sh graph era5            # named tasks
#   REMEASURE=1 sbatch runs/benchmark.sh etth1     # re-time methods the log already has
#   RANKS=1 sbatch runs/benchmark.sh etth1         # one rank instead of both
#   PRED_LENS=96 sbatch runs/benchmark.sh etth1    # one horizon instead of both
#   PROBE=1 sbatch runs/benchmark.sh era5          # memory probe only, see below
#
# Methods per task: Det (point forecast; the backbone-only floor), Diag, LorD at
# each rank in RANKS, SB (M from method/sample_based.yaml), SCORE (two-stage
# fit), and Chol on the tasks where it is run (univariate ETTh and the station
# graph; not ETTm). Time-series tasks run once per horizon in PRED_LENS, since
# the horizon sets V and therefore the cost.
#
# Tasks: the univariate ETT sets (etth1_uni etth2_uni ettm1_uni ettm2_uni),
# their multivariate counterparts (etth1 etth2 ettm1 ettm2), the wide
# multivariate sets (electricity traffic weather illness), and graph era5 depth.
# `sbatch runs/benchmark.sh` runs the time-series tasks -- see TASKS below.
#
# Methods already in logs/benchmark/runtime.jsonl for every seed are SKIPPED
# (`benchmark.skip_measured=true`), so a resubmitted job resumes where the last
# one stopped instead of appending a second set of rows:
# notebooks/make_runtime_table.py keys on the method and not on the date, so two
# sets of rows for the same seeds would pool into one mean. REMEASURE=1 turns
# the skip off; remove the old rows from the log first if you use it.
#
# Sampling settings come from configs/benchmark.yaml: 5 seeds, 20 warmup steps
# discarded and 100 measured steps per seed per stage, i.e. ~500 timed steps per
# method. Epoch counts are NOT swept -- the benchmark never runs an epoch; it
# reads `trainer.max_epochs` and the stage budgets from the experiment config
# and uses them only to weight the two SCORE stages and to project GPU-hours.
#
# Batch size is pinned per task below rather than taken from the dataset config.
# It has to be a value the *heaviest* method fits, which on the gridded sets is
# SB: its activations carry n_train_samples = 5 members. Run the probe first
# when changing a backbone, a resolution or the member count:
#
#   PROBE=1 bash runs/benchmark.sh era5
#
# which times three steps of SB at a ladder of batch sizes and prints peak
# memory for each, so the pinned value is chosen from a measurement rather than
# from a guess. Budget is one 48 GB card.

# Repo root. Under sbatch, `$0` is Slurm's *copy* of this script in the spool
# directory, so a relative `src/train/...` path would resolve against the spool
# and not the repo unless the job happens to start in the submitting directory.
# Walk up from SLURM_SUBMIT_DIR (or from this file, when run as a plain
# `bash runs/benchmark.sh`) to the `.project-root` marker, so the script also
# works when submitted from a subdirectory.
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

# Keep ~/.local out of sys.path: user-site packages can shadow the environment's
# and make an import succeed against the wrong version rather than fail.
export PYTHONNOUSERSITE=1

echo "Job:         ${SLURM_JOB_ID:-none} on $(hostname)"
echo "Python:      $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PY=python
BENCH="$PY src/train/benchmark.py"

# Pinned batch size per task.
#
# etth1 / graph are MEASURED at their training batch sizes, at which nothing is
# memory-bound (graph peaks at 17.5 GiB reserved for every method, well inside
# a 48 GB card; ETT is under 0.4 GiB).
#
# era5 is EXTRAPOLATED from a two-point memory probe (bs 2 and 4 on a 24 GB
# card); confirm with `PROBE=1 bash runs/benchmark.sh era5` on a 48 GB card
# before quoting a number. Peak reserved memory is close to linear in the batch
# size, and SB is the binding constraint -- it carries n_train_samples = 5
# members through the head:
#
#   era5  SB    M=10 4.44, M=5 3.35 GiB/sample -> bs 8 ~ 27 GiB at M=5
#                                   (bs 16 at M=10 needs ~72 GiB and does NOT
#                                    fit, which is why every method uses bs 8)
#   era5  Diag  2.57 GiB/sample  -> bs 8  ~ 21 GiB
#   elec  Diag  0.45 GiB/sample  -> bs 32 ~ 14 GiB; SB does not fit bs 32 on
#                                   24 GB even at M=5, so run the probe here
#
# depth is MEASURED at the `depth_bts_crop` geometry (see EXTRA below):
# V = 238,560. Probed on a 24 GB card, SB again the ceiling:
#
#   depth SB          bs 1  4.05 | bs 2  6.81 | bs 4 13.48 GiB | bs 8 OOM on 24 GB
#                     -> 3.34 GiB/sample + ~0.1 fixed, so bs 8 ~ 27 GiB, which is
#                        inside the 48 GB budget this script targets
#   depth others @bs4 LorD r16 5.84 | SCORE s1 5.48 / s2 3.03 | Diag 3.38 |
#                     Det 3.38 GiB -- all far under SB
#
# 8 is also the training `datamodule.batch_size` for depth, so the cost row and
# the score row are taken at the same batch size. `ms_per_sample` is flat from
# bs 2 up (62.4 -> 61.8 ms), i.e. the step is compute- not launch-bound, so
# `train_hours` barely moves with the pin -- but `step_ms` does, and it is only
# comparable within one dataset anyway. On a 24 GB card drop to 4; the row
# records `batch_size`.
#
# traffic is MEASURED (24 GB card, both horizons). At the datamodule's bs 24 SB
# does not fit a 48 GB card. Memory is linear in the batch size and the same at
# T96 and T192 -- it is set by the B * 862 token streams through the trunk and
# the head's encoder block, which SB replays once per member, not by V:
#
#   traffic SB          bs 1 4.36 | bs 2 8.64 | bs 4 17.21 GiB reserved
#                       (13.22 allocated) | bs 8 OOM on 24 GB -> 4.30 GiB/sample,
#                       so bs 8 ~ 34 GiB reserved, bs 12 ~ 52 GiB does NOT fit
#   traffic others @bs4 LorD r16 7.63 (T192 8.12) | SCORE s1 7.24 / s2 3.29 |
#                       LorD r1 7.24 | Diag = Det 5.60 GiB
#
# Re-time traffic with runs/benchmark_traffic.sh, which first removes rows
# measured at any other batch size.
declare -A BATCH=(
  [etth1_uni]=128   [etth2_uni]=128   [ettm1_uni]=128   [ettm2_uni]=128
  [etth1]=128       [etth2]=128       [ettm1]=128       [ettm2]=128
  [electricity]=32  [traffic]=8       [weather]=128     [illness]=32
  [graph]=8
  [era5]=8           # extrapolated ~27 GiB for the sample-based arm at M=5
  [depth]=8          # measured ~27 GiB for the sample-based arm at V=238,560
)

# Single-stage experiment (Det, Diag, LorD, SB, Chol) and the SCORE experiment.
declare -A EXP=(
  [etth1_uni]=train/time_series/univariate/etth1
  [etth2_uni]=train/time_series/univariate/etth2
  [ettm1_uni]=train/time_series/univariate/ettm1
  [ettm2_uni]=train/time_series/univariate/ettm2
  [etth1]=train/time_series/multivariate/etth1
  [etth2]=train/time_series/multivariate/etth2
  [ettm1]=train/time_series/multivariate/ettm1
  [ettm2]=train/time_series/multivariate/ettm2
  [electricity]=train/time_series/multivariate/electricity
  [traffic]=train/time_series/multivariate/traffic
  [weather]=train/time_series/multivariate/weather
  [illness]=train/time_series/multivariate/illness
  [graph]=train/euppbench
  [era5]=train/era5
  [depth]=train/depth
)
declare -A EXP_SCORE=(
  [etth1_uni]=train/time_series/univariate/etth1_score
  [etth2_uni]=train/time_series/univariate/etth2_score
  [ettm1_uni]=train/time_series/univariate/ettm1_score
  [ettm2_uni]=train/time_series/univariate/ettm2_score
  [etth1]=train/time_series/multivariate/etth1_score
  [etth2]=train/time_series/multivariate/etth2_score
  [ettm1]=train/time_series/multivariate/ettm1_score
  [ettm2]=train/time_series/multivariate/ettm2_score
  [electricity]=train/time_series/multivariate/electricity_score
  [traffic]=train/time_series/multivariate/traffic_score
  [weather]=train/time_series/multivariate/weather_score
  [illness]=train/time_series/multivariate/illness_score
  [graph]=train/euppbench_score
  [era5]=train/era5_score
  [depth]=train/depth_score
)

# Extra per-task Hydra args.
#
# Depth is pinned to `depth_bts_crop` -- NYU Depth v2 (Eigen split) restricted
# to the Eigen crop (426x560, V = 238,560) -- because that is the geometry the
# paper trains and scores on, while `experiments/train/depth` composes the
# low-resolution `datamodule: depth` (128x160, V = 20,480). Without this the cost
# row would be measured at a twelfth of the event dimension the score row
# describes. `make_runtime_table._DEPTH_DATASET` maps the datamodule *config
# name* to a dataset folder (`depth_bts_crop` -> `nyu_bts`, the same folder as
# the checkpoints and calibration diagnostics), and cost and accuracy only join
# there.
#
# `preload=false` is redundant against depth_bts_crop.yaml (which already streams
# -- ~37 GB on disk) but is kept explicit: it is a host-RAM limit, not a GPU one.
# Streaming costs the benchmark nothing -- the dataloader and the H2D copy sit
# outside the CUDA-event window by construction, so only the wall-clock column
# moves.
declare -A EXTRA=(
  [depth]="datamodule=depth_bts_crop datamodule.num_workers=4 datamodule.preload=false"
)

# Ranks swept for LorD. Both by default: the rank is nearly free (r=1 and r=16
# are within 0.1% on the station graph), so the pair documents that rather than
# leaving it assumed.
RANKS=${RANKS:-"1 16"}

# Forecast horizons swept per time-series task: the two the paper reports. The
# horizon sets the event dimension (V = pred_len x C), so it moves the cost of
# every covariance family and cannot be assumed away. The non-series tasks have
# no `pred_len` and are marked "-", which runs them once.
PRED_LENS=${PRED_LENS:-"96 192"}
# illness (966 rows) has its own reported pair, T24/T36.
declare -A TASK_HORIZONS=(
  [illness]="24 36"
)

TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  # Default sweep: the time-series tasks -- all four ETT sets in both modes,
  # weather and electricity -- at both reported horizons. ETTh and ETTm do NOT
  # cost the same: ETTm (and weather, electricity) train a d_model 128 backbone
  # against ETTh's 16, so every set is timed rather than one standing in for its
  # family. graph, era5, depth and illness are run by naming them; traffic runs
  # from runs/benchmark_traffic.sh. Measured cells are skipped (see above).
  # Roughly ~10 min per ETT task and horizon, ~1 h per electricity horizon,
  # ~5-6 h in all.
  TASKS=(etth1_uni etth2_uni ettm1_uni ettm2_uni etth1 etth2 ettm1 ettm2 weather electricity)
fi

# Resume rather than append: skip arms the log already holds for every seed.
SKIP="benchmark.skip_measured=true"
[ "${REMEASURE:-0}" = "1" ] && SKIP="benchmark.skip_measured=false"

# --- memory probe ---------------------------------------------------------
# Three steps of SB per candidate batch size, to find the largest one that fits
# the card. Deliberately SB: it is the memory ceiling on every gridded task, so
# a value it fits, all methods fit.
if [ "${PROBE:-0}" = "1" ]; then
  for task in "${TASKS[@]}"; do
    for bs in 1 2 4 8 16 32 64; do
      echo "--- probe $task bs=$bs ---"
      $BENCH experiments="${EXP[$task]}" method=sample_based \
        benchmark.batch_size=$bs benchmark.seeds=[1] \
        benchmark.warmup_steps=2 benchmark.measure_steps=3 \
        benchmark.out=/dev/null ${EXTRA[$task]:-} 2>&1 | grep -E "peak|OOM|Error" || true
    done
  done
  exit 0
fi

# --- benchmark ------------------------------------------------------------
for task in "${TASKS[@]}"; do
  bs=${BATCH[$task]}
  exp=${EXP[$task]}
  extra=${EXTRA[$task]:-}

  # "-" means the task has no forecast horizon (graph, era5, depth): one pass,
  # with no `datamodule.pred_len` override to hand Hydra.
  case "$exp" in
    */time_series/*) horizons=${TASK_HORIZONS[$task]:-$PRED_LENS} ;;
    *)               horizons="-" ;;
  esac

for T in $horizons; do
  if [ "$T" = "-" ]; then
    hz=""
    echo "===== $task (batch_size=$bs) ====="
  else
    hz="datamodule.pred_len=$T"
    echo "===== $task T=$T (batch_size=$bs) ====="
  fi

  # Det: no covariance at all, so this is the backbone-only floor -- what the
  # task costs before any joint model is asked for. Trained on the MSE, its only
  # sensible objective; it has no probabilistic score to share with the others,
  # so it bounds the table rather than joining the comparison.
  $BENCH experiments="$exp" method=deterministic method.loss=mse \
      benchmark.batch_size=$bs $hz $extra $SKIP

  # Diag: the cheapest probabilistic model, and the denominator every other
  # method's overhead is quoted against.
  $BENCH experiments="$exp" method=diag method.loss=kernel_score \
      benchmark.batch_size=$bs $hz $extra $SKIP

  # LorD (low-rank-plus-diagonal), one run per rank in RANKS.
  for r in $RANKS; do
    $BENCH experiments="$exp" method=lord method.loss=kernel_score \
        method.rank=$r benchmark.batch_size=$bs $hz $extra $SKIP
  done

  # SB, fitted with the fair kernel score (the method default). The ensemble
  # size M is the compute knob; it comes from method/sample_based.yaml (M = 5)
  # and is recorded in every row, so override it here only to sweep it.
  $BENCH experiments="$exp" method=sample_based benchmark.batch_size=$bs $hz $extra $SKIP

  # SCORE. Two-stage fit, so both stages are timed and the epoch-weighted
  # effective per-step cost is derived from the configured stage budgets.
  $BENCH experiments="${EXP_SCORE[$task]}" benchmark.batch_size=$bs $hz $extra $SKIP

  # Chol, where it is run: it carries one row of L per coordinate and refuses
  # event_dim > 1024, so it is defined on the univariate ETTh horizons
  # (V = 96/192) and the station graph (V = 115). ETTm univariate has the same V
  # but Chol is not run there, so it is not timed. It is the upper bound the
  # structured families are cheaper than.
  case "$task" in
    etth1_uni|etth2_uni|graph)
      $BENCH experiments="$exp" method=cholesky method.loss=kernel_score \
          benchmark.batch_size=$bs $hz $extra $SKIP
      ;;
  esac
done
done

echo "===== table ====="
# Prints the aggregate to stdout only. `--export` additionally writes
# <experiment>/runtime_analysis/runtime.jsonl and
# <dataset>/[<mode>/][T<pred_len>/]runtime.csv into the results tree, so it is
# opt-in. Publish with EXPORT=1, or afterwards by hand.
if [ "${EXPORT:-0}" = "1" ]; then
  $PY notebooks/make_runtime_table.py --export
else
  $PY notebooks/make_runtime_table.py
  echo
  echo "(not published; re-run with EXPORT=1, or:"
  echo "   python notebooks/make_runtime_table.py --export )"
fi
