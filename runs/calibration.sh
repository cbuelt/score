#!/bin/bash
#SBATCH --job-name=mvg-calibration
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --time=24:00:00
set -euo pipefail

# Multivariate calibration diagnostics for a tree of trained checkpoints.
#
#   sbatch runs/calibration.sh depth generative_kernel_score   # SB only
#   DRYRUN=1 bash runs/calibration.sh depth                    # print the run list
#   sbatch runs/calibration.sh depth                           # every stochastic slug
#   sbatch runs/calibration.sh era5 'phys_lora_r8_*'           # LorD, r = 8
#   SEEDS='1 2' sbatch runs/calibration.sh graph 'generative_*'
#
# Slugs are the results-folder names: SCORE is `whitened_*`, LorD `phys_lora_*`,
# Diag `phys_diag_*`, Chol `phys_cholesky_*`, SB `generative_*`.
#
# Writes <experiment>/calibration/<dataset>/<slug>/calibration_seed<N>.npz in the
# results tree; `src/evaluation/calibration.py` works the destination out from
# each run folder's own config.yaml, so nothing about the path is passed here.
#
# GPU MEMORY: SB NEEDS A 48 GB CARD ON DEPTH
#
# The SB (`generative_*`) runs cannot be diagnosed on a 24 GB card at depth
# resolution. Two separate calls in calibration.py draw samples and only one of
# them is chunkable:
#
#   model(x, n_samples=chunk)   <- `--draw-chunk` controls this
#   pred = model(x)             <- no n_samples, so SampleBased falls back to
#                                  the config's `n_eval_samples` (100)
#
# The second runs once per batch for every method, just to get `mu`, and at
# depth's V = 238,560 it asks for a single 11.59 GiB tensor inside F.conv2d
# (~27 GiB in total, independent of `--draw-chunk`). Do NOT work around it by
# lowering `n_eval_samples`: `get_mean` estimates the mean *from* the members,
# so fewer members changes the estimate, not just the memory. On EUPPBench
# (V = 115) the call is cheap.
#
# THE AXIS FLAGS
#
# `dep_lags`, `dep_bin_edges_km` and `n_draws` are *axis keys*: two npz files
# disagreeing on any of them cannot be pooled, and `load_pooled` in
# notebooks/calibration_histograms.ipynb refuses them. The defaults in
# calibration.py are not what the reported diagnostics used, so every ladder
# below is passed explicitly:
#
#   depth  schema 2, n_draws 199, grid_lag,          dirs axis0/axis1/diag/antidiag/axial_mean
#   era5   schema 2, n_draws 999, grid_distance_bin, dirs axis0/axis1/diag/antidiag/pooled
#   graph  schema 2, n_draws 999, distance_bin,      dirs distance_bin (no isotropy: not a grid)
#
# Re-running is a safe resume -- `needs_refresh` keys the skip on
# `schema_version`, so current files are dismissed in seconds and only missing or
# half-written ones are redone. **Never pass `--overwrite` to "make sure"**: it
# recomputes everything, and for SB that is hours. The one case that genuinely
# needs it is a *replaced checkpoint*, which the schema check cannot see.

# Repo root, the same `.project-root` walk runs/epoch_budget.sh and
# runs/benchmark.sh do: under sbatch, `$0` is Slurm's copy in the spool
# directory, so relative paths would resolve there and not in the repo.
root=$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" && pwd)
while [ ! -e "$root/.project-root" ] && [ "$root" != "/" ]; do
  root=$(dirname "$root")
done
if [ ! -e "$root/.project-root" ]; then
  echo "cannot find the repo root (.project-root) from ${SLURM_SUBMIT_DIR:-$(dirname "$0")}" >&2
  exit 1
fi
cd "$root"

# Machine-specific environment (conda activation, SHARED, ERA5_*); see runs/env.sh.example.
[ -f runs/env.sh ] && source runs/env.sh

SHARED=${SHARED:-results}

# Checkpoint tree per task nickname, matching runs/benchmark.sh's names.
declare -A TREE=(
  [depth]="$SHARED/depth/nyu_bts"
  [era5]="$SHARED/surface_temperature/era5"
  [graph]="$SHARED/station_postprocessing/euppbench"
)

# Axis flags per task. See the note above: these are not defaults, they are the
# ladders the reported diagnostics use, and changing one makes new files
# unpoolable with the existing ones.
#
# era5 additionally needs both of its recorded paths overridden when the
# checkpoints were trained on another machine, and the *second* one matters:
# unset, `statistics_path` falls back to <data_path>/era5_statistics.npy, which is
# computed over 1990-2022 *including the val/test years*, while training
# standardizes on the training years 1990-2014 only. That fallback would not be a
# missing file but a different normalization -- diagnostics silently describing
# a test set the checkpoints were never fitted for.
ERA5_DATA=${ERA5_DATA:-data/era5/}
ERA5_STATS=${ERA5_STATS:-data/era5/era5_statistics_train1990-2014.npy}
declare -A FLAGS=(
  [depth]="--lags 1,2,4,8,16,32,64 --n-draws 199 --draw-chunk 32"
  [era5]="--lags 1,2,3,4,6,8,11,16,22,32,40 --bin-edges-km 0,50,100,200,400,800 --num-workers 4 --data-path $ERA5_DATA --dm-override statistics_path=$ERA5_STATS"
  [graph]="--bin-edges-km 0,50,100,200,300,500,inf --num-workers 4"
)

task=${1:-}
if [ -z "$task" ] || [ -z "${TREE[$task]:-}" ]; then
  echo "usage: sbatch runs/calibration.sh <task> [slug-glob ...]" >&2
  echo "  tasks: ${!TREE[*]}" >&2
  echo "  time_series is deliberately NOT here: its calibration is a chosen" >&2
  echo "  subset, not every arm, and it has its own lag ladder per horizon" >&2
  echo "  (ETTh T96 / ETTh T192 / illness T24 / illness T36 / weather)." >&2
  echo "  Run those by hand -- pass the axis flags (--lags / --bin-edges-km) explicitly." >&2
  exit 1
fi
shift

tree=${TREE[$task]}
flags=${FLAGS[$task]}
[ -d "$tree" ] || { echo "no checkpoint tree at $tree" >&2; exit 1; }

# Slug globs; default to everything in the tree. `deterministic_*` is dropped
# rather than left for calibration.py to skip -- a Dirac has sigma = 0 and no
# PIT, so it is not a missing diagnostic and should not look like one in the log.
globs=("$@")
[ ${#globs[@]} -eq 0 ] && globs=("*")

# Seeds; default to whatever the tree holds.
SEEDS=${SEEDS:-}

run_dirs=()
for g in "${globs[@]}"; do
  for slug_dir in "$tree"/$g; do
    [ -d "$slug_dir" ] || continue
    slug=$(basename "$slug_dir")
    case "$slug" in deterministic_*) continue ;; esac
    if [ -n "$SEEDS" ]; then
      for s in $SEEDS; do
        [ -d "$slug_dir/seed$s" ] && run_dirs+=("$slug_dir/seed$s")
      done
    else
      for sd in "$slug_dir"/seed*; do
        [ -d "$sd" ] && run_dirs+=("$sd")
      done
    fi
  done
done

if [ ${#run_dirs[@]} -eq 0 ]; then
  echo "no stochastic seed folders matched ${globs[*]} under $tree" >&2
  exit 1
fi

# Keep ~/.local out of sys.path: user-site packages can shadow the environment's
# and make an import succeed against the wrong version rather than fail.
export PYTHONNOUSERSITE=1

echo "Job:     ${SLURM_JOB_ID:-none} on $(hostname)"
echo "Python:  $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Task:    $task"
echo "Tree:    $tree"
echo "Flags:   $flags"
echo "Runs:    ${#run_dirs[@]} seed folder(s)"
printf '           %s\n' "${run_dirs[@]#$tree/}"

# Warn rather than fail: the closed-form methods run fine on a smaller card, only
# SB on depth needs the 48 GB. Checking here means the message lands at the top
# of the log instead of as an OOM traceback 20 minutes in.
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 || echo "")
if [ -n "$total_mib" ] && [ "$total_mib" -lt 40000 ]; then
  case " ${run_dirs[*]} " in
    *generative_*)
      echo "WARNING: this GPU has ${total_mib} MiB and the run list includes a" >&2
      echo "         generative_* arm, which needs ~27 GiB (see the header)." >&2
      echo "         Those seeds will OOM; the closed-form ones will still run." >&2
      ;;
  esac
fi

if [ "${DRYRUN:-0}" = "1" ]; then
  echo
  echo "DRYRUN: would run"
  echo "  python -m src.evaluation.calibration \\"
  printf '      %s \\\n' "${run_dirs[@]}"
  echo "      --device cuda $flags"
  exit 0
fi

# One invocation, all run folders: calibration.py loops over them and skips the
# ones already current, which is what makes a bare resubmit a resume.
exec python -m src.evaluation.calibration "${run_dirs[@]}" --device cuda $flags
