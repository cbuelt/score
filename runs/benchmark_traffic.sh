#!/bin/bash
#SBATCH --job-name=mvg-bench-traffic
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail

# Runtime benchmark of the traffic task (862 variates, V = 82,752 at T96 and
# 165,504 at T192) at both reported horizons, on a 48 GB card:
#
#   sbatch runs/benchmark_traffic.sh
#   PRED_LENS=192 sbatch runs/benchmark_traffic.sh     # one horizon
#
# Expected ~1.5-2 h (~35 min of timed steps at T96 and ~40 min at T192, plus a
# datamodule build per method); 6 h is margin.
#
# A thin wrapper around `runs/benchmark.sh traffic`, which owns the methods, the
# seeds and the pinned batch size (`BATCH[traffic]` there). It adds one step
# that script does not do on its own: moving rows measured at any *other* batch
# size out of the log before the new ones land.
#
# Why that is needed. At the datamodule's own batch size (24) SB does not fit a
# 48 GB card -- it carries M = 5 members through the head's encoder block, i.e.
# B * 862 * 5 sequences -- so traffic is timed at the lower pin for every
# method. `skip_measured` matches on batch size, so rows at another batch size do
# not stop the run; but `make_runtime_table.aggregate` keys on (dataset,
# features, pred_len, method, rank) only, so they would *pool* with the new ones
# into one mean of two different batch sizes. They are moved, not deleted, to a
# `retired_*.jsonl` file next to the log. Only the horizons this job times are
# touched. Idempotent: a resubmission finds nothing left to move, and
# benchmark.sh's skip resumes where the previous job stopped.
#
# Publishing stays opt-in, as in benchmark.sh: EXPORT=1, or afterwards
# `python notebooks/make_runtime_table.py --export`.

root=$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" && pwd)
while [ ! -e "$root/.project-root" ] && [ "$root" != "/" ]; do
  root=$(dirname "$root")
done
if [ ! -e "$root/.project-root" ]; then
  echo "cannot find the repo root (.project-root) from ${SLURM_SUBMIT_DIR:-$(dirname "$0")}" >&2
  exit 1
fi
cd "$root"

export PRED_LENS=${PRED_LENS:-"96 192"}

# Machine-specific environment (conda activation, data locations); see runs/env.sh.example.
[ -f runs/env.sh ] && source runs/env.sh

# The pin is read from benchmark.sh rather than repeated here, so the two can
# never disagree about which rows are stale.
BS=$(sed -n 's/.*\[traffic\]=\([0-9]\+\).*/\1/p' runs/benchmark.sh | head -1)
if [ -z "$BS" ]; then
  echo "cannot read BATCH[traffic] from runs/benchmark.sh" >&2
  exit 1
fi
echo "traffic pinned at batch_size=$BS, horizons: $PRED_LENS"

python - "$BS" $PRED_LENS <<'EOF'
import json, os, sys
from pathlib import Path

bs, horizons = int(sys.argv[1]), {int(t) for t in sys.argv[2:]}
log = Path("logs/benchmark/runtime.jsonl")
retired = Path("logs/benchmark/retired_traffic_bs24_2026-10-01.jsonl")
if not log.is_file():
    sys.exit(0)
keep, move = [], []
for line in log.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    stale = (r["dataset"] == "traffic" and r.get("pred_len") in horizons
             and r.get("batch_size") != bs)
    (move if stale else keep).append(line)
if move:
    with retired.open("a") as f:
        f.write("\n".join(move) + "\n")
    tmp = log.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(keep) + "\n")
    os.replace(tmp, log)
print(f"retired {len(move)} traffic rows not at batch_size={bs} -> {retired}")
EOF

exec bash runs/benchmark.sh traffic
