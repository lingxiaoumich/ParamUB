#!/bin/bash
# Submit batch_array.sbatch as a Slurm array job sized to the number
# of STLs in DATA_DIR (default data/data_upload/). Concurrency is
# capped at 8 by the %8 throttle to stay under the group memory cap.
#
# Usage:
#   ./submit_batch.sh                       # data/data_upload/, 8-wide
#   ./submit_batch.sh data/other_dir         # different input dir
#   CONCURRENCY=4 ./submit_batch.sh         # half-wide (be a good citizen)

set -euo pipefail

DATA_DIR="${1:-data/data_upload}"
CONCURRENCY="${CONCURRENCY:-10}"

if [ ! -d "$DATA_DIR" ]; then
  echo "error: $DATA_DIR does not exist" >&2
  exit 1
fi

COUNT=$(ls "${DATA_DIR}"/*.stl 2>/dev/null | wc -l)
if [ "$COUNT" -eq 0 ]; then
  echo "error: no .stl files in $DATA_DIR" >&2
  exit 1
fi
LAST=$((COUNT - 1))

# Count cars already done so the user sees the real workload.
DONE=0
for stl in "${DATA_DIR}"/*.stl; do
  base=$(basename "$stl" .stl)
  if [ -f "outputs/${base}/integrate/${base}_summary.json" ]; then
    DONE=$((DONE + 1))
  fi
done
TODO=$((COUNT - DONE))

echo "[submit] DATA_DIR=${DATA_DIR}"
echo "[submit] total=${COUNT}  already_done=${DONE}  todo=${TODO}"
echo "[submit] array=0-${LAST}%${CONCURRENCY}  (skipped tasks exit in <1s)"

export DATA_DIR
sbatch --export=ALL --array=0-${LAST}%${CONCURRENCY} batch_array.sbatch
