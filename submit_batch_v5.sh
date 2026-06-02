#!/bin/bash
# Submit batch_v5_array.sbatch as a Slurm array sized to the number of STLs
# in DATA_DIR, with %18 concurrency (18 runs in parallel). Outputs go to a
# SEPARATE root (outputs/batch_v5) so they never mix with old results.
#
# Usage:
#   ./submit_batch_v5.sh                 # data/data_upload, 18-wide
#   CONCURRENCY=8 ./submit_batch_v5.sh   # fewer in parallel
set -euo pipefail

REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"
DATA_DIR="${1:-data/data_upload}"
OUTROOT="${OUTROOT:-outputs/batch_v5}"
CONCURRENCY="${CONCURRENCY:-18}"

COUNT=$(ls "${DATA_DIR}"/*.stl 2>/dev/null | wc -l)
if [ "$COUNT" -eq 0 ]; then echo "error: no .stl in $DATA_DIR" >&2; exit 1; fi
LAST=$((COUNT - 1))

DONE=0
for stl in "${DATA_DIR}"/*.stl; do
  base=$(basename "$stl" .stl)
  [ -f "${OUTROOT}/${base}/integrate/${base}_clean_deflector.stl" ] && DONE=$((DONE+1))
done
echo "[submit] DATA_DIR=${DATA_DIR}  OUTROOT=${OUTROOT}"
echo "[submit] total=${COUNT}  already_done=${DONE}  todo=$((COUNT-DONE))"
echo "[submit] array=0-${LAST}%${CONCURRENCY}  (1 CPU / 6G / 4h per task)"

mkdir -p logs
export DATA_DIR OUTROOT
sbatch --export=ALL --array=0-${LAST}%${CONCURRENCY} batch_v5_array.sbatch
