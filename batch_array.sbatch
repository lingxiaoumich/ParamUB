#!/bin/bash
#SBATCH --job-name=ub_array
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Slurm array task: process one STL from DATA_DIR per array index.
# Skips cars whose summary.json already exists, so re-running this
# array picks up where it left off.
#
# Submit via: ./submit_batch.sh [DATA_DIR]

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/data_upload}"
mapfile -t STLS < <(ls "${DATA_DIR}"/*.stl | sort)
TOTAL=${#STLS[@]}

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TOTAL" ]; then
  echo "[array] task ${SLURM_ARRAY_TASK_ID} >= ${TOTAL} stls -- nothing to do"
  exit 0
fi

INPUT="${STLS[$SLURM_ARRAY_TASK_ID]}"
BASE=$(basename "$INPUT" .stl)
SUMMARY="outputs/${BASE}/integrate/${BASE}_summary.json"

mkdir -p logs

echo "[$(date)] node=$(hostname) task=${SLURM_ARRAY_TASK_ID}/$((TOTAL-1)) base=${BASE}"
echo "[$(date)] input=${INPUT}"
echo "[$(date)] CPUs=${SLURM_CPUS_PER_TASK} MEM=${SLURM_MEM_PER_NODE}MB"

if [ -f "$SUMMARY" ]; then
  echo "[skip] ${SUMMARY} already exists -- pipeline already complete"
  exit 0
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENVDB_THREADS=$SLURM_CPUS_PER_TASK
PARAMUB_PY=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/bin/python
export LD_LIBRARY_PATH=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/lib:${LD_LIBRARY_PATH:-}

$PARAMUB_PY -u make_pipeline.py --input "$INPUT"
echo "[$(date)] done"
