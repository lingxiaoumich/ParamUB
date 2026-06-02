#!/bin/bash
#SBATCH --job-name=render_v5
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Render one batch_v5 car (clean body + 4 wheels) into 10 views + Y sections.
# Reads the list of successfully-finished cars from RENDER_LIST (built by
# submit_render_v5.sh from the on-disk _clean_deflector.stl inventory).
# Writes per-view PNGs to OUTDIR/views/<view_slug>/<car>.png, which is what
# make_contact_sheets_v5.py reads.
# Skip cars already rendered (idempotent re-runs).
#
# Submit via:  ./submit_render_v5.sh

set -euo pipefail

REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"

LIST="${RENDER_LIST:-render_list_v5.txt}"
BATCH_OUT="${BATCH_OUT:-outputs/batch_v5}"
OUTDIR="${OUTDIR:-outputs/batch_v5_summary}"

mapfile -t CARS < "$LIST"
TOTAL=${#CARS[@]}

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TOTAL" ]; then
  echo "[render] task ${SLURM_ARRAY_TASK_ID} >= ${TOTAL} -- nothing to do"
  exit 0
fi

CAR="${CARS[$SLURM_ARRAY_TASK_ID]}"
INTEG="${BATCH_OUT}/${CAR}/integrate"
SKIP_FILE="${OUTDIR}/views/bottom_front_iso/${CAR}.png"

mkdir -p logs

if [ -f "$SKIP_FILE" ]; then
  echo "[skip] ${CAR} already rendered"
  exit 0
fi

CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYVISTA_OFF_SCREEN=true
export DISPLAY=""

echo "[$(date)] node=$(hostname) task=${SLURM_ARRAY_TASK_ID}/$((TOTAL-1)) car=${CAR}"
echo "[$(date)] integ=${INTEG}  outdir=${OUTDIR}"

"$PY" -u render_paramub.py \
    --car "$CAR" \
    --integrate-dir "$INTEG" \
    --out "$OUTDIR"

echo "[$(date)] done ${CAR}"
