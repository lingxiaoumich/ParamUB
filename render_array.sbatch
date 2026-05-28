#!/bin/bash
#SBATCH --job-name=ub_render
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

# Render one successful car (clean body + 4 wheels) into 10 views.
# The list of successful cars is rendered_cars.txt (one base name per
# line) -- regenerate it before submitting:
#   ls outputs/*/integrate/*_clean.stl | sed -E 's#.*/([^/]+)_clean.stl#\1#' | sort -u > rendered_cars.txt
#
# Submit via submit_render.sh, which builds the list + sizes the array.

set -euo pipefail

LIST="${RENDER_LIST:-rendered_cars.txt}"
mapfile -t CARS < "$LIST"
TOTAL=${#CARS[@]}

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TOTAL" ]; then
  echo "[render] task ${SLURM_ARRAY_TASK_ID} >= ${TOTAL} -- nothing to do"
  exit 0
fi

CAR="${CARS[$SLURM_ARRAY_TASK_ID]}"
mkdir -p logs

# Skip if the montage already exists (idempotent re-runs).
if [ -f "outputs/summary/renders/${CAR}_10view.png" ]; then
  echo "[skip] ${CAR} already rendered"
  exit 0
fi

PARAMUB_PY=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/bin/python
export LD_LIBRARY_PATH=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/lib:${LD_LIBRARY_PATH:-}
# Headless GL: force off-screen software/EGL path on compute nodes.
export PYVISTA_OFF_SCREEN=true
export DISPLAY=""

echo "[$(date)] node=$(hostname) task=${SLURM_ARRAY_TASK_ID} car=${CAR}"
$PARAMUB_PY -u render_paramub.py --car "$CAR"
echo "[$(date)] done ${CAR}"
