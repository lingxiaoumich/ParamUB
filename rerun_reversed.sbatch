#!/bin/bash
#SBATCH --job-name=ub_revfix
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Re-run the reverse_x cars from raw input through the FULL v5 pipeline so
# the new stage-1 auto-orient (greenhouse_x < -0.05 -> 180deg flip) takes
# effect. Output to a SEPARATE root so the existing batch_v5 is untouched
# until the fix is verified. Same v5 recipe: depth 10, --no-verify-wheels,
# standalone deflector, bpy trim onto the watertight body.
#
# Car list: rerun_reversed.txt (one base name per line).
# Resources match the v5 batch: 1 CPU / 6G / 4h.
# Submit via: ./submit_rerun_reversed.sh

set -euo pipefail
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"

LIST="${LIST:-rerun_reversed.txt}"
OUTROOT="${OUTROOT:-outputs/batch_v5_reversed_fix}"
mapfile -t CARS < "$LIST"
TOTAL=${#CARS[@]}
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TOTAL" ]; then
  echo "[array] task ${SLURM_ARRAY_TASK_ID} >= ${TOTAL} -- nothing to do"; exit 0
fi

CAR="${CARS[$SLURM_ARRAY_TASK_ID]}"
INPUT="data/data_upload/${CAR}.stl"
OUT="${OUTROOT}/${CAR}"
INTEG="${OUT}/integrate"
FINAL="${INTEG}/${CAR}_clean_deflector.stl"
mkdir -p logs

echo "[$(date)] node=$(hostname) task=${SLURM_ARRAY_TASK_ID}/$((TOTAL-1)) car=${CAR}"
if [ -f "$FINAL" ]; then echo "[skip] $FINAL exists"; exit 0; fi

CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENVDB_THREADS=${SLURM_CPUS_PER_TASK:-1}

T0=$SECONDS
"$PY" -u make_pipeline.py \
    --input "$INPUT" \
    --outputs-root "$OUTROOT" \
    --depth 10 \
    --no-verify-wheels
echo "[$(date)] PIPELINE wall = $((SECONDS - T0)) s"

TD=$SECONDS
"$PY" make_deflector.py \
    --shell-meta "${OUT}/shell/${CAR}_meta.json" \
    --output-dir "$INTEG"
echo "[$(date)] DEFLECTOR wall = $((SECONDS - TD)) s"

TT=$SECONDS
"$PY" -u trim_deflector.py \
    --body      "${INTEG}/${CAR}_clean.stl" \
    --deflector "${INTEG}/${CAR}_deflector.stl" \
    --out       "$FINAL" \
    --solver FAST
echo "[$(date)] TRIM wall = $((SECONDS - TT)) s"
echo "[$(date)] DONE ${CAR} total = $((SECONDS - T0)) s -> ${FINAL}"
