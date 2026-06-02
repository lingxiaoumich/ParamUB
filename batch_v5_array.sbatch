#!/bin/bash
#SBATCH --job-name=ub_v5
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

# v5 batch: one STL per array index, full pipeline + deflector + trim, into
# a SEPARATE output root (outputs/batch_v5/<base>/...) so it never mixes
# with the old outputs/<base>/ results.
#
# v5 recipe (matches the validated single-car run_v5.sbatch):
#   * 4-stage pipeline with stage-4 remesh depth 10 and --no-verify-wheels
#     (verify samples default 5000); per-stage WATCHDOG timeouts left at
#     make_pipeline.py defaults — i.e. identical to the previous batch run
#     (batch_array.sbatch passed no timeout flags): shell 30 / integrate 15
#     / fillgap 15 / watertight 45 min.
#   * standalone front-tire deflector
#   * FINAL: trim (bpy union) the deflector into the watertight body
#
# Resources: 1 CPU, 6G, 4h walltime (per request).
# Re-running resumes: tasks whose <base>_clean_deflector.stl already exists
# exit immediately.
#
# Submit via: ./submit_batch_v5.sh   (sizes the array + sets %18 concurrency)

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/data_upload}"
OUTROOT="${OUTROOT:-outputs/batch_v5}"
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"

mapfile -t STLS < <(ls "${DATA_DIR}"/*.stl | sort)
TOTAL=${#STLS[@]}
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TOTAL" ]; then
  echo "[array] task ${SLURM_ARRAY_TASK_ID} >= ${TOTAL} stls -- nothing to do"
  exit 0
fi

INPUT="${STLS[$SLURM_ARRAY_TASK_ID]}"
BASE=$(basename "$INPUT" .stl)
OUT="${OUTROOT}/${BASE}"
INTEG="${OUT}/integrate"
FINAL="${INTEG}/${BASE}_clean_deflector.stl"

mkdir -p logs

echo "[$(date)] node=$(hostname) task=${SLURM_ARRAY_TASK_ID}/$((TOTAL-1)) base=${BASE}"
echo "[$(date)] input=${INPUT}  outroot=${OUTROOT}"
echo "[$(date)] CPUs=${SLURM_CPUS_PER_TASK} MEM=${SLURM_MEM_PER_NODE}MB"

if [ -f "$FINAL" ]; then
  echo "[skip] ${FINAL} already exists -- v5 already complete"
  exit 0
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENVDB_THREADS=$SLURM_CPUS_PER_TASK
CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

T0=$SECONDS

# --- 4-stage pipeline into the v5 output root (depth 10, no wheel verify) ---
"$PY" -u make_pipeline.py \
    --input "$INPUT" \
    --outputs-root "$OUTROOT" \
    --depth 10 \
    --no-verify-wheels
echo "[$(date)] PIPELINE wall = $((SECONDS - T0)) s"

# --- standalone front-tire deflector (cheap) ---
TD=$SECONDS
"$PY" make_deflector.py \
    --shell-meta "${OUT}/shell/${BASE}_meta.json" \
    --output-dir "$INTEG"
echo "[$(date)] DEFLECTOR wall = $((SECONDS - TD)) s"

# --- FINAL: trim deflector into the watertight body (bpy union) ---
TT=$SECONDS
"$PY" -u trim_deflector.py \
    --body      "${INTEG}/${BASE}_clean.stl" \
    --deflector "${INTEG}/${BASE}_deflector.stl" \
    --out       "$FINAL" \
    --solver FAST
echo "[$(date)] TRIM wall = $((SECONDS - TT)) s"

echo "[$(date)] DONE base=${BASE}  TOTAL wall = $((SECONDS - T0)) s -> ${FINAL}"
