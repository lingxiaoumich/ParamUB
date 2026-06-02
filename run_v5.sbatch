#!/bin/bash
#SBATCH --job-name=paramub_v5
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# v5: full pipeline + deflector + trim, ONE self-contained job (end to end
# on a single compute node, so no cross-job dependency / ID juggling).
# Efficiency tweaks vs v4: stage-4 remesh depth 11->10, verify samples
# 30000->5000 (make_watertight default), and --no-verify-wheels.
# Deflector geometry (in code): +10mm forward, both Y bounds +20mm inboard.
# Deflector trim is applied AFTER stage 4 (bpy union onto the watertight body).
# Usage: sbatch run_v5.sbatch <car-id>   (default 0mjLG)

set -euo pipefail
CAR="${1:-0mjLG}"
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"
CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENVDB_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p logs

INPUT="data/data_upload/${CAR}.stl"
OUT="outputs/${CAR}_v5"
INTEG="$OUT/integrate"
echo "[$(date)] node=$(hostname) car=$CAR  input=$INPUT  out=$OUT"
echo "[$(date)] CPUs=${SLURM_CPUS_PER_TASK:-?}  MEM=${SLURM_MEM_PER_NODE:-?}MB"

T0=$SECONDS

# --- 4-stage pipeline: depth 10 + skip wheel verify (samples default 5000) ---
"$PY" -u make_pipeline.py \
    --input "$INPUT" \
    --shell-dir "$OUT/shell" \
    --integrate-dir "$INTEG" \
    --logs-dir "$OUT/logs" \
    --depth 10 \
    --no-verify-wheels \
    --timeout-shell-min 30 \
    --timeout-integrate-min 60 \
    --timeout-fillgap-min 30 \
    --timeout-watertight-min 180
echo "[$(date)] PIPELINE wall = $((SECONDS - T0)) s"

# --- standalone deflector (cheap) ---
TD=$SECONDS
"$PY" make_deflector.py \
    --shell-meta "$OUT/shell/${CAR}_meta.json" \
    --output-dir "$INTEG"
echo "[$(date)] DEFLECTOR wall = $((SECONDS - TD)) s"

# --- FINAL: trim deflector into the watertight body (bpy union) ---
TT=$SECONDS
"$PY" -u trim_deflector.py \
    --body      "$INTEG/${CAR}_clean.stl" \
    --deflector "$INTEG/${CAR}_deflector.stl" \
    --out       "$INTEG/${CAR}_clean_deflector.stl" \
    --solver FAST
echo "[$(date)] TRIM wall = $((SECONDS - TT)) s"

echo "[$(date)] DONE car=$CAR  TOTAL wall = $((SECONDS - T0)) s"
echo "  final body : $INTEG/${CAR}_clean_deflector.stl"
