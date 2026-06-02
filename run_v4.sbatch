#!/bin/bash
#SBATCH --job-name=paramub_v4
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# FULL 4-stage pipeline on one car, fresh, with all current geometry fixes
# (15 spokes + filled hub, wheelhouse base-drop, asymmetric front steering,
# fine floor tessellation). Per-step timing is already instrumented in
# run_shell.py / integrate_underbody.py; make_pipeline.py logs each stage's
# elapsed. Generous stage-4 watchdog so the Blender remesh completes (v2
# took ~25 min; v3 timed out at the old 45-min default on the denser mesh).
# Also builds the standalone deflector STL (cheap) so it's ready to trim in.
# Usage: sbatch run_v4.sbatch <car-id>   (default 0mjLG)

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
OUT="outputs/${CAR}_v4"
echo "[$(date)] node=$(hostname) car=$CAR  input=$INPUT  out=$OUT"
echo "[$(date)] CPUs=${SLURM_CPUS_PER_TASK:-?}  MEM=${SLURM_MEM_PER_NODE:-?}MB"

T0=$SECONDS

# Full 4-stage pipeline (run_shell -> integrate -> fill_gap -> watertight).
"$PY" -u make_pipeline.py \
    --input "$INPUT" \
    --shell-dir "$OUT/shell" \
    --integrate-dir "$OUT/integrate" \
    --logs-dir "$OUT/logs" \
    --timeout-shell-min 30 \
    --timeout-integrate-min 60 \
    --timeout-fillgap-min 30 \
    --timeout-watertight-min 180

echo "[$(date)] pipeline wall = $((SECONDS - T0)) s"

# Standalone front-tire deflector (login-node-cheap; just for completeness).
TD=$SECONDS
"$PY" make_deflector.py \
    --shell-meta "$OUT/shell/${CAR}_meta.json" \
    --output-dir "$OUT/integrate" || echo "[warn] deflector step failed (non-fatal)"
echo "[$(date)] deflector wall = $((SECONDS - TD)) s"

echo "[$(date)] DONE car=$CAR  total wall = $((SECONDS - T0)) s"
echo "  clean body : $OUT/integrate/${CAR}_clean.stl"
echo "  deflector  : $OUT/integrate/${CAR}_deflector.stl"
echo "  summary    : $OUT/integrate/${CAR}_summary.json"
