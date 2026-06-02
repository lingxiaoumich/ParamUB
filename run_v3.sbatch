#!/bin/bash
#SBATCH --job-name=paramub_v3
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# v3: asymmetric (inboard-only) front-steering-clearance fix for the
# wheelhouse builder. Reuses the v2 shell (stage 1 unaffected) and re-runs
# integrate -> fill_gap -> watertight into outputs/<car>_v3, then renders
# the production montage + per-step debug composites (full + front zoom).
# v2 is left intact for before/after comparison.
# Usage: sbatch run_v3.sbatch <car-id>

set -euo pipefail
CAR="${1:?usage: sbatch run_v3.sbatch <car-id>}"
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"
CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PARAMUB_PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENVDB_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYVISTA_OFF_SCREEN=true
mkdir -p logs

INPUT="data/data_upload/${CAR}.stl"
V2="outputs/${CAR}_v2"
OUT="outputs/${CAR}_v3"
echo "[$(date)] node=$(hostname) car=$CAR  reuse-shell=$V2/shell  out=$OUT"

# Stages 2-4, reusing the v2 shell (stage 1).
"$PARAMUB_PY" -u make_pipeline.py \
    --input "$INPUT" \
    --shell-dir "$V2/shell" --skip-shell \
    --integrate-dir "$OUT/integrate" \
    --logs-dir "$OUT/logs" \
    --timeout-integrate-min 60

# Production montage (10 views + Y sections).
"$PARAMUB_PY" -u render_paramub.py --car "$CAR" \
    --integrate-dir "$OUT/integrate" --out "$OUT"

# Per-step debug composites: full body + front-wheelhouse zoom.
"$PARAMUB_PY" -u render_ub_debug.py --car "$CAR" --bundle "$OUT"
"$PARAMUB_PY" -u render_ub_debug.py --car "$CAR" --bundle "$OUT" --focus front

echo "[$(date)] DONE car=$CAR"
echo "  clean STL : $OUT/integrate/${CAR}_clean.stl"
echo "  UB STL    : $OUT/integrate/${CAR}_underbody.stl  (+ _underbody_trimmed.stl)"
echo "  montage   : $OUT/renders/${CAR}_10view.png"
echo "  debug     : $OUT/debug/"
