#!/bin/bash
#SBATCH --job-name=paramub_v2
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Full reconstructed-car pipeline with the v2 UB tweaks, into a SEPARATE
# per-car bundle outputs/<car>_v2/{shell,integrate,logs,renders}, leaving
# the original outputs/<car>/ untouched.
#   1. spoke windows 5 -> 15, central hub bore filled (no hole at centre)
#   2. front wheelhouse inboard (steering) clearance +80 mm
#   3. wheelhouse roof +60 mm above wheel centre (radial clearance 20 -> 80)
#   4. render Y sections at y = 0,200,400,600,800 mm
#   5. per-step wall-clock timing logged in every stage
# Usage: sbatch run_v2.sbatch <car-id>

set -euo pipefail

CAR="${1:?usage: sbatch run_v2.sbatch <car-id>}"
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"

CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PARAMUB_PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENVDB_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p logs
INPUT="data/data_upload/${CAR}.stl"
OUT="outputs/${CAR}_v2"
echo "[$(date)] node=$(hostname) car=$CAR input=$INPUT out=$OUT py=$PARAMUB_PY"

# 4-stage geometry pipeline (run_shell -> integrate -> fill_gap ->
# make_watertight) into the v2 bundle.
"$PARAMUB_PY" -u make_pipeline.py \
    --input "$INPUT" \
    --shell-dir "$OUT/shell" \
    --integrate-dir "$OUT/integrate" \
    --logs-dir "$OUT/logs" \
    --timeout-integrate-min 60

# Render: 10 views + Y sections (0,200,400,600,800 mm) from the v2 body.
"$PARAMUB_PY" -u render_paramub.py \
    --car "$CAR" \
    --integrate-dir "$OUT/integrate" \
    --out "$OUT"

echo "[$(date)] DONE car=$CAR -> $OUT/renders/${CAR}_10view.png"
