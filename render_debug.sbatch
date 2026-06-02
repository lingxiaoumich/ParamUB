#!/bin/bash
#SBATCH --job-name=paramub_dbg
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Per-step debug composites for one car bundle (integrate chain), plus the
# shell extraction per-step renders. Usage: sbatch render_debug.sbatch <car>
set -euo pipefail
CAR="${1:?usage: sbatch render_debug.sbatch <car-id> [extra render_ub_debug args]}"
shift
EXTRA=("$@")   # forwarded to render_ub_debug.py (e.g. --focus front)
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"
CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PARAMUB_PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYVISTA_OFF_SCREEN=true
mkdir -p logs
BUNDLE="outputs/${CAR}_v2"
echo "[$(date)] node=$(hostname) debug-render car=$CAR bundle=$BUNDLE"

# Integrate-chain composites (detected / generated / trim / combined / clean).
"$PARAMUB_PY" -u render_ub_debug.py --car "$CAR" --bundle "$BUNDLE" "${EXTRA[@]}"

echo "[$(date)] DONE -> $BUNDLE/debug/"
