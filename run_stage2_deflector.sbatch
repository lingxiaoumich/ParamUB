#!/bin/bash
#SBATCH --job-name=pub_defl
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# STAGE 2 ONLY (integrate_underbody) with the new front tire deflector,
# on a compute node to avoid the login-node OOM. Reuses the existing v2
# shell (stage 1). Writes STLs to outputs/<car>_deflector/integrate for
# manual inspection. No renders, no stages 3/4.
# Usage: sbatch run_stage2_deflector.sbatch <car-id>   (default 0mjLG)

set -euo pipefail
CAR="${1:-0mjLG}"
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"
CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"  # pymeshlab remesh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p logs

SHELL_META="outputs/${CAR}_v2/shell/${CAR}_meta.json"
OUT="outputs/${CAR}_deflector/integrate"
echo "[$(date)] node=$(hostname) car=$CAR  shell_meta=$SHELL_META  out=$OUT"

"$PY" -u integrate_underbody.py \
    --shell-meta "$SHELL_META" \
    --output-dir "$OUT"

echo "[$(date)] DONE car=$CAR"
echo "  underbody     : $OUT/${CAR}_underbody.stl"
echo "  underbody trim: $OUT/${CAR}_underbody_trimmed.stl"
