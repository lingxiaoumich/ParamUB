#!/bin/bash
#SBATCH --job-name=pub_trim
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# FINAL step: trim/merge the front tire deflector into the finished
# watertight body (Blender boolean union). Compute node only — the body is
# ~14M triangles. Usage: sbatch trim_deflector.sbatch <car> [bundle-suffix]
set -euo pipefail
CAR="${1:-0mjLG}"
SUF="${2:-v4}"
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"
CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PY="$CONDA_PREFIX/bin/python"
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p logs
INTEG="outputs/${CAR}_${SUF}/integrate"
echo "[$(date)] node=$(hostname) trim deflector for $CAR ($INTEG)"
"$PY" -u trim_deflector.py \
    --body      "$INTEG/${CAR}_clean.stl" \
    --deflector "$INTEG/${CAR}_deflector.stl" \
    --out       "$INTEG/${CAR}_clean_deflector.stl" \
    --solver FAST
echo "[$(date)] DONE -> $INTEG/${CAR}_clean_deflector.stl"
