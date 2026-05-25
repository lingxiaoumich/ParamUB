#!/bin/bash
#SBATCH --job-name=paramub_pipeline
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

# Full reconstructed-car pipeline (run_shell → integrate_underbody
# → fill_gap → make_watertight) on one raw STL input.
#
# Usage:
#   sbatch make_pipeline.sbatch <raw-input.stl> [extra args]
#
# Extra args are forwarded to make_pipeline.py (e.g. --depth 11,
# --skip-shell, --skip-integrate). See `python make_pipeline.py --help`.

set -euo pipefail

INPUT="${1:-data/alfa_romeo_giuliazhuliye_2025_image10_71415_shadowfill.stl}"
shift || true
EXTRA_ARGS=("$@")

mkdir -p logs

echo "[$(date)] node=$(hostname) input=$INPUT"
echo "[$(date)] CPUs=$SLURM_CPUS_PER_TASK MEM=${SLURM_MEM_PER_NODE}MB"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENVDB_THREADS=$SLURM_CPUS_PER_TASK

# Use the paramub env's python by absolute path. Activating via
# `conda activate paramub` is fragile inside Slurm batch shells because
# ~/.bashrc may re-activate a different env after us; the absolute path
# guarantees sys.executable is paramub's interpreter so subprocess
# spawns from make_pipeline.py use the same env.
PARAMUB_PY=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/bin/python
export LD_LIBRARY_PATH=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/lib:${LD_LIBRARY_PATH:-}

$PARAMUB_PY -u make_pipeline.py --input "$INPUT" "${EXTRA_ARGS[@]}"

echo "[$(date)] done"
