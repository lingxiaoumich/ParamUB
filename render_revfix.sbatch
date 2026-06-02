#!/bin/bash
#SBATCH --job-name=revfix_rnd
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd /scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
CP=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
export CONDA_PREFIX=$CP LD_LIBRARY_PATH="$CP/lib:${LD_LIBRARY_PATH:-}" PYVISTA_OFF_SCREEN=true DISPLAY=""
mapfile -t CARS < rerun_reversed.txt
CAR="${CARS[$SLURM_ARRAY_TASK_ID]}"
"$CP/bin/python" -u render_paramub.py --car "$CAR" \
  --integrate-dir "outputs/batch_v5_reversed_fix/$CAR/integrate" \
  --out "outputs/batch_v5_reversed_fix_summary"
echo "done $CAR"
