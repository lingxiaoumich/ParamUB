#!/bin/bash
#SBATCH --job-name=contact_v5
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"
CONDA_PREFIX=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
export CONDA_PREFIX LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "[$(date)] node=$(hostname)"
"$CONDA_PREFIX/bin/python" make_contact_sheets_v5.py
echo "[$(date)] done"
ls -lh outputs/batch_v5_summary/contact_*.png | awk '{print $5, $NF}'
