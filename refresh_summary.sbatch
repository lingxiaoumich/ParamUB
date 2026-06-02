#!/bin/bash
#SBATCH --job-name=refresh_sum
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
CP=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
export LD_LIBRARY_PATH="$CP/lib:${LD_LIBRARY_PATH:-}"
echo "[$(date)] rebuilding contact sheets"
"$CP/bin/python" make_contact_sheets_v5.py
echo "[$(date)] rebuilding labeling UI (references views/)"
"$CP/bin/python" make_label_ui.py
echo "[$(date)] rebuilding labeling UI (embedded, portable)"
"$CP/bin/python" make_label_ui.py --embed
echo "[$(date)] done"
