#!/bin/bash
#SBATCH --job-name=ub_profile
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Profile run_shell phase timings for one car.
#   sbatch profile_shell.sbatch <car>
set -euo pipefail
CAR="${1:?usage: sbatch profile_shell.sbatch <car>}"
cd /scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
mkdir -p logs outputs/_profile
PARAMUB_PY=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/bin/python
export LD_LIBRARY_PATH=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[$(date)] node=$(hostname) car=${CAR}"
$PARAMUB_PY -u run_shell_profiled.py \
    --input "data/data_upload/${CAR}.stl" \
    --output-dir "outputs/_profile/${CAR}"
echo "[$(date)] done ${CAR}"
