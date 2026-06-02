#!/bin/bash
#SBATCH --job-name=ub_forceflip
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Force-flip the 3 cab-at-+x reversals through the full v5 pipeline.
set -euo pipefail
REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB; cd "$REPO"
LIST=rerun_forceflip.txt
OUTROOT=outputs/batch_v5_reversed_fix
mapfile -t CARS < "$LIST"
CAR="${CARS[$SLURM_ARRAY_TASK_ID]}"
INPUT="data/data_upload/${CAR}.stl"
OUT="${OUTROOT}/${CAR}"; INTEG="${OUT}/integrate"
FINAL="${INTEG}/${CAR}_clean_deflector.stl"
mkdir -p logs
echo "[$(date)] node=$(hostname) FORCE-FLIP car=${CAR}"
CP=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub
PY="$CP/bin/python"; export CONDA_PREFIX=$CP LD_LIBRARY_PATH="$CP/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENVDB_THREADS=1
# Wipe any prior (non-flipped) outputs for this car so the pipeline regenerates.
rm -rf "$OUT"
"$PY" -u make_pipeline.py --input "$INPUT" --outputs-root "$OUTROOT" \
    --depth 10 --no-verify-wheels --force-flip
"$PY" make_deflector.py --shell-meta "${OUT}/shell/${CAR}_meta.json" --output-dir "$INTEG"
"$PY" -u trim_deflector.py --body "${INTEG}/${CAR}_clean.stl" \
    --deflector "${INTEG}/${CAR}_deflector.stl" --out "$FINAL" --solver FAST
echo "[$(date)] DONE force-flip ${CAR} -> ${FINAL}"
