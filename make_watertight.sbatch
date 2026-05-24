#!/bin/bash
#SBATCH --job-name=watertight
#SBATCH --account=jjparkcv_owned1
#SBATCH --qos=arph
#SBATCH --partition=standard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 4 of the reconstructed-car pipeline (see paramub/docs/watertight.html).
# Runs Blender smooth remesh + trimesh extract+verify on the
# combined STL produced by fill_gap.py.
#
# Usage:
#   sbatch make_watertight.sbatch <combined.stl> <output-tag> [extra args]
#
# Extra args are forwarded to make_watertight.py — most useful:
#   --depth N             octree depth (default 11; lower → smaller file)
#   --no-symmetrize       skip y=0 symmetry pass
#   --no-fix-nonmanifold  skip the non-manifold fix loop
#
# Outputs (in outputs/integrate/<tag>_*):
#   <tag>_fixed.stl   Blender-stage output (closed per Blender topology)
#   <tag>_clean.stl   final strictly-watertight single-body STL
#   <tag>_clean.json  verify report (watertight / volume / surface distance)

set -euo pipefail

INPUT="$1"; shift
TAG="$1"; shift
EXTRA_ARGS=("$@")

OUT_DIR="$(dirname "$INPUT")"
FIXED="$OUT_DIR/${TAG}_fixed.stl"
CLEAN="$OUT_DIR/${TAG}_clean.stl"
REPORT="$OUT_DIR/${TAG}_clean.json"

mkdir -p logs "$OUT_DIR"

echo "[$(date)] node=$(hostname) input=$INPUT tag=$TAG"
echo "[$(date)] CPUs=$SLURM_CPUS_PER_TASK MEM=${SLURM_MEM_PER_NODE}MB"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENVDB_THREADS=$SLURM_CPUS_PER_TASK

# Use the paramub env's python by absolute path (see make_pipeline.sbatch
# for why activation isn't enough inside a Slurm batch shell).
PARAMUB_PY=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/bin/python
export LD_LIBRARY_PATH=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/miniconda3/envs/paramub/lib:${LD_LIBRARY_PATH:-}

$PARAMUB_PY -u make_watertight.py \
    --input     "$INPUT" \
    --output    "$CLEAN" \
    --fixed-out "$FIXED" \
    --json      "$REPORT" \
    --reference "$INPUT" \
    "${EXTRA_ARGS[@]}"

echo "[$(date)] done"
ls -lh "$FIXED" "$CLEAN" "$REPORT"
