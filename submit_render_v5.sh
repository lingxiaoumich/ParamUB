#!/bin/bash
# Build the car list from on-disk _clean_deflector.stl files, then submit
# render_array_v5.sbatch sized to that list at %18 concurrency.
# Output goes to outputs/batch_v5_summary/ (separate from batch_v1 summary).
#
# Usage:
#   ./submit_render_v5.sh              # defaults below
#   CONCURRENCY=8 ./submit_render_v5.sh
set -euo pipefail

REPO=/scratch/jjparkcv_owned_root/jjparkcv_owned1/lxxiao/ParamUB
cd "$REPO"

BATCH_OUT="${BATCH_OUT:-outputs/batch_v5}"
OUTDIR="${OUTDIR:-outputs/batch_v5_summary}"
LIST="render_list_v5.txt"
CONCURRENCY="${CONCURRENCY:-18}"

# Build list from finished _clean_deflector.stl
printf "" > "$LIST"
for f in "${BATCH_OUT}"/*/integrate/*_clean_deflector.stl; do
  [ -f "$f" ] || continue
  base=$(basename "$f" _clean_deflector.stl)
  echo "$base"
done | sort > "$LIST"

COUNT=$(wc -l < "$LIST")
if [ "$COUNT" -eq 0 ]; then
  echo "error: no _clean_deflector.stl found under $BATCH_OUT" >&2
  exit 1
fi

DONE=0
for car in $(cat "$LIST"); do
  [ -f "${OUTDIR}/views/bottom_front_iso/${car}.png" ] && DONE=$((DONE+1))
done

echo "[submit] cars with clean_deflector.stl : $COUNT"
echo "[submit] already rendered               : $DONE"
echo "[submit] to render                      : $((COUNT - DONE))"
echo "[submit] list -> $LIST"
echo "[submit] outdir -> $OUTDIR"

mkdir -p logs "$OUTDIR"

export RENDER_LIST="$LIST"
export BATCH_OUT OUTDIR
sbatch --export=ALL \
       --array=0-$((COUNT-1))%${CONCURRENCY} \
       render_array_v5.sbatch
