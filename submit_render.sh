#!/bin/bash
# Build the list of successful cars (those with a clean body STL) and
# submit render_array.sbatch as a Slurm array sized to that list.
#
# Usage:
#   ./submit_render.sh                 # all cars with a clean.stl, 10-wide
#   CONCURRENCY=6 ./submit_render.sh   # gentler

set -euo pipefail

LIST="rendered_cars.txt"
CONCURRENCY="${CONCURRENCY:-10}"

ls outputs/*/integrate/*_clean.stl 2>/dev/null \
  | sed -E 's#.*/([^/]+)_clean\.stl#\1#' \
  | sort -u > "$LIST"

COUNT=$(wc -l < "$LIST")
if [ "$COUNT" -eq 0 ]; then
  echo "error: no successful cars (no *_clean.stl found)" >&2
  exit 1
fi
LAST=$((COUNT - 1))

# Count how many still need rendering.
TODO=0
while read -r car; do
  [ -f "outputs/summary/renders/${car}_10view.png" ] || TODO=$((TODO + 1))
done < "$LIST"

echo "[render] successful cars=${COUNT}  to_render=${TODO}"
echo "[render] array=0-${LAST}%${CONCURRENCY}"

export RENDER_LIST="$LIST"
sbatch --export=ALL --array=0-${LAST}%${CONCURRENCY} render_array.sbatch
