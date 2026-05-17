"""Sweep stl_tolerance across a few values and report mesh stats.

Runs the build + export directly (no trimesh repair, no render) at
several tolerance values. For each tolerance: builds the assembly, exports
the STL, then computes face count and file size from the STL header.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cadquery as cq
from cadquery import exporters

from underbody.build import UnderbodySpec, compute_layout, build_underbody


SWEEP = [0.5, 0.1]   # coarse -> fine


def stl_face_count(path: Path) -> int:
    """Read the face count from a binary STL header (no full mesh load)."""
    with path.open("rb") as fh:
        fh.read(80)   # header
        return struct.unpack("<I", fh.read(4))[0]


def main():
    output_dir = REPO_ROOT / "outputs" / "underbody"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tol in SWEEP:
        spec = UnderbodySpec(stl_tolerance=tol)
        stl_path = output_dir / f"underbody_t{tol:g}.stl"
        print(f"\n=== tolerance = {tol} ===", flush=True)

        t0 = time.time()
        asy, _layout = build_underbody(spec)
        t_build = time.time() - t0
        print(f"  build: {t_build:.1f}s", flush=True)

        t1 = time.time()
        compound = asy.toCompound()
        exporters.export(
            compound,
            str(stl_path),
            exporters.ExportTypes.STL,
            tolerance=tol,
            angularTolerance=tol,
        )
        t_export = time.time() - t1
        print(f"  export: {t_export:.1f}s", flush=True)

        faces = stl_face_count(stl_path)
        size_mb = stl_path.stat().st_size / (1024 * 1024)
        print(f"  faces = {faces:,}  size = {size_mb:.1f} MB",
              flush=True)
        rows.append((tol, faces, size_mb, t_build, t_export))

    print("\n=== SUMMARY ===")
    print(f"{'tolerance':>10}  {'faces':>12}  {'STL MB':>8}  "
          f"{'build':>7}  {'export':>7}")
    for tol, faces, size_mb, t_build, t_export in rows:
        print(f"{tol:>10.4f}  {faces:>12,}  {size_mb:>7.1f}M  "
              f"{t_build:>6.1f}s  {t_export:>6.1f}s")


if __name__ == "__main__":
    main()
