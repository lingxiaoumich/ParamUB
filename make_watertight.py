"""Driver for the watertight finisher — stage 4 of the
reconstructed-car pipeline.

Reads the combined STL produced by ``fill_gap.py`` (shell + UB +
ribbons, ~5 k open edges / ~1 k non-manifold edges), runs the smooth
dual-contouring remesh + symmetrize + non-manifold cleanup loop, then
extracts the biggest connected body and writes a strictly watertight,
bilaterally symmetric, single-body STL ready for CFD / 3-D printing /
volume computation. See :mod:`paramub.watertight` for the algorithm.

Pipeline summary
----------------
::

    Input STL (non-watertight, many bodies, gaps)
            │
            ▼
       ┌─────────────────────────────────────────────────┐
       │ STAGE A : Blender (bpy 4.2)                     │
       │  1. import STL                                  │
       │  2. Smooth Remesh modifier (dual-contouring)    │
       │  3. keep largest connected component            │
       │  4. merge / dissolve / fill_holes / normals     │
       │  5. symmetrize at y=0 (NEGATIVE_Y)              │
       │  6. iterative non-manifold edge cleanup         │
       │  7. keep largest connected component            │
       │  8. export STL  (<stem>_fixed.stl)              │
       └─────────────────────────────────────────────────┘
            │
            ▼
       ┌─────────────────────────────────────────────────┐
       │ STAGE B : trimesh                               │
       │  9. split by connected components               │
       │ 10. keep the largest body                       │
       │ 11. verify watertight + record metrics          │
       │ 12. export STL  (<stem>_clean.stl) + JSON       │
       └─────────────────────────────────────────────────┘
            │
            ▼
    Output : <stem>_clean.stl   strictly watertight, single body
             <stem>_clean.json  verify report

Usage
-----
A single Slurm node at depth 11 is the recommended setup (the Blender
peak is ~96 GB at 12.8 M tris). See ``make_watertight.sbatch`` for
the wrapper. Direct invocation (lighter depths on the login node)::

    conda activate paramub
    python make_watertight.py \\
        --input  outputs/integrate/<base>_combined.stl \\
        --output outputs/integrate/<base>_clean.stl \\
        --depth 11

The intermediate ``<base>_fixed.stl`` (Blender-stage output) is
written next to ``--output`` by default; override with
``--fixed-out``. A JSON verify report is written alongside
``--output`` with a ``.json`` extension (override with ``--json``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from paramub.watertight import (
    WHEEL_CORNERS, discover_wheel_paths, make_watertight,
    make_wheel_watertight,
)


DEFAULT_BASE = (
    "outputs/integrate/"
    "alfa_romeo_giuliazhuliye_2025_image10_71415_shadowfill"
)


def main():
    p = argparse.ArgumentParser(
        description="Stage 4: turn the fill_gap combined STL into a "
                    "strictly watertight single-body STL.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path(f"{DEFAULT_BASE}_combined.stl"),
                   help="Combined STL from fill_gap.py.")
    p.add_argument("--output", type=Path,
                   default=Path(f"{DEFAULT_BASE}_clean.stl"),
                   help="Final strictly-watertight single-body STL.")
    p.add_argument("--fixed-out", type=Path, default=None,
                   help="Intermediate Blender-stage STL. Defaults to "
                        "`<output>_fixed.stl` next to --output.")
    p.add_argument("--json", type=Path, default=None,
                   help="Verify report JSON path. Defaults to "
                        "`<output>.json` next to --output.")
    p.add_argument("--reference", type=Path, default=None,
                   help="Optional reference STL (usually the same as "
                        "--input) to include surface-distance metrics "
                        "in the verify report. Defaults to --input; "
                        "pass an empty string to skip.")
    p.add_argument("--n-samples", type=int, default=30_000,
                   help="Points sampled per surface for distance metrics.")
    # Blender stage knobs (forwarded to smooth_remesh)
    p.add_argument("--depth", type=int, default=11,
                   help="Octree depth for the Smooth Remesh modifier "
                        "(2^depth cells across the longest bbox axis). "
                        "Memory/time scale ~8x per +1. Default 11.")
    p.add_argument("--scale", type=float, default=0.99,
                   help="Remesh modifier grid scale (0..0.99). "
                        "Lower = more bbox padding.")
    p.add_argument("--no-symmetrize", action="store_true",
                   help="Skip the y=0 symmetry pass.")
    p.add_argument("--symmetry-direction", default="NEGATIVE_Y",
                   choices=["POSITIVE_X", "NEGATIVE_X",
                            "POSITIVE_Y", "NEGATIVE_Y",
                            "POSITIVE_Z", "NEGATIVE_Z"],
                   help="Source side for symmetrize. Default NEGATIVE_Y "
                        "= keep y<0, mirror to y>0.")
    p.add_argument("--no-fix-nonmanifold", action="store_true",
                   help="Skip the iterative non-manifold edge cleanup.")
    p.add_argument("--max-nm-passes", type=int, default=8,
                   help="Max passes of the non-manifold fix loop. "
                        "Lower default (8) because the loop is "
                        "known to diverge on some inputs; raise if "
                        "your input genuinely needs more.")
    p.add_argument("--mode", choices=("auto", "lightweight", "full"),
                   default="auto",
                   help="Stage-A strategy. auto (default) = lightweight "
                        "if input is already is_watertight, else full. "
                        "lightweight = Blender symmetrize only, no "
                        "dual-contour remesh (preserves all input "
                        "detail). full = smooth dual-contour remesh + "
                        "symmetrize + iterative NM-fix.")
    p.add_argument("--merge-distance", type=float, default=1e-3,
                   help="Vertex merge threshold in mm.")
    p.add_argument("--fill-sides", type=int, default=16,
                   help="Max hole size (in edges) for fill_holes.")
    # Wheel processing (4 corners adjacent to --input)
    p.add_argument("--include-wheels", action="store_true", default=True,
                   help="Also process the 4 wheel STLs adjacent to "
                        "--input (named <base>_wheel_<corner>.stl). "
                        "Each wheel runs through a trimesh-only "
                        "fill_holes + verify pipeline (no Blender "
                        "remesh — wheels are CadQuery-tessellated and "
                        "only have ~6 chord-discretization open edges). "
                        "Default ON.")
    p.add_argument("--no-wheels", action="store_false", dest="include_wheels",
                   help="Skip the wheel processing stage.")
    args = p.parse_args()

    fixed_path = (args.fixed_out
                  if args.fixed_out is not None
                  else args.output.with_name(args.output.stem + "_fixed.stl"))
    json_path = (args.json
                 if args.json is not None
                 else args.output.with_suffix(".json"))
    reference = (args.reference
                 if args.reference is not None
                 else args.input)

    make_watertight(
        args.input,
        fixed_stl=fixed_path,
        clean_stl=args.output,
        report_json=json_path,
        reference_stl=str(reference) if reference else None,
        n_samples=args.n_samples,
        mode=args.mode,
        depth=args.depth,
        scale=args.scale,
        symmetrize=not args.no_symmetrize,
        symmetry_direction=args.symmetry_direction,
        fix_nonmanifold=not args.no_fix_nonmanifold,
        max_nm_passes=args.max_nm_passes,
        merge_distance=args.merge_distance,
        fill_sides=args.fill_sides,
    )

    # ---- Wheel stage (cheap trimesh-only repair on each of 4 corners) ----
    if args.include_wheels:
        wheels = discover_wheel_paths(args.input)
        if not wheels:
            print(f"[wheels] no wheel STLs found alongside {args.input} "
                  f"(expected <base>_wheel_{{{','.join(WHEEL_CORNERS)}}}.stl) "
                  f"— skipping.", flush=True)
        else:
            print(f"[wheels] processing {len(wheels)} wheel(s): "
                  f"{', '.join(wheels)}", flush=True)
            out_stem = args.output.with_suffix("").name
            if out_stem.endswith("_clean"):
                out_stem = out_stem[: -len("_clean")]
            for corner, src in wheels.items():
                out_path = args.output.with_name(
                    f"{out_stem}_wheel_{corner}_clean.stl")
                report_path = out_path.with_suffix(".json")
                make_wheel_watertight(
                    src, out_path,
                    report_json=report_path,
                    reference_stl=str(src),
                    n_samples=args.n_samples,
                )


if __name__ == "__main__":
    main()
