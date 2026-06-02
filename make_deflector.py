"""Standalone front-tire-deflector generator.

Builds the two front tire deflectors for a car as a SEPARATE STL, in the
same shell-aligned frame / scale that ``integrate_underbody.py`` produces
its underbody in — so the deflector STL drops straight onto the integrated
body. NO trimming is applied: the raw plates are written as-is, to be
trimmed into the floor+splitter+diffuser surface as a final step (e.g.
after stage 4) when desired.

Reuses ``integrate_underbody``'s own helpers (``extract_hints``,
``measure_shell_anchors``, ``build_spec``, ``align_to_shell_frame``,
``cq_obj_to_trimesh_via_stl``) so the deflector's axle/tire/arch geometry
and the shell-frame transform exactly match the underbody it sits against.

Runs fine on a login node (no heavy boolean / remesh) — no Slurm needed.

Usage::

    python make_deflector.py \
        --shell-meta outputs/0mjLG_v2/shell/0mjLG_meta.json \
        --output-dir outputs/0mjLG_deflector/integrate
    # writes <output-dir>/0mjLG_deflector.stl
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

import cadquery as cq

from integrate_underbody import (
    TARGET_WHEELBASE_MM,
    align_to_shell_frame,
    build_spec,
    cq_obj_to_trimesh_via_stl,
    extract_hints,
    measure_shell_anchors,
)
from paramub.builders.deflector import (
    FrontDeflectorSpec,
    build_front_deflector_solid,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shell-meta", type=Path, required=True,
                   help="Shell metadata JSON (e.g. outputs/<car>_v2/shell/"
                        "<car>_meta.json) — same input integrate uses.")
    p.add_argument("--output-dir", "-o", type=Path, required=True,
                   help="Directory to write <base>_deflector.stl into.")
    p.add_argument("--front-gap-mm", type=float, default=20.0,
                   help="Gap ahead of the wheelhouse front wall. Default 20.")
    p.add_argument("--thickness-mm", type=float, default=3.0,
                   help="Plate thickness in X. Default 3.")
    p.add_argument("--outboard-inset-mm", type=float, default=120.0,
                   help="Outboard bound, inboard of the tire OUTBOARD edge. "
                        "Default 120.")
    p.add_argument("--inboard-inset-mm", type=float, default=50.0,
                   help="Inboard bound, inboard of the tire INBOARD edge "
                        "(toward centerline). Default 50.")
    p.add_argument("--drop-mm", type=float, default=40.0,
                   help="How far below the floor the plate hangs. Default 40.")
    p.add_argument("--top-extend-mm", type=float, default=50.0,
                   help="How far above the floor the plate rises (for clean "
                        "later trimming). Default 50.")
    p.add_argument("--radial-clearance-mm", type=float, default=80.0,
                   help="Wheelhouse radial clearance (arch_r = tire_r + this); "
                        "must match the integrate run. Default 80.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    meta = json.loads(args.shell_meta.read_text())
    shell_stl_path = Path(meta["final_path"])
    if not shell_stl_path.is_absolute():
        shell_stl_path = args.shell_meta.parent / shell_stl_path.name
    shell = trimesh.load(str(shell_stl_path), force="mesh", process=False)
    print(f"[load] shell = {shell_stl_path}  faces={len(shell.faces):,}")

    # Scale so wheelbase = 2700 mm (identical to integrate_underbody.main).
    wheels = sorted(meta["wheels_3d"], key=lambda w: w["x"])
    wb_shell = abs(wheels[-1]["x"] - wheels[0]["x"])
    scale = TARGET_WHEELBASE_MM / wb_shell
    shell_mm = trimesh.Trimesh(vertices=np.asarray(shell.vertices) * scale,
                               faces=np.asarray(shell.faces), process=False)

    hints = extract_hints(shell, meta, scale)
    y_int_splitter = hints["front_wheel_y_mm"]
    y_int_diffuser = hints["rear_wheel_y_mm"]
    section_ys = sorted({0.0, y_int_splitter, y_int_diffuser})
    anchors = measure_shell_anchors(shell_mm, ys=section_ys)
    spec = build_spec(hints, anchors,
                      y_intermediate_splitter=y_int_splitter,
                      y_intermediate_diffuser=y_int_diffuser,
                      radial_clearance_mm=args.radial_clearance_mm)
    midpoint_x = hints["midpoint_x_shell_mm"]

    # Front-axle geometry (matches _make_wheelhouse_specs / WheelhouseSpec).
    front_axle_x = +spec.wheelbase_mm / 2.0
    tire_radius = spec.wheel.tire_radius_mm
    tire_width = spec.wheel.tire_section_width_mm
    arch_radius = tire_radius + spec.wheel_house_radial_clearance_mm
    ride_h = spec.ride_height_mm
    print(f"[geom] front_axle_x={front_axle_x:.1f}  "
          f"track_front={spec.track_front_mm:.1f}  tire_w={tire_width:.1f}  "
          f"tire_r={tire_radius:.1f}  arch_r={arch_radius:.1f}  "
          f"ride_h={ride_h:.1f}")

    solids = []
    for side in ("left", "right"):
        sign = +1.0 if side == "right" else -1.0
        d = FrontDeflectorSpec(
            axle_x=front_axle_x,
            y_track=sign * spec.track_front_mm / 2.0,
            side=side,
            tire_width_mm=tire_width,
            arch_radius_mm=arch_radius,
            ride_height_mm=ride_h,
            front_gap_mm=args.front_gap_mm,
            thickness_mm=args.thickness_mm,
            outboard_inset_mm=args.outboard_inset_mm,
            inboard_inset_mm=args.inboard_inset_mm,
            drop_mm=args.drop_mm,
            top_extend_mm=args.top_extend_mm,
        )
        xb, yb, zb = d.x_bounds_mm, d.y_bounds_mm, d.z_bounds_mm
        print(f"[deflector] {side}: X=[{xb[0]:.1f},{xb[1]:.1f}] "
              f"Y=[{yb[0]:.1f},{yb[1]:.1f}] (w={yb[1]-yb[0]:.1f}) "
              f"Z=[{zb[0]:.1f},{zb[1]:.1f}] (below {ride_h-zb[0]:.0f}, "
              f"above {zb[1]-ride_h:.0f})")
        solids.append(build_front_deflector_solid(d).val())

    compound = cq.Compound.makeCompound(solids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.output_dir / "_scratch_defl"
    scratch.mkdir(parents=True, exist_ok=True)
    raw = cq_obj_to_trimesh_via_stl(compound, scratch / "deflectors.stl",
                                    tolerance=0.1, angular_tolerance=0.1)
    aligned = align_to_shell_frame(raw, midpoint_x)

    base = args.shell_meta.stem.replace("_meta", "")
    out = args.output_dir / f"{base}_deflector.stl"
    aligned.export(str(out), file_type="stl")
    b = aligned.bounds
    print(f"[out] {out}  ({len(aligned.faces):,} faces)  "
          f"bounds X[{b[0,0]:.0f},{b[1,0]:.0f}] Y[{b[0,1]:.0f},{b[1,1]:.0f}] "
          f"Z[{b[0,2]:.0f},{b[1,2]:.0f}]")

    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
