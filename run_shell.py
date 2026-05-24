"""Driver for the upper-body shell extraction pipeline.

Runs the full-car (no y=0 cut) extraction on a source STL and writes
the final upper shell to ``outputs/shell/``.

Pipeline summary (see :mod:`paramub.shell_extract` for the full step
docstrings):

    1.  canonicalize         — rotate to canonical (+x rearward, +y
                                lateral, +z up); ground-snap to z=0;
                                centre on x=0
    3.  wheel (x, y)         — z-slab cluster
    4.  wheel hub Z + radius — Y-slab + component-from-contact + circle fit
    5.  cylinder remove      — tight tire-hugging cylinder + keep-largest
    6.  wheelhouse remove    — 6a wheel-center rays + 6b zone-inward
    7.  lower cut            — two-hemisphere visibility classifier
                                constrained to a cut zone
                                (floor band ∪ enlarged wheel cylinders)
    8.  rim re-cut           — extract the longest open-boundary loop of
                                the step-7 shell and use it to flood-cut
                                the canonical (raw) mesh — fills in any
                                small holes accidentally introduced by
                                steps 5-7.

Usage::

    conda activate paramub
    python run_shell.py                  # production (final STL only)
    python run_shell.py --debug          # also export per-step STLs
    python run_shell.py --render         # also render per-step PNGs
    python run_shell.py --debug --render # both
    python run_shell.py --input X --output Y

Outputs (always):
    outputs/shell/<basename>_final.stl   — single-component upper shell
                                            (raw output of steps 1, 3–7)
    outputs/shell/<basename>_upper_recut.stl
                                         — re-cut from canonical mesh
                                            using the longest boundary
                                            loop of _final.stl as a
                                            3-D cutter (no small holes)
    outputs/shell/<basename>_meta.json   — numeric metadata

Outputs (--debug):
    outputs/shell/<basename>_debug/step1_canonical.stl
                                  /step5_kept.stl
                                  /step5_wheel_cylinders.stl
                                  /step6a_kept.stl
                                  /step6b_kept.stl
                                  /step7_cut_zone_volume.stl
                                  /step7_cut_zone_faces.stl
                                  /step7_keep_raw.stl
                                  /step7_remove.stl
                                  /step7_drop.stl
                                  /step7_boundary.stl

Outputs (--render):
    outputs/shell/<basename>_renders/stepN_*.png  (10-panel composites)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from paramub.shell.extract import (
    LABEL_KEEP, LABEL_REMOVE, LABEL_DROP,
    build_cut_zone, build_cut_zone_volume_mesh,
    edges_to_tube_mesh,
    keep_largest_component, split_keep_remove,
    s1_canonicalize,
    s3_locate_wheel_xy_full, s4_locate_wheel_z,
    s5_cylinder_remove,
    s6a_wheel_center_visibility, s6b_wheelhouse_zone_inward,
    s7_classify_visibility,
    trace_boundary_edges,
)
from paramub.shell.recut import (
    recut_with_longest_rim,
    find_open_boundary_components,
    export_edges_as_tubes,
)


DEFAULT_INPUT = Path(
    "data/alfa_romeo_giuliazhuliye_2025_image10_71415_shadowfill.stl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/shell")

# Symmetry-plane tolerance — vertices with |y| < this are considered on y=0.
PLANE_EPS = 1e-6
# Render decimation target (face count) for the matplotlib renderer.
RENDER_DECIMATE_TARGET = 30_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_submesh(mesh: trimesh.Trimesh, face_idx: np.ndarray,
                    out_path: Path) -> int:
    """Write ``mesh.submesh([face_idx])`` to ``out_path``. Returns face count."""
    if len(face_idx) == 0:
        return 0
    sub = mesh.submesh([face_idx], append=True)
    if isinstance(sub, list):
        sub = sub[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.export(str(out_path), file_type="stl")
    return int(len(sub.faces))


def _export_keep(mesh: trimesh.Trimesh, remove_mask: np.ndarray,
                 out_path: Path) -> int:
    return _export_submesh(mesh, np.flatnonzero(~remove_mask), out_path)


def _decimate_via_pyvista_subproc(in_stl: Path, out_stl: Path,
                                   target_faces: int) -> None:
    """Decimate ``in_stl`` to ~``target_faces`` via VTK (PyVista). Run
    in a subprocess to keep VTK out of this process."""
    src = """
import sys
import pyvista as pv
in_p, out_p, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
mesh = pv.read(in_p)
n0 = mesh.n_cells
if n0 <= target:
    mesh.save(out_p, binary=True)
else:
    mesh.decimate(1.0 - target / float(n0)).save(out_p, binary=True)
"""
    cmd = [sys.executable, "-c", src,
           str(in_stl), str(out_stl), str(target_faces)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"decimate subprocess failed: {proc.stderr[-1000:]}")


def _decimate_to_stl(mesh: trimesh.Trimesh, target: int,
                     scratch_dir: Path, name: str) -> Path:
    """Write ``mesh`` then decimate to ``target`` faces. Returns the
    decimated STL path."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    full = scratch_dir / f"{name}.full.stl"
    deci = scratch_dir / f"{name}.stl"
    mesh.export(str(full), file_type="stl")
    if len(mesh.faces) > target:
        _decimate_via_pyvista_subproc(full, deci, target)
    else:
        mesh.export(str(deci), file_type="stl")
    full.unlink(missing_ok=True)
    return deci


def _render_panel(*, body_stl: Path, remove_mask: np.ndarray, title: str,
                  out_path: Path, view_bounds: np.ndarray,
                  scratch_dir: Path, **overlay_kwargs) -> None:
    """Spawn one paramub.shell_render subprocess to render a 10-panel
    composite PNG. Overlay kwargs are passed through to the payload
    (see :func:`paramub.shell_render.main` for the schema)."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stl_path": str(body_stl),
        "remove_indices": np.flatnonzero(remove_mask).tolist(),
        "title": title,
        "out_path": str(out_path),
        "view_bounds": view_bounds.tolist(),
        **overlay_kwargs,
    }
    # Normalize wheel dataclass lists -> dict.
    for k in ("wheels_xy", "wheels_3d"):
        v = payload.get(k)
        if v is None:
            continue
        payload[k] = [asdict(w) if hasattr(w, "__dataclass_fields__")
                       else dict(w) for w in v]
    if payload.get("extra_origin") is not None:
        payload["extra_origin"] = list(payload["extra_origin"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                      dir=str(scratch_dir)) as fh:
        json.dump(payload, fh)
        payload_path = Path(fh.name)
    cmd = [sys.executable, "-m", "paramub.shell_render", str(payload_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    payload_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        print("render subprocess stderr:", proc.stderr[-2000:])
        raise RuntimeError(f"render subprocess failed (rc={proc.returncode})")
    if proc.stdout.strip():
        print(proc.stdout.rstrip())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract upper-body shell from a reconstructed car mesh.")
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                   help=f"Source mesh STL/glb (default {DEFAULT_INPUT}).")
    p.add_argument("--output-dir", "-o", type=Path,
                   default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory (default {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--debug", action="store_true",
                   help="Export per-step intermediate STLs to "
                        "<output>/<name>_debug/.")
    p.add_argument("--render", action="store_true",
                   help="Render per-step PNG composites to "
                        "<output>/<name>_renders/.")
    p.add_argument("--prefilter-normal-z-max", type=float, default=-0.30,
                   help="Step 7 prefilter: faces with normal_z below this "
                        "and z in the bottom prefilter_z_band_frac of the "
                        "envelope are pre-marked lower_visible (cheap "
                        "underbody flag). More negative (closer to -1) = "
                        "stricter, fewer pre-marks. Default -0.30.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    basename = args.input.stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / f"{basename}_debug"
    render_dir = output_dir / f"{basename}_renders"
    scratch_dir = output_dir / f"{basename}_scratch"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if args.render:
        render_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {"input": str(args.input)}

    # =====================================================================
    # Step 1 — canonicalize
    # =====================================================================
    body1, frame = s1_canonicalize(args.input, flip_length=True)
    meta["frame_R"] = frame.R.tolist()
    meta["frame_t"] = frame.t.tolist()
    meta["original_bounds"] = frame.original_bounds.tolist()
    meta["canonical_bounds"] = frame.canonical_bounds.tolist()
    meta["step1_faces"] = int(len(body1.faces))
    # Per-step view bounds: each step's renders use bounds based on what is
    # actually visible in that step (mesh + any cylinder overlays). The
    # canonical full-car bounds are kept around for steps 1-4 (where the body
    # is still the full canonical mesh). Steps 5-7 use the post-removal mesh
    # bounds so the upper-shell panels are not unnecessarily zoomed out.
    canonical_bounds = body1.bounds.copy()

    if args.debug:
        body1.export(str(debug_dir / "step1_canonical.stl"), file_type="stl")
        print(f"  [debug] step1_canonical.stl ({len(body1.faces):,} faces)")
    if args.render:
        body1_deci = _decimate_to_stl(body1, RENDER_DECIMATE_TARGET,
                                      scratch_dir, "step1_body")
        _render_panel(
            body_stl=body1_deci,
            remove_mask=np.zeros(_load_face_count(body1_deci), dtype=bool),
            title="Step 1 — canonical frame (+x rearward, +y lateral, +z up)",
            out_path=render_dir / "step1_canonical.png",
            view_bounds=canonical_bounds,
            scratch_dir=scratch_dir,
            show_axes=True,
        )

    # Step 2 (planar cut at y=0) is intentionally skipped: extract operates
    # on the full canonical mesh. Wheel detection, cylinder removal,
    # wheelhouse removal, and the visibility classifier are all per-wheel
    # or per-face and work on both sides of the car.
    body = body1
    body_deci = body1_deci if args.render else None

    # =====================================================================
    # Step 3 — wheel (x, y) footprints (full car: 4 corners)
    # =====================================================================
    wheels_xy = s3_locate_wheel_xy_full(body, z_band_height=0.02, n_axles=2)
    meta["wheels_xy"] = [asdict(w) for w in wheels_xy]
    if args.render:
        _render_panel(
            body_stl=body_deci,
            remove_mask=np.zeros(_load_face_count(body_deci), dtype=bool),
            title="Step 3 — wheel (x, y) footprints from z-slice (magenta = footprint)",
            out_path=render_dir / "step3_wheel_xy.png",
            view_bounds=body.bounds.copy(),
            scratch_dir=scratch_dir,
            wheels_xy=wheels_xy, wheels_xy_color="#c026d3",
        )

    # =====================================================================
    # Step 4 — wheel hub Z + radius
    # =====================================================================
    wheels = s4_locate_wheel_z(body, wheels_xy)
    meta["wheels_3d"] = [asdict(w) for w in wheels]
    if args.render:
        _render_panel(
            body_stl=body_deci,
            remove_mask=np.zeros(_load_face_count(body_deci), dtype=bool),
            title="Step 4 — refined wheel hub + radius (orange = hub)",
            out_path=render_dir / "step4_wheel_z.png",
            view_bounds=body.bounds.copy(),
            scratch_dir=scratch_dir,
            wheels_3d=wheels, wheels_3d_color="#ea580c",
        )

    # =====================================================================
    # Step 5 — wheel cylinder removal (tight envelope, factors 1.00)
    # =====================================================================
    cyl_rfac, cyl_afac = 1.00, 1.00
    remove5_raw = s5_cylinder_remove(body, wheels,
                                     radius_factor=cyl_rfac,
                                     axial_factor=cyl_afac)
    remove5 = keep_largest_component(body, remove5_raw)
    meta["step5_faces"] = int((~remove5).sum())

    if args.debug:
        _export_keep(body, remove5, debug_dir / "step5_kept.stl")
        # Debug overlay: the cylinder primitives themselves.
        from paramub.shell.extract import _wheel_cyl_y_bounds  # type: ignore
        # Use a thin overlay: the s5 cylinder uses radius_factor=1 and
        # axial_factor=1, symmetric, no outboard extension.
        cyl_overlay = build_cut_zone_volume_mesh(
            wheels, floor_z_max=0.0,        # no floor box
            floor_x_range=(0.0, 0.0),
            floor_y_range=(0.0, 0.0),
            wheel_radius_factor=cyl_rfac,
            wheel_axial_factor=cyl_afac,
            outboard_y_limit=None,
        )
        # Drop the degenerate floor box if it slipped in
        cyl_overlay.export(str(debug_dir / "step5_wheel_cylinders.stl"),
                            file_type="stl")
        print(f"  [debug] step5_kept.stl + step5_wheel_cylinders.stl")
    if args.render:
        body5_kept_mesh, _ = split_keep_remove(body, remove5)
        body5_deci = _decimate_to_stl(body5_kept_mesh, RENDER_DECIMATE_TARGET,
                                      scratch_dir, "step5_body")
        rm5_deci = np.zeros(_load_face_count(body5_deci), dtype=bool)
        _render_panel(
            body_stl=body5_deci,
            remove_mask=rm5_deci,
            title=f"Step 5 — wheel cylinder ({cyl_rfac:.2f}×r, "
                  f"{cyl_afac:.2f}×ax) + keep-largest",
            out_path=render_dir / "step5_cylinder.png",
            view_bounds=body5_kept_mesh.bounds.copy(),
            scratch_dir=scratch_dir,
            wheels_3d=wheels, wheels_3d_color="#ea580c",
            show_cylinders=True,
            cylinder_radius_factor=cyl_rfac,
            cylinder_axial_factor=cyl_afac,
        )

    # =====================================================================
    # Step 6a — wheel-center ray visibility
    # =====================================================================
    r6a = s6a_wheel_center_visibility(
        body, wheels,
        n_rays_per_origin=16384,
        n_upper_edge=6, edge_radial_frac=1.0,
        include_hub_origin=True,
        zone_radius_factor=2.5, zone_axial_factor=2.0,
        z_max_factor=2.2,
        exclude_mask=remove5,
    )
    remove6a = keep_largest_component(body, remove5 | r6a)
    meta["step6a_faces"] = int((~remove6a).sum())
    if args.debug:
        _export_keep(body, remove6a, debug_dir / "step6a_kept.stl")
        print(f"  [debug] step6a_kept.stl")
    if args.render:
        body6a_kept_mesh, _ = split_keep_remove(body, remove6a)
        body6a_deci = _decimate_to_stl(body6a_kept_mesh, RENDER_DECIMATE_TARGET,
                                       scratch_dir, "step6a_body")
        _render_panel(
            body_stl=body6a_deci,
            remove_mask=np.zeros(_load_face_count(body6a_deci), dtype=bool),
            title="Step 6a — wheel-center ray visibility",
            out_path=render_dir / "step6a_wheel_rays.png",
            view_bounds=body6a_kept_mesh.bounds.copy(),
            scratch_dir=scratch_dir,
            wheels_3d=wheels, wheels_3d_color="#ea580c",
        )

    # =====================================================================
    # Step 6b — wheelhouse-zone inward (geometric backstop)
    # =====================================================================
    r6b = s6b_wheelhouse_zone_inward(body, wheels,
                                      radius_factor=1.4,
                                      axial_factor=1.7,
                                      inward_cos_threshold=0.3,
                                      extend_inboard_to_y=0.0)
    remove6b = keep_largest_component(body, remove6a | r6b)
    meta["step6b_faces"] = int((~remove6b).sum())
    if args.debug:
        _export_keep(body, remove6b, debug_dir / "step6b_kept.stl")
        print(f"  [debug] step6b_kept.stl")
    if args.render:
        body6b_kept_mesh, _ = split_keep_remove(body, remove6b)
        body6b_deci = _decimate_to_stl(body6b_kept_mesh, RENDER_DECIMATE_TARGET,
                                       scratch_dir, "step6b_body")
        _render_panel(
            body_stl=body6b_deci,
            remove_mask=np.zeros(_load_face_count(body6b_deci), dtype=bool),
            title="Step 6b — wheelhouse-zone inward (geometric backstop, "
                  "inboard extended to y=0)",
            out_path=render_dir / "step6b_wheelhouse_zone.png",
            view_bounds=body6b_kept_mesh.bounds.copy(),
            scratch_dir=scratch_dir,
            wheels_3d=wheels, wheels_3d_color="#ea580c",
        )

    # =====================================================================
    # Step 7 — lower-perimeter cut via visibility classifier + cut zone
    # =====================================================================
    keep6_idx = np.flatnonzero(~remove6b)
    body6 = body.submesh([keep6_idx], append=True)
    if isinstance(body6, list):
        body6 = body6[0]
    meta["step7_input_faces"] = int(len(body6.faces))

    CUT_ZONE_FLOOR_FRAC = 0.15
    CUT_ZONE_WHEEL_FAC = 1.3
    floor_z_max = (body6.bounds[0, 2]
                   + CUT_ZONE_FLOOR_FRAC
                   * (body6.bounds[1, 2] - body6.bounds[0, 2]))
    # Full car: outboard cuts extend to both ±y bounds.
    outboard_y_limit = (float(body6.bounds[0, 1]),
                        float(body6.bounds[1, 1]))
    print(f"\n[step7] cut zone: floor z<{floor_z_max:.4f} "
          f"({CUT_ZONE_FLOOR_FRAC:.0%} of body), wheel cyl "
          f"{CUT_ZONE_WHEEL_FAC}× r, inboard {CUT_ZONE_WHEEL_FAC}× ax, "
          f"outboard ends → y∈[{outboard_y_limit[0]:+.4f}, "
          f"{outboard_y_limit[1]:+.4f}]")
    cut_zone = build_cut_zone(body6, wheels,
                              floor_z_max=floor_z_max,
                              wheel_radius_factor=CUT_ZONE_WHEEL_FAC,
                              wheel_axial_factor=CUT_ZONE_WHEEL_FAC,
                              outboard_y_limit=outboard_y_limit)

    if args.debug:
        cz_volume = build_cut_zone_volume_mesh(
            wheels, floor_z_max=floor_z_max,
            floor_x_range=(float(body6.bounds[0, 0]),
                           float(body6.bounds[1, 0])),
            floor_y_range=(float(body6.bounds[0, 1]),
                           float(body6.bounds[1, 1])),
            wheel_radius_factor=CUT_ZONE_WHEEL_FAC,
            wheel_axial_factor=CUT_ZONE_WHEEL_FAC,
            outboard_y_limit=outboard_y_limit,
        )
        cz_volume.export(str(debug_dir / "step7_cut_zone_volume.stl"),
                          file_type="stl")
        _export_submesh(body6, np.flatnonzero(cut_zone),
                         debug_dir / "step7_cut_zone_faces.stl")
        print(f"  [debug] step7_cut_zone_volume.stl + step7_cut_zone_faces.stl")

    labels, upper_visible, lower_visible = s7_classify_visibility(
        body6,
        n_dirs=96,
        equator_band_deg=15.0,
        min_outward_cos=0.05,
        eps_rel=1e-5,
        prefilter=True,
        prefilter_z_band_frac=0.31,
        prefilter_normal_z_max=args.prefilter_normal_z_max,
        cut_zone=cut_zone,
    )
    meta["step7_prefilter_normal_z_max"] = args.prefilter_normal_z_max

    keep_idx = np.flatnonzero(labels == LABEL_KEEP)
    remove_idx = np.flatnonzero(labels == LABEL_REMOVE)
    drop_idx = np.flatnonzero(labels == LABEL_DROP)
    meta["step7_classifier"] = {
        "keep": int(len(keep_idx)),
        "remove": int(len(remove_idx)),
        "drop": int(len(drop_idx)),
    }

    if args.debug:
        _export_submesh(body6, keep_idx, debug_dir / "step7_keep_raw.stl")
        _export_submesh(body6, remove_idx, debug_dir / "step7_remove.stl")
        _export_submesh(body6, drop_idx, debug_dir / "step7_drop.stl")
        # Boundary edges between KEEP and not-KEEP, exported as a
        # thin-tube mesh for visualization.
        cut_mask = labels != LABEL_KEEP
        boundary_edges = trace_boundary_edges(body6, cut_mask)
        if len(boundary_edges) > 0:
            diag6 = float(np.linalg.norm(body6.bounds[1] - body6.bounds[0]))
            tube = edges_to_tube_mesh(body6.vertices, boundary_edges,
                                       radius=0.001 * diag6, sections=4)
            tube.export(str(debug_dir / "step7_boundary.stl"), file_type="stl")
        meta["step7_boundary_edges"] = int(len(boundary_edges))
        print(f"  [debug] step7_keep_raw/remove/drop/boundary STLs")

    # Final upper shell: KEEP-only + keep-largest. May still have small
    # holes where steps 5-7 over-removed; step 8 (rim re-cut) plugs those.
    not_keep_mask = labels != LABEL_KEEP
    final_remove = keep_largest_component(body6, not_keep_mask)
    final_keep_idx = np.flatnonzero(~final_remove)
    final_mesh = body6.submesh([final_keep_idx], append=True)
    if isinstance(final_mesh, list):
        final_mesh = final_mesh[0]

    final_path = output_dir / f"{basename}_final.stl"
    final_mesh.export(str(final_path), file_type="stl")
    meta["final_faces"] = int(len(final_mesh.faces))
    meta["final_vertices"] = int(len(final_mesh.vertices))
    meta["final_path"] = str(final_path)
    print(f"\n[final] {final_path}  ({len(final_mesh.faces):,} faces, "
          f"{len(final_mesh.vertices):,} verts)")

    if args.render:
        body_final_deci = _decimate_to_stl(final_mesh, RENDER_DECIMATE_TARGET,
                                            scratch_dir, "final_body")
        _render_panel(
            body_stl=body_final_deci,
            remove_mask=np.zeros(_load_face_count(body_final_deci), dtype=bool),
            title=f"Step 7 (final) — upper shell ({len(final_mesh.faces):,} faces)",
            out_path=render_dir / "step7_final_shell.png",
            view_bounds=final_mesh.bounds.copy(),
            scratch_dir=scratch_dir,
        )

    # =====================================================================
    # Step 8 — open-boundary components + re-cut via longest rim
    # =====================================================================
    # NOTE: `final_mesh` is built via submesh+process — its boundary edges
    # already use deduplicated vertex indices, so connected-component
    # walks see real adjacency. Components > loops here because the
    # boundary graph has T-junctions (verts of degree ≥ 3 where two
    # surface sheets meet at a single point); using components instead
    # of single-path-walked loops is robust to those.
    boundary_edges, components = find_open_boundary_components(final_mesh)
    print(f"[boundaries] {len(boundary_edges):,} open edges, "
          f"{len(components)} components")
    boundaries_all_path = output_dir / f"{basename}_boundaries_all.stl"
    n_all = export_edges_as_tubes(
        final_mesh, boundary_edges, boundaries_all_path)
    print(f"[boundaries] {boundaries_all_path}  "
          f"({n_all:,} edges as tubes)")
    meta["boundaries_all_path"] = str(boundaries_all_path)
    meta["boundaries_all_edge_count"] = int(n_all)
    meta["boundaries_component_count"] = len(components)

    if components:
        longest_verts, longest_edges, longest_peri = components[0]
        boundaries_longest_path = (
            output_dir / f"{basename}_boundaries_longest.stl")
        n_long = export_edges_as_tubes(
            final_mesh, longest_edges, boundaries_longest_path)
        print(f"[boundaries] {boundaries_longest_path}  "
              f"(longest component: {len(longest_verts)} verts, "
              f"{n_long:,} edges, perimeter ≈ {longest_peri:.3f})")
        meta["boundaries_longest_path"] = str(boundaries_longest_path)
        meta["boundaries_longest_vertex_count"] = int(len(longest_verts))
        meta["boundaries_longest_edge_count"] = int(len(longest_edges))
        meta["boundaries_longest_perimeter"] = float(longest_peri)

    recut_mesh, recut_info = recut_with_longest_rim(body, final_mesh)
    recut_path = output_dir / f"{basename}_upper_recut.stl"
    recut_mesh.export(str(recut_path), file_type="stl")
    meta["recut_path"] = str(recut_path)
    meta["recut_faces"] = int(len(recut_mesh.faces))
    meta["recut_vertices"] = int(len(recut_mesh.vertices))
    meta["recut_info"] = recut_info
    print(f"[recut] {recut_path}  ({len(recut_mesh.faces):,} faces, "
          f"{len(recut_mesh.vertices):,} verts)")

    meta_path = output_dir / f"{basename}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"[meta] {meta_path}")

    # Clean up scratch dir if it's empty (render decimated STLs).
    if scratch_dir.exists() and not args.debug:
        try:
            for f in scratch_dir.iterdir():
                f.unlink()
            scratch_dir.rmdir()
        except OSError:
            pass

    return 0


def _load_face_count(stl_path: Path) -> int:
    m = trimesh.load(str(stl_path), file_type="stl",
                     force="mesh", process=False)
    return int(len(m.faces))


if __name__ == "__main__":
    sys.exit(main())
