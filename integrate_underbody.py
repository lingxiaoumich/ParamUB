"""Generate a parametric underbody for a previously extracted upper shell.

The shell extraction (``run_shell.py``) writes:
  * ``<base>_final.stl``       — full-car upper shell (no y=0 cut)
  * ``<base>_upper_recut.stl`` — re-cut version of the upper shell
                                  with internal holes plugged
  * ``<base>_meta.json``        — frame, wheel hubs, per-step counts

This script reads those, derives :class:`paramub.UnderbodySpec`
parameters from the shell + meta, runs ``build_underbody`` on the
full car (no half), and writes the resulting body + 4 wheels as STLs.

No trimming, no y=0 capping, no shell ↔ UB combination — those steps
are handled downstream now.

Outputs (in ``--output-dir``)
=============================

  <base>_shell.stl              upper shell from run_shell, scaled to
                                mm (canonical frame).
  <base>_underbody.stl          parametric UB (floor + wheelhouses),
                                aligned to shell frame, untrimmed.
  <base>_underbody_trimmed.stl  UB trimmed to the shell-rim cut polygon
                                (unless ``--no-trim``).
  <base>_cut_polygon.stl        cut polygon as a thin vertical wall
                                (debug; only when trim runs).
  <base>_wheel_front_left.stl   4 wheels, aligned to shell frame.
  <base>_wheel_front_right.stl
  <base>_wheel_rear_left.stl
  <base>_wheel_rear_right.stl
  <base>_integrate_meta.json    hints, anchors, spec, face counts.

Coordinate / unit handling
==========================

Shell canonical frame (set in :mod:`paramub.shell_extract`):
    +x = rearward, +y = lateral, +z = up; native units are whatever the
    input STL was in (typically meters at Hunyuan3D normalized scale,
    e.g. ~1.2 m long).

ParamUB underbody frame (set in :mod:`paramub.ub_assem`):
    +x = forward, +y = right, +z = up, mm units, body centered on x=0
    (front axle at +wheelbase/2).

We bridge the two by:
  1. SCALE = 2700 / wheelbase_shell  — pick a scale so the wheelbase is
     2700 mm, putting the resulting "real-size car" mesh in mm where
     ParamUB defaults (fillets etc.) are sensible.
  2. After ParamUB generates the underbody in its own frame, we apply
       (x, y, z) -> (-x + dx, y, z)
     to mirror X and align the wheel midpoints.  The X reflection
     inverts triangle winding; ``align_to_shell_frame`` flips face
     vertex order to keep outward normals consistent.

The optional remesh step uses pymeshlab, which needs a newer libstdc++
than the system one — invoke with the conda env's libs on LD_LIBRARY_PATH:

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python integrate_underbody.py

Or pass ``--no-remesh`` to skip remesh entirely.

Usage::

    python integrate_underbody.py
    python integrate_underbody.py --shell-meta outputs/shell/<name>_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_META = Path(
    "outputs/shell/alfa_romeo_giuliazhuliye_2025_image10_71415_shadowfill_meta.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/integrate")
TARGET_WHEELBASE_MM = 2700.0


# ---------------------------------------------------------------------------
# Shell-derived geometry hints
# ---------------------------------------------------------------------------

REAR_EDGE_BAND_MM = 50.0   # band from body_x_max used to measure rear-edge z
ROCKER_LINE_FRAC = 0.20    # fraction of body z-envelope considered "low" for the rocker


def extract_hints(shell_mesh: trimesh.Trimesh, meta: dict,
                  scale: float) -> dict:
    """Pull every geometry hint the underbody spec needs out of the shell
    mesh + meta. Returns everything in ParamUB-style mm."""
    verts = np.asarray(shell_mesh.vertices) * scale
    faces = np.asarray(shell_mesh.faces)
    centroids = verts[faces].mean(axis=1)

    wheels = meta["wheels_3d"]
    # Shell convention: +x = rear. Sort wheels by x: smaller x = front.
    sorted_w = sorted(wheels, key=lambda w: w["x"])
    front_w = sorted_w[0]
    rear_w = sorted_w[-1]
    front_x_shell = front_w["x"] * scale
    rear_x_shell = rear_w["x"] * scale
    wheelbase = abs(rear_x_shell - front_x_shell)
    midpoint_x_shell = 0.5 * (front_x_shell + rear_x_shell)

    # Use mean(|y|) — for full-car wheels are mirrored so mean(y) ≈ 0.
    track = 2.0 * float(np.mean([abs(w["y"]) for w in wheels])) * scale
    tire_od = 2.0 * np.mean([w["radius"] for w in wheels]) * scale
    tire_width = 2.0 * np.mean([w["axial_half"] for w in wheels]) * scale
    hub_z = np.mean([w["z"] for w in wheels]) * scale

    # Per-axle wheel center y (half-track), in mm. For a full car we
    # have 4 wheels (2 front, 2 rear); for half-car only the 2 left
    # wheels are listed. front_wheels = lowest-x pair, rear_wheels =
    # highest-x pair.
    n = len(sorted_w)
    front_wheels = sorted_w[:2] if n >= 4 else [sorted_w[0]]
    rear_wheels = sorted_w[-2:] if n >= 4 else [sorted_w[-1]]
    front_wheel_y_mm = float(np.mean(
        [abs(w["y"]) for w in front_wheels])) * scale
    rear_wheel_y_mm = float(np.mean(
        [abs(w["y"]) for w in rear_wheels])) * scale

    body_bounds = np.array([verts.min(axis=0), verts.max(axis=0)])
    x_min_body, x_max_body = body_bounds[0, 0], body_bounds[1, 0]
    # Overhangs measured from wheels to body ends (shell frame, +x = rear).
    front_overhang = front_x_shell - x_min_body
    rear_overhang = x_max_body - rear_x_shell

    # Ride height: the lowest shell point that ISN'T near a wheel cylinder.
    # We exclude xy bands around each wheel (radius * 1.2 in x, axial * 1.5
    # in y) and take the min z of the remaining face centroids.
    wheel_mask = np.zeros(len(centroids), dtype=bool)
    for w in wheels:
        wx, wy = w["x"] * scale, w["y"] * scale
        wr = w["radius"] * scale * 1.2
        wax = w["axial_half"] * scale * 1.5
        wheel_mask |= (np.abs(centroids[:, 0] - wx) < wr) & \
                      (np.abs(centroids[:, 1] - wy) < wax)
    non_wheel = centroids[~wheel_mask]
    ride_height = float(non_wheel[:, 2].min())

    # Wheelhouse top: max z in a cylinder ~1.5 r around each wheel hub.
    wh_top = {}
    for label, w in (("front", front_w), ("rear", rear_w)):
        wx, wy = w["x"] * scale, w["y"] * scale
        wr = w["radius"] * scale * 1.5
        near = centroids[(np.abs(centroids[:, 0] - wx) < wr) &
                         (np.abs(centroids[:, 1] - wy) < wr)]
        wh_top[label] = float(near[:, 2].max()) if len(near) else None

    # Rear-edge z: lowest shell vertex within REAR_EDGE_BAND_MM of the
    # very rear (x ≥ x_max_body − band). This is the actual bumper bottom
    # AT THE TAIL — the diffuser exit should match this, not the min over
    # the full last 15% of the body (which can include the much lower
    # underside well in front of the bumper tip).
    rear_edge_mask = verts[:, 0] > (x_max_body - REAR_EDGE_BAND_MM)
    rear_edge_min_z = float(verts[rear_edge_mask, 2].min()) if rear_edge_mask.any() else ride_height

    # Per-wheel outboard extent: max |y| of the SHELL'S WHEELHOUSE
    # OPENING for each wheel. Used to size the parametric wheelhouse so
    # its outboard edge reaches the shell's rocker line.
    wheel_outboard_y = {}
    for label, w in (("front", front_w), ("rear", rear_w)):
        wx = w["x"] * scale
        wr = w["radius"] * scale * 1.3
        # Take low-z shell vertices in a band around the wheel x; their
        # most-outboard y (max |y|) gives the rocker line at this x.
        z_low = ride_height + 0.30 * (body_bounds[1, 2] - body_bounds[0, 2])
        col_mask = (np.abs(verts[:, 0] - wx) < wr) & (verts[:, 2] < z_low)
        wheel_outboard_y[label] = float(np.abs(verts[col_mask, 1]).max()) \
            if col_mask.any() else None

    # Floor width: how wide the underbody footprint is at low z. We sample
    # faces in the bottom 25% of the body's z range and take the y extent.
    z_low = ride_height + 0.25 * (body_bounds[1, 2] - body_bounds[0, 2])
    low_y = centroids[centroids[:, 2] < z_low, 1]
    # left-only shell: y <= 0. The other side is at y = -low_y (by symmetry).
    if len(low_y):
        body_half_width = float(np.abs(low_y).max())
    else:
        body_half_width = float(np.abs(body_bounds[:, 1]).max())
    floor_width = 2.0 * body_half_width

    hints = dict(
        scale=scale,
        wheelbase_mm=wheelbase,
        front_overhang_mm=front_overhang,
        rear_overhang_mm=rear_overhang,
        track_mm=track,
        tire_od_mm=tire_od,
        tire_width_mm=tire_width,
        hub_z_mm=hub_z,
        ride_height_mm=ride_height,
        floor_width_mm=floor_width,
        body_bounds_mm=body_bounds.tolist(),
        midpoint_x_shell_mm=midpoint_x_shell,
        wheelhouse_top_mm=wh_top,
        rear_edge_min_z_mm=rear_edge_min_z,
        wheel_outboard_y_mm=wheel_outboard_y,
        front_x_shell_mm=front_x_shell,
        rear_x_shell_mm=rear_x_shell,
        front_wheel_y_mm=front_wheel_y_mm,
        rear_wheel_y_mm=rear_wheel_y_mm,
    )
    print("[hints]")
    for k, v in hints.items():
        print(f"  {k:24s} = {v}")
    return hints


def measure_shell_anchors(shell_mm: trimesh.Trimesh, ys: list[float],
                          y_band: float = 25.0,
                          rear_search_mm: float = 200.0,
                          front_search_mm: float = 200.0) -> dict:
    """Probe the shell at each y_target in ``ys`` and find the
    front-bumper-bottom and rear-bumper-bottom anchors used as the
    splitter leading edges and the diffuser trailing edges.

    Algorithm: in the lateral band ``|y - y_target| < y_band``, search
    the last ``rear_search_mm`` of x (resp. first ``front_search_mm``)
    for the vertex with the lowest z. That vertex's (x, z) is the
    anchor — i.e. the physical bumper-bottom *edge*. Reporting just
    ``min(z)`` in a narrow slab next to ``x_max`` is wrong on a body
    with a rounded bumper (the rear-most x sits at the bumper *top*
    where it rounds into the trunk, ~50–150 mm above the bumper
    bottom).

    The shell is symmetric in y, so we accept vertices at either +y or -y.

    Returns ``{y_target: {y_target, front_x_shell, front_z_shell,
    rear_x_shell, rear_z_shell}}`` — all in the shell frame (mm).
    """
    verts = np.asarray(shell_mm.vertices)
    out: dict[float, dict] = {}
    for y in ys:
        near = (np.abs(verts[:, 1] - y) < y_band) | \
               (np.abs(verts[:, 1] + y) < y_band)
        band = verts[near]
        if len(band) == 0:
            raise ValueError(
                f"no shell vertices within y_band={y_band} mm of Y={y}")
        x_min = float(band[:, 0].min())
        x_max = float(band[:, 0].max())
        front = band[band[:, 0] < x_min + front_search_mm]
        rear = band[band[:, 0] > x_max - rear_search_mm]
        i_fbot = int(np.argmin(front[:, 2]))
        i_rbot = int(np.argmin(rear[:, 2]))
        out[y] = {
            "y_target": float(y),
            "front_x_shell": float(front[i_fbot, 0]),
            "front_z_shell": float(front[i_fbot, 2]),
            "rear_x_shell": float(rear[i_rbot, 0]),
            "rear_z_shell": float(rear[i_rbot, 2]),
            "front_x_extent": x_min,
            "rear_x_extent": x_max,
        }
        a = out[y]
        print(f"[anchor] Y={y:>5.0f}  "
              f"front (x,z)=({a['front_x_shell']:>7.1f}, "
              f"{a['front_z_shell']:>6.1f}) "
              f"[bumper-tip x={x_min:.1f}]  "
              f"rear (x,z)=({a['rear_x_shell']:>7.1f}, "
              f"{a['rear_z_shell']:>6.1f}) "
              f"[bumper-tip x={x_max:.1f}]")
    return out


def lateral_clearance_overrides(hints: dict,
                                 extra_extend_mm: float = 0.0) -> dict:
    """Per-wheel lateral_clearance for the wheelhouse so its outboard
    edge reaches the shell's rocker line at that wheel (plus an optional
    ``extra_extend_mm`` margin so the arch overshoots the rocker — the
    boundary trim clips it back).

    The wheelhouse arch is symmetric in y about y_track with half-length
    = (tire_width + 2 * lateral_clearance) / 2 = tire_width/2 +
    lateral_clearance. So to make the outboard edge land at
    ``wheel_outboard_y + extra_extend_mm``, we need:

        lateral_clearance = (wheel_outboard_y + extra_extend_mm)
                            − track/2 − tire_width/2

    Inboard side extends symmetrically (the wheelhouse builder doesn't
    support per-side asymmetry) — typically harmless against the floor.
    """
    overrides: dict[tuple[str, str], float] = {}
    track_half = hints["track_mm"] / 2.0
    tire_half = hints["tire_width_mm"] / 2.0
    for axle_label in ("front", "rear"):
        outboard = hints["wheel_outboard_y_mm"].get(axle_label)
        if outboard is None:
            continue
        target_y = outboard + extra_extend_mm
        need = max(0.0, target_y - track_half - tire_half)
        # Add a small margin so the wheelhouse fully reaches the target.
        need += 5.0
        overrides[(axle_label, "left")] = need
        overrides[(axle_label, "right")] = need
        print(f"[wheelhouse] {axle_label}: rocker y={outboard:.0f}  "
              f"+extend {extra_extend_mm:.0f}  →  target y={target_y:.0f}  "
              f"→  lateral_clearance={need:.1f} mm  "
              f"(track/2={track_half:.0f}, tire_half={tire_half:.0f})")
    return overrides


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_spec(hints: dict, anchors: dict,
                lat_overrides: dict | None = None,
                *,
                y_intermediate: float = 700.0,
                y_intermediate_splitter: float | None = None,
                y_intermediate_diffuser: float | None = None,
                length_extend_mm: float = 0.0,
                width_extend_mm: float = 100.0,
                splitter_kick_offset_mm: float = 50.0,
                diffuser_kick_offset_mm: float = 50.0,
                splitter_front_angle_deg: float = +30.0,
                diffuser_rear_angle_deg: float = 5.0,
                handle_strength: float = 0.30):
    """Build an UnderbodySpec with multisection splitter + diffuser
    Bezier lofts pinned to the shell's measured front/rear edges.

    Sections per face:
        Y = 0                 (centerline)
        Y = y_intermediate    (≈ rocker line; measured anchor)
        Y = y_outboard        (= max(y_intermediate, body_half) + width_extend
                                — same shape as the intermediate section so
                                the surface stays constant outboard until
                                the boundary trim clips it back to the body)

    Splitter leading edge at (Y=0, Y=y_intermediate) is placed at the
    shell-measured front-edge bottom + ``length_extend_mm`` further
    forward. Same on the diffuser at the rear.

    Splitter kick = front_axle + ``splitter_kick_offset_mm`` (forward).
    Diffuser kick = rear_axle − ``diffuser_kick_offset_mm`` (rearward).
    """
    from paramub import (
        UnderbodySpec, WheelSpec, TireSpec, SpokeSpec,
        SplitterSection, DiffuserSection,
    )

    # ---- Tire / spoke (unchanged) -------------------------------------
    target_od = hints["tire_od_mm"]
    target_sw = hints["tire_width_mm"]
    rim_in = 18.0
    rim_r = rim_in * 25.4 / 2.0
    aspect = max(25.0, min(75.0,
                            (target_od / 2.0 - rim_r) / target_sw * 100.0))
    section_w = (target_od / 2.0 - rim_r) / (aspect / 100.0)
    tire = TireSpec(
        section_width_mm=section_w,
        aspect_ratio=aspect,
        rim_diameter_in=rim_in,
        tread_width_mm=max(60.0, 0.85 * target_sw),
        crown_radius_mm=max(0.85 * target_sw / 2 + 50.0, 0.6 * target_od),
        sidewall_bulge_mm=6.0,
        shoulder_radius_mm=20.0,
        rim_flange_mm=14.0,
    )
    spoke = SpokeSpec(
        wheel_width_mm=target_sw,
        rim_diameter_in=rim_in,
        num_spokes=5,
    )

    # ---- Bezier sections from shell anchors ---------------------------
    midpoint_x = hints["midpoint_x_shell_mm"]
    front_axle_x = +hints["wheelbase_mm"] / 2.0
    rear_axle_x = -hints["wheelbase_mm"] / 2.0
    splitter_kick_x = front_axle_x + splitter_kick_offset_mm
    diffuser_kick_x = rear_axle_x - diffuser_kick_offset_mm

    def shell_x_to_paramub_x(x_shell: float) -> float:
        return midpoint_x - x_shell

    if y_intermediate_splitter is None:
        y_intermediate_splitter = y_intermediate
    if y_intermediate_diffuser is None:
        y_intermediate_diffuser = y_intermediate

    body_half = hints["floor_width_mm"] / 2.0
    # Single y_outboard for both lofts. floor_width is symmetric and
    # must accommodate the maximum of the two intermediate Y values.
    y_outboard = max(y_intermediate_splitter,
                      y_intermediate_diffuser,
                      body_half) + width_extend_mm
    floor_width = 2.0 * y_outboard
    print(f"[spec] y_intermediate_splitter={y_intermediate_splitter:.0f}  "
          f"y_intermediate_diffuser={y_intermediate_diffuser:.0f}  "
          f"body_half={body_half:.0f}  width_extend={width_extend_mm:.0f}  "
          f"→  y_outboard={y_outboard:.0f}  floor_width={floor_width:.0f}")

    ride_h = hints["ride_height_mm"]

    def _safe_end_strength(end_x: float, end_z: float, kick_x: float,
                            angle_deg: float, request: float,
                            label: str, y_mm: float) -> float:
        """Cap end_strength so the cubic Bezier's P2 control point stays
        above ride_h (the flat floor). With t3 having a positive Z
        component (curve heading away from the floor at the endpoint),
        P2 = P3 − end_strength · chord · t3 drops in Z. If P2 drops
        below ride_h the curve typically dips below the floor between
        kick and endpoint."""
        import math
        chord = math.hypot(end_x - kick_x, end_z - ride_h)
        sin_a = math.sin(math.radians(angle_deg))
        if sin_a <= 1e-6:
            return request                       # tangent has no Z component
        margin = 5.0
        # P2.z = P3.z − end_strength · chord · sin_a  ≥  ride_h + margin
        max_safe = max(0.05, (end_z - ride_h - margin) / (chord * sin_a))
        if request > max_safe:
            print(f"[strength] {label} Y={y_mm:.0f}: end_strength "
                  f"{request:.3f} → {max_safe:.3f} "
                  f"(request would dip P2 below floor)")
        return min(request, max_safe)

    def splitter_at(y_anchor: float, y_mm: float) -> SplitterSection:
        a = anchors[y_anchor]
        front_x = shell_x_to_paramub_x(a["front_x_shell"]) + length_extend_mm
        end_str = _safe_end_strength(
            front_x, a["front_z_shell"], splitter_kick_x,
            splitter_front_angle_deg, handle_strength,
            "splitter", y_mm)
        return SplitterSection(
            y_mm=y_mm,
            kick_x_mm=splitter_kick_x,
            front_x_mm=front_x,
            front_z_mm=a["front_z_shell"],
            front_angle_deg=splitter_front_angle_deg,
            start_strength=handle_strength,
            end_strength=end_str,
        )

    def diffuser_at(y_anchor: float, y_mm: float) -> DiffuserSection:
        a = anchors[y_anchor]
        rear_x = shell_x_to_paramub_x(a["rear_x_shell"]) - length_extend_mm
        # _safe_end_strength uses |end_x − kick_x| via hypot, so it works
        # for either direction (diffuser kick > rear, splitter kick < front).
        end_str = _safe_end_strength(
            rear_x, a["rear_z_shell"], diffuser_kick_x,
            diffuser_rear_angle_deg, handle_strength,
            "diffuser", y_mm)
        return DiffuserSection(
            y_mm=y_mm,
            kick_x_mm=diffuser_kick_x,
            rear_x_mm=rear_x,
            rear_z_mm=a["rear_z_shell"],
            rear_angle_deg=diffuser_rear_angle_deg,
            start_strength=handle_strength,
            end_strength=end_str,
        )

    splitter_sections = [
        splitter_at(0.0, 0.0),
        splitter_at(y_intermediate_splitter, y_intermediate_splitter),
        splitter_at(y_intermediate_splitter, y_outboard),   # outboard = copy of intermediate
    ]
    diffuser_sections = [
        diffuser_at(0.0, 0.0),
        diffuser_at(y_intermediate_diffuser, y_intermediate_diffuser),
        diffuser_at(y_intermediate_diffuser, y_outboard),
    ]

    spec = UnderbodySpec(
        wheelbase_mm=hints["wheelbase_mm"],
        front_overhang_mm=hints["front_overhang_mm"],
        rear_overhang_mm=hints["rear_overhang_mm"],
        track_front_mm=hints["track_mm"],
        track_rear_mm=hints["track_mm"],
        ride_height_mm=hints["ride_height_mm"],
        floor_width_mm=floor_width,
        splitter_sections=splitter_sections,
        diffuser_sections=diffuser_sections,
        wheel_house_axial_clearance_mm=20.0,
        wheel_house_lateral_clearance_mm=25.0,
        front_steering_clearance_mm=0.0,
        rear_steering_clearance_mm=0.0,
        camber_front_deg=0.0,
        camber_rear_deg=0.0,
        toe_front_deg=0.0,
        toe_rear_deg=0.0,
        wheel=WheelSpec(tire=tire, spoke=spoke),
        lateral_clearance_overrides_mm=lat_overrides,
    )
    return spec


# ---------------------------------------------------------------------------
# CadQuery -> trimesh; align to shell frame; densify
# ---------------------------------------------------------------------------

def align_to_shell_frame(mesh: trimesh.Trimesh,
                          midpoint_x_shell: float) -> trimesh.Trimesh:
    """ParamUB frame (+x forward, +y right, +z up) -> shell frame
    (+x rear, +y lateral, +z up).

    Reflect x; shift so the wheel-midpoint lands at the shell's. The
    reflection inverts winding, so we also reverse triangle vertex order
    to keep outward normals correct.
    """
    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    v[:, 0] = -v[:, 0] + midpoint_x_shell
    f = f[:, [0, 2, 1]]  # flip winding
    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return out


def subdivide_to_edge(mesh: trimesh.Trimesh, max_edge: float) -> trimesh.Trimesh:
    """CadQuery's STL exporter tessellates flat faces (like the floor) with
    a handful of giant triangles — the central floor between the
    wheelhouses can be a single triangle covering > 1 m². This breaks the
    y = 0 slice (triangles that straddle the plane are dropped wholesale
    instead of being split) and makes the perimeter-trim's centroid test
    meaningless. Subdivide repeatedly until every edge is ≤ max_edge."""
    out = mesh
    for _ in range(20):
        v, f = trimesh.remesh.subdivide_to_size(
            out.vertices, out.faces, max_edge=max_edge, max_iter=10)
        new = trimesh.Trimesh(vertices=v, faces=f, process=False)
        if len(new.faces) == len(out.faces):
            return new
        out = new
    return out


def cq_obj_to_trimesh_via_stl(cq_obj, stl_tmp: Path,
                               tolerance: float = 0.1,
                               angular_tolerance: float = 0.1
                               ) -> trimesh.Trimesh:
    """Export a single CadQuery object (Workplane / Shape / Assembly child
    ``.obj``) to STL and load as trimesh. Wraps it in a one-child
    Assembly for the cq.exporters API."""
    import cadquery as cq
    from cadquery import exporters
    sub = cq.Assembly()
    sub.add(cq_obj, name="part")
    compound = sub.toCompound()
    exporters.export(compound, str(stl_tmp),
                      exporters.ExportTypes.STL,
                      tolerance=tolerance,
                      angularTolerance=angular_tolerance)
    return trimesh.load(str(stl_tmp), force="mesh", process=False)


# ---------------------------------------------------------------------------
# Shell rim → 2-D cut polygon (silhouette − shrink) for trimming the UB
# ---------------------------------------------------------------------------

def build_shell_cut_polygon(shell_mm: trimesh.Trimesh,
                             shrink_mm: float = 50.0,
                             simplify_mm: float = 10.0,
                             concave_ratio: float = 0.05):
    """Z-project the upper-shell's longest open-boundary loop onto the
    XY plane, build a concave-hull polygon, shrink inward by
    ``shrink_mm``, then smooth.

    The shell mesh must already be in mm (caller-side scaling).

    Returns a shapely Polygon (in mm) or ``None`` if the shrink left
    nothing usable.
    """
    import shapely
    from shapely.geometry import MultiPoint, Polygon
    from paramub.shell_recut import find_open_boundary_components

    # final_path was loaded with process=False so its boundary graph is
    # noise (every face has its own vertices). Re-process for dedup.
    sh = trimesh.Trimesh(
        vertices=np.asarray(shell_mm.vertices),
        faces=np.asarray(shell_mm.faces),
        process=True)
    boundary, components = find_open_boundary_components(sh)
    if not components:
        return None
    rim_idx = components[0][0]
    rim_xy = np.asarray(sh.vertices)[rim_idx, :2]
    print(f"[cut-poly] rim {len(rim_idx):,} verts -> "
          f"XY range x∈[{rim_xy[:,0].min():.0f},{rim_xy[:,0].max():.0f}] "
          f"y∈[{rim_xy[:,1].min():.0f},{rim_xy[:,1].max():.0f}]")

    mp = MultiPoint(rim_xy)
    hull = shapely.concave_hull(mp, ratio=concave_ratio)
    if not isinstance(hull, Polygon) or hull.is_empty:
        print(f"[cut-poly] concave hull came back empty / not a polygon: "
              f"{hull.geom_type}")
        return None
    print(f"[cut-poly] concave hull (ratio={concave_ratio}): "
          f"area={hull.area:.0f} mm^2, "
          f"exterior length={hull.exterior.length:.0f} mm")

    shrunk = hull.buffer(-shrink_mm)
    if shrunk.is_empty:
        print(f"[cut-poly] shrink({-shrink_mm}) emptied the polygon")
        return None
    if shrunk.geom_type == "MultiPolygon":
        # Keep the largest piece — a thin neck might split into chunks
        shrunk = max(shrunk.geoms, key=lambda g: g.area)
    print(f"[cut-poly] shrunk by {shrink_mm} mm -> area={shrunk.area:.0f} mm^2, "
          f"exterior length={shrunk.exterior.length:.0f} mm")

    smoothed = shrunk.simplify(simplify_mm, preserve_topology=True)
    if smoothed.is_empty:
        smoothed = shrunk
    print(f"[cut-poly] simplify({simplify_mm}) -> "
          f"exterior coords {len(shrunk.exterior.coords)} → "
          f"{len(smoothed.exterior.coords)}")
    return smoothed


def remesh_underbody_isotropic(mesh: trimesh.Trimesh,
                                 target_edge_mm: float,
                                 iterations: int = 6,
                                 scratch_dir: Path | None = None
                                 ) -> trimesh.Trimesh:
    """Re-mesh ``mesh`` with PyMeshLab's isotropic-explicit remesher to a
    uniform ``target_edge_mm`` edge length. Returns a new trimesh.

    Bounded memory + fast — implemented in C++. Used as a last step to
    bring the parametric underbody's mesh density up to the upper
    shell's, so downstream booleans / projections see comparable
    resolution on both meshes.
    """
    import pymeshlab
    scratch_dir = scratch_dir or Path("/tmp")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    src = scratch_dir / "remesh_in.stl"
    dst = scratch_dir / "remesh_out.stl"
    mesh.export(str(src), file_type="stl")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    f0 = ms.current_mesh().face_number()
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.AbsoluteValue(target_edge_mm),
        iterations=iterations,
        adaptive=False,
        checksurfdist=False,
    )
    f1 = ms.current_mesh().face_number()
    ms.save_current_mesh(str(dst))
    out = trimesh.load(str(dst), force="mesh", process=False)
    print(f"[remesh] pymeshlab isotropic (targetlen={target_edge_mm:.2f} mm, "
          f"iters={iterations}): {f0:,} -> {f1:,} faces")
    return out


def cut_underbody_cq_with_polygon(underbody_cq, polygon,
                                    midpoint_x_shell: float,
                                    z_pad: float = 100.0):
    """BREP-level cut of the CadQuery underbody Compound by the vertical
    extrusion of ``polygon`` (a shapely Polygon in SHELL-frame mm).

    Each face of ``underbody_cq`` is intersected with a prism whose XY
    cross-section is ``polygon`` (transformed to ParamUB frame: X is
    reflected about ``midpoint_x_shell``). The intersection preserves
    analytical curves at the cut, so subsequent STL tessellation gives
    clean triangulated edges along the polygon's perimeter rather than
    the staircased boundary you get from filtering pre-tessellated
    triangles in 2-D.

    Returns a CadQuery Compound of trimmed faces (or an empty compound
    if every face fell outside the prism).
    """
    import cadquery as cq
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

    # Polygon coords (shell frame, mm); drop closing duplicate if present
    coords_shell = list(polygon.exterior.coords)
    if len(coords_shell) >= 2 and coords_shell[0] == coords_shell[-1]:
        coords_shell = coords_shell[:-1]
    # Transform: x_paramub = midpoint - x_shell  (mirror across midpoint)
    # The mirror reverses winding so polyline order goes CW -> CCW; CQ
    # autodetects orientation when extruding closed wires.
    coords_pub = [(midpoint_x_shell - x, y) for x, y in coords_shell]

    bbox = underbody_cq.BoundingBox()
    z_min = bbox.zmin - z_pad
    z_top = bbox.zmax + z_pad
    height = z_top - z_min
    print(f"[brep-cut] polygon {len(coords_pub)} corners; "
          f"prism z=[{z_min:.0f}, {z_top:.0f}]  height={height:.0f} mm")

    prism_wp = (cq.Workplane("XY", origin=(0, 0, z_min))
                .polyline(coords_pub)
                .close()
                .extrude(height))
    prism_solid = prism_wp.val()

    in_faces = list(underbody_cq.Faces())
    trimmed: list = []
    for face in in_faces:
        op = BRepAlgoAPI_Common(face.wrapped, prism_solid.wrapped)
        op.Build()
        if not op.IsDone():
            continue
        res = op.Shape()
        if res.IsNull():
            continue
        trimmed.append(cq.Shape(res))

    if trimmed:
        out = cq.Compound.makeCompound(trimmed)
    else:
        out = cq.Compound.makeCompound([])
    out_faces = list(out.Faces())
    print(f"[brep-cut] {len(in_faces)} input faces -> "
          f"{len(out_faces)} trimmed faces in compound")
    return out


def trim_underbody_xy(underbody: trimesh.Trimesh, polygon) -> trimesh.Trimesh:
    """Keep UB faces whose three vertices' XY all lie inside ``polygon``.

    Equivalent to extruding ``polygon`` vertically through the UB and
    cutting — every face that survives the test is geometrically
    inside the prism for all Z. Strict (all-three-verts-inside) so no
    triangle straddles the boundary."""
    from shapely import contains_xy
    verts_xy = np.asarray(underbody.vertices[:, :2])
    inside = contains_xy(polygon, verts_xy[:, 0], verts_xy[:, 1])
    keep = inside[underbody.faces].all(axis=1)
    keep_idx = np.flatnonzero(keep)
    print(f"[trim] inside-polygon test: keep {keep_idx.size:,} / "
          f"{len(underbody.faces):,} faces "
          f"(dropped {len(underbody.faces) - keep_idx.size:,})")
    if keep_idx.size == 0:
        return trimesh.Trimesh(vertices=underbody.vertices[:0],
                                faces=np.zeros((0, 3), dtype=np.int64),
                                process=False)
    out = underbody.submesh([keep_idx], append=True)
    if isinstance(out, list):
        out = out[0]
    return out


def export_polygon_as_wall_stl(polygon, out_path: Path,
                                 z_base: float = 0.0,
                                 z_top: float = 5.0) -> None:
    """Render the cut polygon as a thin vertical wall STL for debug
    viewing alongside the body."""
    if not hasattr(polygon, "exterior"):
        return
    coords = np.asarray(polygon.exterior.coords)
    if len(coords) >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    n = len(coords)
    if n < 3:
        return
    bot = np.column_stack([coords[:, 0], coords[:, 1],
                            np.full(n, z_base, dtype=np.float64)])
    top = np.column_stack([coords[:, 0], coords[:, 1],
                            np.full(n, z_top, dtype=np.float64)])
    verts = np.vstack([bot, top])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    mesh = trimesh.Trimesh(vertices=verts,
                            faces=np.array(faces, dtype=np.int64),
                            process=False)
    mesh.export(str(out_path), file_type="stl")
    print(f"[cut-poly] wall STL -> {out_path}  "
          f"({n} pts, z∈[{z_base:.0f},{z_top:.0f}])")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a parametric underbody for an extracted shell.")
    p.add_argument("--shell-meta", type=Path, default=DEFAULT_META,
                   help=f"Shell metadata JSON (default {DEFAULT_META}).")
    p.add_argument("--output-dir", "-o", type=Path,
                   default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory (default {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--cut-shrink-mm", type=float, default=100.0,
                   help="Inward shrink applied to the shell-rim polygon "
                        "before cutting the UB. Larger = bigger gap to "
                        "the shell.")
    p.add_argument("--cut-simplify-mm", type=float, default=10.0,
                   help="Douglas-Peucker tolerance (mm) used to smooth "
                        "the shrunk polygon.")
    p.add_argument("--cut-concave-ratio", type=float, default=0.05,
                   help="shapely concave_hull ratio. Larger = closer to "
                        "convex; smaller = follows local indentations.")
    p.add_argument("--no-trim", action="store_true",
                   help="Skip the shell-rim trim. Only the untrimmed "
                        "underbody STL is written.")
    p.add_argument("--cut-stl-tolerance-mm", type=float, default=0.05,
                   help="Linear STL chord tolerance for the trimmed "
                        "underbody. Finer than the untrimmed export "
                        "(0.1) so the new cut boundary tessellates "
                        "smoothly.")
    p.add_argument("--cut-stl-angular-tolerance-rad", type=float,
                   default=0.05,
                   help="Angular STL chord tolerance for the trimmed "
                        "underbody.")
    p.add_argument("--remesh-target-mm", type=float, default=None,
                   help="Target edge length (mm) for the pymeshlab "
                        "isotropic remesh of the trimmed underbody. "
                        "Default = median edge length of the upper "
                        "shell.")
    p.add_argument("--remesh-iterations", type=int, default=6,
                   help="Iterations for pymeshlab "
                        "meshing_isotropic_explicit_remeshing.")
    p.add_argument("--no-remesh", action="store_true",
                   help="Skip the remesh-to-shell-density pass.")
    return p.parse_args()


WHEEL_NAME_MAP = {
    "wheel_fl": "front_left", "wheel_fr": "front_right",
    "wheel_rl": "rear_left",  "wheel_rr": "rear_right",
}


def main():
    args = parse_args()
    meta = json.loads(args.shell_meta.read_text())
    shell_stl_path = Path(meta["final_path"])
    if not shell_stl_path.is_absolute():
        shell_stl_path = args.shell_meta.parent / shell_stl_path.name
    print(f"[load] shell STL = {shell_stl_path}")
    shell = trimesh.load(str(shell_stl_path), force="mesh", process=False)
    print(f"  shell faces={len(shell.faces):,}  verts={len(shell.vertices):,}")

    # SCALE so wheelbase = 2700 mm. Pair the two most-extreme wheels by x
    # (smallest x = front-most, largest x = rear-most) so the wheelbase
    # measurement is robust when meta lists 4 corners.
    wheels = sorted(meta["wheels_3d"], key=lambda w: w["x"])
    wb_shell = abs(wheels[-1]["x"] - wheels[0]["x"])
    scale = TARGET_WHEELBASE_MM / wb_shell
    print(f"[scale] shell wheelbase = {wb_shell:.4f}  →  scale = {scale:.2f}  "
          f"(target wheelbase = {TARGET_WHEELBASE_MM:.0f} mm)")

    shell_mm = trimesh.Trimesh(
        vertices=np.asarray(shell.vertices) * scale,
        faces=np.asarray(shell.faces),
        process=False,
    )

    hints = extract_hints(shell, meta, scale)
    # Splitter loft Y sections: centerline + front wheel center.
    # Diffuser loft Y sections: centerline + rear wheel center.
    # The outboard section copies the wheel-center shape (see build_spec).
    y_int_splitter = hints["front_wheel_y_mm"]
    y_int_diffuser = hints["rear_wheel_y_mm"]
    section_ys = sorted({0.0, y_int_splitter, y_int_diffuser})
    anchors = measure_shell_anchors(shell_mm, ys=section_ys)
    lat_overrides = lateral_clearance_overrides(hints, extra_extend_mm=100.0)
    spec = build_spec(hints, anchors, lat_overrides=lat_overrides,
                       y_intermediate_splitter=y_int_splitter,
                       y_intermediate_diffuser=y_int_diffuser)

    midpoint_x = hints["midpoint_x_shell_mm"]
    if spec.splitter_sections:
        print("[splitter endpoints]")
        for s in spec.splitter_sections:
            x_shell = midpoint_x - s.front_x_mm
            print(f"  Y={s.y_mm:>5.0f}  paramub front=({s.front_x_mm:>7.1f}, "
                  f"{s.front_z_mm:>6.1f})  → shell=({x_shell:>7.1f}, "
                  f"{s.front_z_mm:>6.1f})")
    if spec.diffuser_sections:
        print("[diffuser endpoints]")
        for s in spec.diffuser_sections:
            x_shell = midpoint_x - s.rear_x_mm
            print(f"  Y={s.y_mm:>5.0f}  paramub rear=({s.rear_x_mm:>7.1f}, "
                  f"{s.rear_z_mm:>6.1f})  → shell=({x_shell:>7.1f}, "
                  f"{s.rear_z_mm:>6.1f})")

    print("\n[paramub] build_underbody (full car) ...")
    from paramub.ub_assem import build_underbody
    asy, layout = build_underbody(spec)
    print(f"[paramub] layout = {layout}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.output_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    # Split assembly: body (floor + wheelhouses) gets subdivide+align;
    # wheels are rigid bodies that only need the align transform.
    underbody_cq = None
    wheels_cq: dict[str, object] = {}
    for child in asy.children:
        if child.name == "underbody":
            underbody_cq = child.obj
        else:
            wheels_cq[child.name] = child.obj
    if underbody_cq is None:
        raise RuntimeError("Assembly has no child named 'underbody'")

    # ---- body: align only (no slice, no trim, no subdivide) ------------
    # subdivide_to_edge was only needed for the boundary trim, which has
    # been removed.
    underbody_raw = cq_obj_to_trimesh_via_stl(underbody_cq, scratch / "body_paramub.stl")
    print(f"[underbody raw] faces={len(underbody_raw.faces):,}")
    underbody_aligned = align_to_shell_frame(underbody_raw, midpoint_x)
    print(f"[underbody aligned] faces={len(underbody_aligned.faces):,}  "
          f"y range=({underbody_aligned.bounds[0,1]:.1f}, "
          f"{underbody_aligned.bounds[1,1]:.1f})")

    # ---- wheels: align only --------------------------------------------
    wheel_meshes: dict[str, trimesh.Trimesh] = {}
    for name, cq_obj in wheels_cq.items():
        w_raw = cq_obj_to_trimesh_via_stl(
            cq_obj, scratch / f"{name}_paramub.stl")
        wheel_meshes[name] = align_to_shell_frame(w_raw, midpoint_x)
        print(f"[wheel] {name:<10s} faces={len(wheel_meshes[name].faces):,}")

    # ---- outputs --------------------------------------------------------
    base = args.shell_meta.stem.replace("_meta", "")
    out_shell = args.output_dir / f"{base}_shell.stl"
    shell_mm.export(str(out_shell), file_type="stl")
    print(f"[out] {out_shell}  ({len(shell_mm.faces):,} faces, "
          f"scaled to mm)")

    out_underbody = args.output_dir / f"{base}_underbody.stl"
    underbody_aligned.export(str(out_underbody), file_type="stl")
    print(f"[out] {out_underbody}  ({len(underbody_aligned.faces):,} faces)")

    wheel_out_paths: dict[str, str] = {}
    for name, mesh in wheel_meshes.items():
        short = WHEEL_NAME_MAP.get(name, name)
        out_wheel = args.output_dir / f"{base}_wheel_{short}.stl"
        mesh.export(str(out_wheel), file_type="stl")
        wheel_out_paths[name] = str(out_wheel)
        print(f"[out] {out_wheel}  ({len(mesh.faces):,} faces)")

    # ---- shell-rim trim ------------------------------------------------
    cut_meta: dict = {}
    out_underbody_trimmed = None
    if not args.no_trim:
        print(f"\n[cut-poly] building shell-rim cut polygon "
              f"(shrink={args.cut_shrink_mm} mm, "
              f"simplify={args.cut_simplify_mm} mm, "
              f"concave_ratio={args.cut_concave_ratio})")
        polygon = build_shell_cut_polygon(
            shell_mm,
            shrink_mm=args.cut_shrink_mm,
            simplify_mm=args.cut_simplify_mm,
            concave_ratio=args.cut_concave_ratio)
        if polygon is None:
            print("[cut-poly] no polygon produced; skipping trim")
        else:
            wall_path = args.output_dir / f"{base}_cut_polygon.stl"
            z_top = float(underbody_aligned.bounds[1, 2])
            export_polygon_as_wall_stl(
                polygon, wall_path,
                z_base=0.0, z_top=z_top)

            # BREP cut: intersect the CadQuery Compound (analytical
            # geometry) with the polygon prism, THEN tessellate. The
            # cut boundary becomes a real curve on each face, so STL
            # triangulation along the cut is smooth instead of stepping
            # along the floor's coarse pre-existing triangles.
            cut_cq = cut_underbody_cq_with_polygon(
                underbody_cq, polygon, midpoint_x)
            cut_stl = scratch / "underbody_cut_paramub.stl"
            cut_tol = args.cut_stl_tolerance_mm
            cut_ang = args.cut_stl_angular_tolerance_rad
            print(f"[brep-cut] tessellating trimmed compound "
                  f"(linear_tol={cut_tol} mm, "
                  f"angular_tol={cut_ang} rad)")
            underbody_trimmed_raw = cq_obj_to_trimesh_via_stl(
                cut_cq, cut_stl,
                tolerance=cut_tol,
                angular_tolerance=cut_ang)
            underbody_trimmed = align_to_shell_frame(
                underbody_trimmed_raw, midpoint_x)
            n_pre_remesh = int(len(underbody_trimmed.faces))

            # Remesh to roughly match the upper-shell's edge density so
            # downstream booleans / projections see a comparable resolution
            # on both meshes. Uses pymeshlab's isotropic-explicit remesher
            # (C++; bounded memory). Skipped on --no-remesh.
            remesh_target = None
            if not args.no_remesh:
                if args.remesh_target_mm is not None:
                    remesh_target = float(args.remesh_target_mm)
                else:
                    shell_edge_lengths = shell_mm.edges_unique_length
                    remesh_target = float(np.median(shell_edge_lengths))
                print(f"[remesh] target edge = {remesh_target:.2f} mm "
                      f"(shell median edge = "
                      f"{float(np.median(shell_mm.edges_unique_length)):.2f} mm)")
                try:
                    underbody_trimmed = remesh_underbody_isotropic(
                        underbody_trimmed,
                        target_edge_mm=remesh_target,
                        iterations=args.remesh_iterations,
                        scratch_dir=scratch)
                except ImportError as exc:
                    print(f"[remesh] pymeshlab not importable ({exc}); "
                          f"pass --no-remesh or set LD_LIBRARY_PATH="
                          f"$CONDA_PREFIX/lib")
                    raise

            out_underbody_trimmed = (
                args.output_dir / f"{base}_underbody_trimmed.stl")
            underbody_trimmed.export(str(out_underbody_trimmed),
                                      file_type="stl")
            print(f"[out] {out_underbody_trimmed}  "
                  f"({len(underbody_trimmed.faces):,} faces)")
            cut_meta = {
                "cut_shrink_mm": args.cut_shrink_mm,
                "cut_simplify_mm": args.cut_simplify_mm,
                "cut_concave_ratio": args.cut_concave_ratio,
                "cut_stl_tolerance_mm": cut_tol,
                "cut_stl_angular_tolerance_rad": cut_ang,
                "cut_polygon_area_mm2": float(polygon.area),
                "cut_polygon_exterior_length_mm": float(polygon.exterior.length),
                "underbody_trimmed_faces_pre_remesh": n_pre_remesh,
                "underbody_trimmed_faces": int(len(underbody_trimmed.faces)),
                "remesh_target_mm": remesh_target,
                "out_underbody_trimmed_stl": str(out_underbody_trimmed),
                "out_cut_polygon_wall_stl": str(wall_path),
            }

    # ---- meta dump ------------------------------------------------------
    debug_meta = {
        "hints": hints,
        "shell_anchors": anchors,
        "spec": _spec_to_jsonable(spec),
        "layout": layout,
        "underunderbody_raw_faces": int(len(underbody_raw.faces)),
        "underunderbody_aligned_faces": int(len(underbody_aligned.faces)),
        "wheels": {name: {"faces": int(len(m.faces))}
                    for name, m in wheel_meshes.items()},
        "out_shell_stl": str(out_shell),
        "out_underbody_stl": str(out_underbody),
        "out_wheel_stls": wheel_out_paths,
        **cut_meta,
    }
    (args.output_dir / f"{base}_integrate_meta.json").write_text(
        json.dumps(debug_meta, indent=2, default=float))
    return 0


def _spec_to_jsonable(spec):
    from dataclasses import is_dataclass, asdict as _asdict
    if is_dataclass(spec):
        return {k: _spec_to_jsonable(v) for k, v in _asdict(spec).items()}
    if isinstance(spec, (list, tuple)):
        return [_spec_to_jsonable(v) for v in spec]
    if isinstance(spec, dict):
        # JSON keys must be str; stringify any non-str keys (e.g. tuples).
        return {(k if isinstance(k, (str, int, float, bool, type(None)))
                  else str(k)): _spec_to_jsonable(v)
                for k, v in spec.items()}
    return spec


if __name__ == "__main__":
    sys.exit(main())
