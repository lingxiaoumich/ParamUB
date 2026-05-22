"""Integrate a parametric ParamUB underbody with an extracted upper shell.

The shell pipeline (``run_shell.py``) produces:
  * a single-component upper-body shell STL
  * a metadata JSON with the canonical frame, wheel hubs, and per-step
    face counts

This driver derives a matching parametric underbody from those hints,
generates it via :mod:`paramub.ub_assem`, transforms it into the shell's
canonical frame, takes the left half (y <= 0), trims its perimeter to
the shell footprint, and writes a combined STL plus a 10-panel debug
render.

Pipeline (top-level ``main()``)
===============================

1. Load shell STL + metadata; pick ``SCALE = 2700 mm / wheelbase_shell``
   so the model sits in ParamUB-style millimetres.
2. :func:`extract_hints` — wheelbase, overhangs, track, tire OD,
   ride height, wheelhouse-top z, rocker-line y, etc., all in mm.
3. :func:`measure_shell_anchors` — probe the shell at Y=0 and Y=700
   (lateral) and record the front-most and rear-most x extents plus
   the lowest z within 50 mm of each extremity. These become the
   splitter leading-edge and diffuser trailing-edge endpoints.
4. :func:`lateral_clearance_overrides` — per-wheel
   ``wheel_house_lateral_clearance`` so the arches reach (and slightly
   overshoot) the shell's rocker line.
5. :func:`build_spec` — assemble an :class:`paramub.UnderbodySpec`
   with a 3-section multisection splitter + diffuser pinned to the
   shell anchors. Splitter kick is at ``front_axle + 50 mm``; diffuser
   kick is at ``rear_axle - 50 mm``. Each section's ``end_strength``
   is capped per geometry so the Bezier never dips below the floor
   (see :func:`_safe_end_strength` inside ``build_spec``).
6. :func:`paramub.ub_assem.build_underbody` with ``half_only=True``.
7. Export ParamUB STL, load as trimesh, :func:`subdivide_to_edge` at
   25 mm so the boundary trim has small triangles to work with.
8. :func:`align_to_shell_frame` — mirror X (ParamUB +X_forward becomes
   shell +X_rearward) and shift so wheel midpoints align.
9. :func:`keep_left_half` — slice at y=0 (the half_only-built UB still
   covers the full Y span until this slice).
10. Boundary extraction + trim:
      :func:`extract_shell_boundary_loops_3d` returns the shell's open
      edge loops in 3D for visualisation;
      :func:`export_loops_as_curtain_stl` writes them as a 10 mm-tall
      curtain STL (the ``*_boundary_3d.stl`` artefact);
      :func:`extract_outer_boundary_polygon` computes a 2D concave hull
      of the same open-edge endpoints (mirrored across y=0, clipped to
      y<=0) — this is the polygon used by the actual trim;
      :func:`trim_to_shell_boundary` drops UB faces whose any vertex
      falls outside the polygon (all-vertices test — stricter than a
      centroid test, no large faces straddling the boundary).
11. Write the trimmed UB STL, combined (shell + UB) STL, debug PNGs,
    and a JSON dump of the spec + hints + anchors.

Coordinate / unit handling
==========================

Shell canonical frame (set in :mod:`paramub.shell_extract`):
    +x = rearward, +y = lateral, +z = up; left half is y <= 0; native
    units are whatever the input STL was in (typically meters at
    Hunyuan3D normalized scale, e.g. ~1.2 m long).

ParamUB underbody frame (set in :mod:`paramub.ub_assem`):
    +x = forward, +y = right, +z = up, mm units, body centered on x=0
    (front axle at +wheelbase/2).

We bridge the two by:
  1. SCALE = 2700 / wheelbase_shell  — pick a scale so the wheelbase is
     2700 mm, putting the resulting "real-size car" mesh in mm where
     ParamUB defaults (fillets etc.) are sensible.
  2. After ParamUB generates the underbody in its own frame, we apply
       (x, y, z) -> (-x + dx, y, z)
     i.e. reflect x (ParamUB +x_forward becomes shell +x_rear) and
     translate by dx = midpoint_x_shell so the wheels line up. The x
     reflection inverts triangle winding; we flip face vertex order to
     keep outward normals consistent.
  3. Slice at y = 0 to keep the left half.

Splitter / diffuser geometry tunables (build_spec keyword args)
================================================================

  y_intermediate              700.0 mm   — intermediate section Y.
  length_extend_mm            100.0 mm   — splitter front_x / diffuser
                                            rear_x extended past the
                                            measured shell edge so the
                                            boundary trim has material to
                                            clip against.
  width_extend_mm             100.0 mm   — outboard section sits at
                                            ``max(y_intermediate, body_half)
                                            + width_extend_mm``; floor
                                            width = 2 * that.
  splitter_kick_offset_mm      50.0 mm   — splitter kick X = front_axle +
                                            offset (forward).
  diffuser_kick_offset_mm      50.0 mm   — diffuser kick X = rear_axle -
                                            offset (rearward).
  splitter_front_angle_deg    +30.0      — leading-edge tangent angle
                                            above +X (up).
  diffuser_rear_angle_deg      +5.0      — trailing-edge tangent angle
                                            above -X (up).
  handle_strength               0.30     — relative Bezier handle length
                                            (|P0-P1|/|P0-P3|). Capped per
                                            section to keep the curve
                                            above ride_h.

Outputs (in ``--output-dir``)
=============================

  <stem>_underbody_left.stl     trimmed parametric UB in shell frame.
  <stem>_combined_left.stl      shell + UB concatenated.
  <stem>_boundary_3d.stl        shell open-edge loops as 10 mm curtains.
  <stem>_boundary_xy.stl        2D trim polygon as a thin extruded wall.
  <stem>_boundary.png           top-down debug of the trim polygon.
  <stem>_combined.png           10-panel debug render of shell + UB.
  <stem>_underbody_only.png     10-panel debug render of UB alone.
  <stem>_integrate_meta.json    hints, anchors, spec, face counts.

Usage::

    python integrate_underbody.py
    python integrate_underbody.py --shell-meta outputs/shell/<name>_meta.json
    python integrate_underbody.py --no-render            # skip PNGs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from math import atan2, degrees, tan, radians
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

    track = 2.0 * abs(np.mean([w["y"] for w in wheels])) * scale
    tire_od = 2.0 * np.mean([w["radius"] for w in wheels]) * scale
    tire_width = 2.0 * np.mean([w["axial_half"] for w in wheels]) * scale
    hub_z = np.mean([w["z"] for w in wheels]) * scale

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
    )
    print("[hints]")
    for k, v in hints.items():
        print(f"  {k:24s} = {v}")
    return hints


def measure_shell_anchors(shell_mm: trimesh.Trimesh, ys: list[float],
                          y_band: float = 25.0,
                          edge_band: float = 50.0) -> dict:
    """Probe the shell at each y_target in ``ys`` and record the front
    (smallest x_shell) and rear (largest x_shell) edges plus the lowest
    Z within ``edge_band`` mm of each extremity. Used as anchors for the
    splitter leading edges and the diffuser trailing edges.

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
        front = band[band[:, 0] < x_min + edge_band]
        rear = band[band[:, 0] > x_max - edge_band]
        out[y] = {
            "y_target": float(y),
            "front_x_shell": x_min,
            "front_z_shell": float(front[:, 2].min()),
            "rear_x_shell": x_max,
            "rear_z_shell": float(rear[:, 2].min()),
        }
        a = out[y]
        print(f"[anchor] Y={y:>5.0f}  "
              f"front (x,z)=({a['front_x_shell']:>7.1f}, "
              f"{a['front_z_shell']:>6.1f})  "
              f"rear (x,z)=({a['rear_x_shell']:>7.1f}, "
              f"{a['rear_z_shell']:>6.1f})")
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
                length_extend_mm: float = 100.0,
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

    body_half = hints["floor_width_mm"] / 2.0
    y_outboard = max(y_intermediate, body_half) + width_extend_mm
    floor_width = 2.0 * y_outboard
    print(f"[spec] y_intermediate={y_intermediate:.0f}  "
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
        splitter_at(y_intermediate, y_intermediate),
        splitter_at(y_intermediate, y_outboard),   # outboard = copy of intermediate
    ]
    diffuser_sections = [
        diffuser_at(0.0, 0.0),
        diffuser_at(y_intermediate, y_intermediate),
        diffuser_at(y_intermediate, y_outboard),
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
# Generate -> STL -> trimesh; transform to shell frame; keep left half
# ---------------------------------------------------------------------------

def cq_to_trimesh_via_stl(asy, stl_tmp: Path) -> trimesh.Trimesh:
    """Export a cadquery Assembly to STL and load as trimesh."""
    import cadquery as cq
    from cadquery import exporters
    compound = asy.toCompound()
    exporters.export(compound, str(stl_tmp),
                     exporters.ExportTypes.STL,
                     tolerance=0.1, angularTolerance=0.1)
    return trimesh.load(str(stl_tmp), force="mesh", process=False)


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


def keep_left_half(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Slice at y = 0 and keep y <= 0."""
    cut = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=np.array([0.0, -1.0, 0.0]),
        plane_origin=np.array([0.0, 0.0, 0.0]),
        cap=False,
    )
    cut.face_normals = None
    return cut


# ---------------------------------------------------------------------------
# Perimeter trim: drop underbody faces whose xy is far from the shell xy
# ---------------------------------------------------------------------------

def extract_shell_boundary_loops_3d(
        shell_mm: trimesh.Trimesh, min_points: int = 10,
        ignore_y_eps: float = 5.0) -> list[np.ndarray]:
    """Connected loops of open boundary edges of the shell mesh, as 3D
    polylines (Nx3 arrays). One loop per element of the returned list.

    These trace the shell's actual edges in 3D — outer body silhouette,
    wheelhouse openings, etc. — and go UP around features instead of
    being flattened to z=0 like the XY hull.

    Loops that lie entirely on the y=0 symmetry plane (the artificial
    cut from the half-shell) are filtered out.
    """
    sh = trimesh.Trimesh(
        vertices=shell_mm.vertices.copy(),
        faces=shell_mm.faces.copy(),
        process=True)
    outline = sh.outline()
    loops: list[np.ndarray] = []
    if outline is None or not hasattr(outline, "entities"):
        return loops
    for entity in outline.entities:
        try:
            pts = entity.discrete(outline.vertices)
        except Exception:
            continue
        if pts is None or len(pts) < min_points:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        if (np.abs(pts[:, 1]) < ignore_y_eps).all():
            continue       # artificial y=0 cut
        loops.append(pts)
    return loops


def export_loops_as_curtain_stl(loops: list[np.ndarray], out_path: Path,
                                 height: float = 10.0) -> None:
    """Each 3D loop becomes a thin vertical curtain (the loop polyline
    extruded DOWN by ``height`` mm in Z). The result is a single STL of
    all loops, visible in 3D viewers alongside the shell + underbody so
    the trim boundary can be inspected exactly where it lives in 3D —
    including up around the wheel arches."""
    all_v = []
    all_f = []
    off = 0
    for loop in loops:
        if len(loop) < 3:
            continue
        n = len(loop)
        top = loop.copy()
        bot = loop.copy()
        bot[:, 2] -= height
        verts = np.vstack([top, bot])
        faces = []
        for i in range(n - 1):
            faces.append([i, i + 1, i + n + 1])
            faces.append([i, i + n + 1, i + n])
        # Close the loop if its first/last vertex differ noticeably
        if not np.allclose(loop[0], loop[-1], atol=1.0):
            faces.append([n - 1, 0, n])
            faces.append([n - 1, n, 2 * n - 1])
        all_v.append(verts)
        all_f.append(np.asarray(faces, dtype=np.int64) + off)
        off += 2 * n
    if not all_v:
        print(f"[boundary 3D] no loops to export to {out_path}")
        return
    mesh = trimesh.Trimesh(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        process=False)
    mesh.export(str(out_path), file_type="stl")
    print(f"[boundary 3D] wrote {out_path}  "
          f"({len(loops)} loops, {len(mesh.faces)} triangles, "
          f"curtain height={height:.0f}mm)")


def extract_outer_boundary_polygon(shell_mm: trimesh.Trimesh,
                                     ignore_y_eps: float = 5.0,
                                     concave_ratio: float = 0.02):
    """Build a closed polygon (in xy) that traces the outer lower
    silhouette of the shell. We:

    1. Process the shell to merge duplicate vertices (CadQuery STL output
       has no vertex sharing).
    2. Collect every OPEN boundary-edge endpoint (each edge appearing in
       exactly one face). These lie on the shell's actual boundary loops
       (rocker bottoms, bumper bottoms, wheelhouse openings, etc.).
    3. Drop endpoints on the y=0 symmetry plane (artificial cut, not a
       body silhouette).
    4. Mirror the points across y=0 so the resulting hull is symmetric
       and we get a single closed contour around the WHOLE car (instead
       of a half-car with a flat side on y=0).
    5. Compute a concave hull (alpha shape) and clip it back to y ≤ 0.

    Returns a shapely Polygon in xy, in mm.
    """
    from shapely.geometry import MultiPoint, Polygon, box

    sh = trimesh.Trimesh(vertices=shell_mm.vertices.copy(),
                          faces=shell_mm.faces.copy(),
                          process=True)
    edges = sh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_pairs = unique[counts == 1]
    v = sh.vertices
    a = v[boundary_pairs[:, 0]]
    b = v[boundary_pairs[:, 1]]
    keep = ~((np.abs(a[:, 1]) < ignore_y_eps) & (np.abs(b[:, 1]) < ignore_y_eps))
    a = a[keep]; b = b[keep]
    pts_xy = np.vstack([a[:, :2], b[:, :2]])
    print(f"[boundary] {len(a):,} open boundary edges → "
          f"{len(pts_xy):,} xy points (y=0 cut filtered)")

    # Mirror so the concave hull spans both sides.
    mirrored = np.column_stack([pts_xy[:, 0], -pts_xy[:, 1]])
    all_pts = np.vstack([pts_xy, mirrored])

    mp = MultiPoint(all_pts)
    try:
        hull = mp.concave_hull(ratio=concave_ratio)
    except AttributeError:
        # shapely < 2.0
        hull = mp.convex_hull

    if isinstance(hull, Polygon) and not hull.is_empty:
        # Clip to y <= 0 (the half we care about).
        b = sh.bounds
        bb = box(b[0, 0] - 100, b[0, 1] - 100, b[1, 0] + 100, 0.0)
        clipped = hull.intersection(bb)
        if clipped.is_empty:
            clipped = hull
        # If the intersection produced a MultiPolygon, pick the largest.
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        print(f"[boundary] hull area={hull.area:.0f} mm²  "
              f"clipped (y≤0) area={clipped.area:.0f} mm²")
        return clipped
    print("[boundary] hull came back empty; falling back to convex hull")
    return mp.convex_hull


def _loop_to_polygon(loop_3d: np.ndarray, min_area: float = 100.0):
    """Project a 3D polyline to XY and try to build a shapely Polygon.
    trimesh.outline() returns open chains (first/last point not matched);
    we close the chain manually and use ``buffer(0)`` to clean up small
    self-intersections."""
    from shapely.geometry import Polygon

    xy = loop_3d[:, :2]
    if len(xy) < 4:
        return None
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[:1]])
    try:
        p = Polygon(xy)
    except Exception:
        return None
    if not p.is_valid:
        p = p.buffer(0)
        if p.is_empty:
            return None
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        if p.geom_type != "Polygon":
            return None
    if p.area < min_area:
        return None
    return p


def boundary_polygon_with_holes(shell_mm: trimesh.Trimesh,
                                 loops3d: list[np.ndarray] | None = None,
                                 ignore_y_eps: float = 5.0,
                                 min_hole_area: float = 5000.0):
    """Build a 2D polygon WITH HOLES from the shell's 3D open-edge loops.

    Largest projected loop = exterior body silhouette.
    Smaller loops contained within the exterior = holes (wheelhouse
    openings, etc.).

    Returns a shapely Polygon (potentially with holes), clipped to y ≤ 0
    to match the left-half underbody. None if no usable loops.
    """
    from shapely.geometry import Polygon, box
    from shapely import affinity
    from shapely.ops import unary_union

    if loops3d is None:
        loops3d = extract_shell_boundary_loops_3d(
            shell_mm, ignore_y_eps=ignore_y_eps)

    polys = []
    for loop in loops3d:
        p = _loop_to_polygon(loop)
        if p is not None:
            polys.append(p)
    print(f"[boundary] {len(polys)}/{len(loops3d)} loops → valid polygons")
    if not polys:
        return None

    polys.sort(key=lambda p: p.area, reverse=True)
    outer_half = polys[0]
    # Mirror exterior across y=0 so the boundary covers both halves;
    # we clip back to y≤0 at the end.
    outer_mirror = affinity.scale(outer_half, xfact=1, yfact=-1, origin=(0, 0))
    outer_full = unary_union([outer_half, outer_mirror]).buffer(0)
    if outer_full.geom_type != "Polygon":
        # MultiPolygon — pick the largest piece
        outer_full = max(outer_full.geoms, key=lambda g: g.area)

    holes = []
    for p in polys[1:]:
        if p.area < min_hole_area:
            continue
        if not outer_full.contains(p):
            continue
        holes.append(p)
        # Mirror the hole too so the full-car polygon has both wheel arches
        p_mirror = affinity.scale(p, xfact=1, yfact=-1, origin=(0, 0))
        if outer_full.contains(p_mirror):
            holes.append(p_mirror)

    exterior_coords = list(outer_full.exterior.coords)
    hole_coords = [list(h.exterior.coords) for h in holes]
    poly_with_holes = Polygon(exterior_coords, hole_coords)

    # Clip to y ≤ 0
    b = poly_with_holes.bounds
    clip = box(b[0] - 100, b[1] - 100, b[2] + 100, 0.0)
    clipped = poly_with_holes.intersection(clip)
    if clipped.is_empty:
        clipped = poly_with_holes
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)

    n_holes = len(clipped.interiors) if hasattr(clipped, "interiors") else 0
    print(f"[boundary] polygon-with-holes: outer area={outer_full.area:.0f} "
          f"mm², {len(holes)} holes (post-mirror), "
          f"clipped area={clipped.area:.0f} mm² ({n_holes} holes after y≤0 clip)")
    return clipped


def trim_to_shell_boundary(underbody: trimesh.Trimesh,
                            boundary_polygon) -> trimesh.Trimesh:
    """Trim UB faces against the shell's outer boundary polygon (xy).

    Uses an ALL-VERTICES test: a face is kept only when all three of its
    vertices' (x, y) lie inside the polygon. This is stricter than a
    centroid test — it prevents large faces from straddling the boundary
    with the centroid inside but corners poking past. Combine with a
    fine subdivision (small ``max_edge``) for a sharp trim.
    """
    try:
        from shapely import contains_xy
    except ImportError:
        from shapely.vectorized import contains as contains_xy
    if not hasattr(boundary_polygon, "geom_type"):
        keep = np.ones(len(underbody.faces), dtype=bool)
    else:
        verts_xy = underbody.vertices[:, :2]
        vert_inside = contains_xy(
            boundary_polygon, verts_xy[:, 0], verts_xy[:, 1])
        keep = vert_inside[underbody.faces].all(axis=1)
    keep_idx = np.flatnonzero(keep)
    print(f"[trim] boundary-polygon (all-verts inside) -> keep "
          f"{int(keep.sum()):,}/{len(keep):,} faces "
          f"(dropped {int((~keep).sum()):,})")
    if len(keep_idx) == 0:
        # nothing kept — return an empty trimesh to avoid crashing the rest
        return trimesh.Trimesh(vertices=underbody.vertices[:0],
                                faces=np.zeros((0, 3), dtype=np.int64),
                                process=False)
    kept = underbody.submesh([keep_idx], append=True)
    if isinstance(kept, list):
        kept = kept[0]
    return kept


def export_boundary_polygon_stl(polygon, out_path: Path,
                                 z_base: float = 0.0,
                                 z_height: float = 5.0) -> None:
    """Save the 2D boundary polygon as a thin extruded WALL STL so it
    can be inspected in 3D viewers next to the shell + underbody.

    The wall sits between z=z_base and z=z_base + z_height (no top or
    bottom caps — those would require a 2D triangulation engine that
    isn't always available in this env). The wall traces the polygon's
    exterior, which is what matters for visual inspection of the trim
    boundary."""
    if not hasattr(polygon, "exterior"):
        print(f"[boundary STL] polygon has no exterior; skipping {out_path}")
        return
    coords = np.array(polygon.exterior.coords)
    if len(coords) >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    n = len(coords)
    if n < 3:
        print(f"[boundary STL] polygon has <3 unique points; skipping")
        return
    bottom = np.column_stack(
        [coords[:, 0], coords[:, 1], np.full(n, z_base)])
    top = np.column_stack(
        [coords[:, 0], coords[:, 1], np.full(n, z_base + z_height)])
    verts = np.vstack([bottom, top])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        # Two triangles per edge.
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    mesh = trimesh.Trimesh(
        vertices=verts, faces=np.array(faces, dtype=np.int64), process=False)
    mesh.export(str(out_path), file_type="stl")
    print(f"[boundary STL] wrote {out_path}  "
          f"({len(coords)} polygon points; wall at "
          f"z=[{z_base:.1f}, {z_base + z_height:.1f}])")


def trim_to_shell_footprint(underbody: trimesh.Trimesh,
                             shell_mm: trimesh.Trimesh) -> trimesh.Trimesh:
    """Trim underbody faces whose xy lies outside the shell's xy
    silhouette. For each face centroid, cast a vertical ray upward
    starting just below the shell's lowest z — if the ray hits the shell
    anywhere above, the face is under the silhouette and is kept; if not,
    the face is poking past the body's outline (typically the corners of
    the floor rectangle past the rounded bumper) and is dropped."""
    centroids = underbody.vertices[underbody.faces].mean(axis=1)
    n = len(centroids)
    z_lo = float(shell_mm.bounds[0, 2]) - 10.0
    origins = np.column_stack([centroids[:, 0], centroids[:, 1],
                                np.full(n, z_lo)])
    directions = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(shell_mm) \
        if trimesh.ray.has_embree else \
        trimesh.ray.ray_triangle.RayMeshIntersector(shell_mm)
    keep = inter.intersects_any(ray_origins=origins, ray_directions=directions)
    print(f"[trim] under_shell  -> keep {int(keep.sum()):,}/{n:,} faces "
          f"(dropped {int((~keep).sum()):,} outside the xy silhouette)")
    kept = underbody.submesh([np.flatnonzero(keep)], append=True)
    if isinstance(kept, list):
        kept = kept[0]
    return kept


# ---------------------------------------------------------------------------
# Render (reuses paramub.shell_render via subprocess)
# ---------------------------------------------------------------------------

RENDER_FACE_BUDGET = 30_000


def _decimate_subproc(src_stl: Path, dst_stl: Path, target_faces: int) -> None:
    """PyVista decimate in a subprocess (PyVista / matplotlib are happier
    not loaded in the same interpreter as cadquery).

    Uses decimate_pro for high-ratio reductions (>= 0.9) since
    vtkDecimatePro handles extreme ratios more robustly than vtkDecimate.
    """
    code = """
import sys
import pyvista as pv
src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
mesh = pv.read(src)
n = mesh.n_cells
if n <= target:
    mesh.save(dst, binary=True)
else:
    ratio = 1.0 - target / float(n)
    if ratio >= 0.9:
        try:
            out = mesh.decimate_pro(ratio, preserve_topology=False)
        except Exception:
            out = mesh.decimate(min(ratio, 0.95))
    else:
        out = mesh.decimate(ratio)
    out.save(dst, binary=True)
"""
    cmd = [sys.executable, "-c", code, str(src_stl), str(dst_stl), str(target_faces)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"decimate failed: {proc.stderr[-1000:]}")


def render_panel(*, stl_path: Path, title: str, out_png: Path,
                 view_bounds: np.ndarray, scratch_dir: Path,
                 keep_color: str = "#b8c0c8",
                 remove_color: str = "#dc2626",
                 remove_indices: list[int] | None = None) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stl_path": str(stl_path),
        "remove_indices": remove_indices or [],
        "title": title,
        "out_path": str(out_png),
        "view_bounds": view_bounds.tolist(),
        "keep_color": keep_color,
        "remove_color": remove_color,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                      dir=str(scratch_dir)) as fh:
        json.dump(payload, fh)
        payload_path = Path(fh.name)
    cmd = [sys.executable, "-m", "paramub.shell_render", str(payload_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    payload_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        print("[render stderr]:", proc.stderr[-2000:])
        raise RuntimeError(f"render failed rc={proc.returncode}")
    if proc.stdout.strip():
        print(proc.stdout.rstrip())


def render_combined(shell_mm: trimesh.Trimesh,
                     underbody: trimesh.Trimesh,
                     out_png: Path, scratch_dir: Path,
                     title_suffix: str = "") -> None:
    """Render shell + underbody together. Both halves are decimated
    independently to fit in matplotlib's face budget, then concatenated.
    Indices [0..N_shell_deci) are shell faces (grey); the rest are
    underbody faces (rendered as 'REMOVE' = orange for contrast)."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    shell_full = scratch_dir / "shell_full.stl"
    ub_full = scratch_dir / "ub_full.stl"
    shell_deci = scratch_dir / "shell_deci.stl"
    ub_deci = scratch_dir / "ub_deci.stl"
    shell_mm.export(str(shell_full), file_type="stl")
    underbody.export(str(ub_full), file_type="stl")

    budget = RENDER_FACE_BUDGET
    # Split budget roughly evenly between shell and underbody.
    _decimate_subproc(shell_full, shell_deci, budget // 2)
    _decimate_subproc(ub_full, ub_deci, budget // 2)

    shell_d = trimesh.load(str(shell_deci), force="mesh", process=False)
    ub_d = trimesh.load(str(ub_deci), force="mesh", process=False)
    n_shell_d = len(shell_d.faces)
    combined = trimesh.util.concatenate([shell_d, ub_d])
    combined_stl = scratch_dir / "combined_deci.stl"
    combined.export(str(combined_stl), file_type="stl")

    n_total = len(combined.faces)
    remove_indices = list(range(n_shell_d, n_total))
    bounds = combined.bounds.copy()
    title = (f"Shell ({len(shell_mm.faces):,} → {n_shell_d:,} faces, grey) + "
             f"parametric underbody ({len(underbody.faces):,} → "
             f"{len(ub_d.faces):,} faces, orange){title_suffix}")
    render_panel(
        stl_path=combined_stl,
        title=title,
        out_png=out_png,
        view_bounds=bounds,
        scratch_dir=scratch_dir,
        keep_color="#b8c0c8",       # shell — grey
        remove_color="#fb923c",     # underbody — orange
        remove_indices=remove_indices,
    )


def render_underbody_only(underbody: trimesh.Trimesh,
                            out_png: Path, scratch_dir: Path) -> None:
    """Render just the underbody (decimated). Useful when the combined
    render is hard to read."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    ub_full = scratch_dir / "ub_only_full.stl"
    ub_deci = scratch_dir / "ub_only_deci.stl"
    underbody.export(str(ub_full), file_type="stl")
    _decimate_subproc(ub_full, ub_deci, RENDER_FACE_BUDGET)
    bounds = underbody.bounds.copy()
    render_panel(
        stl_path=ub_deci,
        title=f"Parametric underbody only ({len(underbody.faces):,} faces, "
              f"shell-aligned, left half)",
        out_png=out_png,
        view_bounds=bounds,
        scratch_dir=scratch_dir,
        keep_color="#fb923c",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Integrate parametric underbody with extracted upper shell.")
    p.add_argument("--shell-meta", type=Path, default=DEFAULT_META,
                   help=f"Shell metadata JSON (default {DEFAULT_META}).")
    p.add_argument("--output-dir", "-o", type=Path,
                   default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory (default {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--no-render", action="store_true",
                   help="Skip the matplotlib render (still writes STL).")
    return p.parse_args()


def main():
    args = parse_args()
    meta = json.loads(args.shell_meta.read_text())
    shell_stl_path = Path(meta["final_path"])
    if not shell_stl_path.is_absolute():
        shell_stl_path = args.shell_meta.parent / shell_stl_path.name
    print(f"[load] shell STL = {shell_stl_path}")
    shell = trimesh.load(str(shell_stl_path), force="mesh", process=False)
    print(f"  shell faces={len(shell.faces):,}  verts={len(shell.vertices):,}")

    # SCALE so wheelbase = 2700 mm. All ParamUB-mm calculations after this
    # are at scaled (mm) coordinates.
    wheels = meta["wheels_3d"]
    wb_shell = abs(wheels[0]["x"] - wheels[1]["x"])
    scale = TARGET_WHEELBASE_MM / wb_shell
    print(f"[scale] shell wheelbase = {wb_shell:.4f}  →  scale = {scale:.2f}  "
          f"(target wheelbase = {TARGET_WHEELBASE_MM:.0f} mm)")

    # Scale the shell vertices into mm and rebuild as a trimesh.
    shell_mm = trimesh.Trimesh(
        vertices=np.asarray(shell.vertices) * scale,
        faces=np.asarray(shell.faces),
        process=False,
    )

    hints = extract_hints(shell, meta, scale)
    # Measure shell front/rear edges at the section Y values used to anchor
    # the splitter and diffuser Bezier lofts.
    anchors = measure_shell_anchors(shell_mm, ys=[0.0, 700.0])
    # Wheelhouses extend +100mm past the rocker line so the boundary trim
    # has material to clip against the body silhouette.
    lat_overrides = lateral_clearance_overrides(hints, extra_extend_mm=100.0)
    spec = build_spec(hints, anchors, lat_overrides=lat_overrides)

    print("\n[paramub] build_underbody (half_only=True) ...")
    from paramub.ub_assem import build_underbody
    asy, layout = build_underbody(spec, half_only=True)
    print(f"[paramub] layout = {layout}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.output_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    # Export the raw assembly STL (still in ParamUB frame).
    raw_stl = scratch / "underbody_paramub_frame.stl"
    ub_raw = cq_to_trimesh_via_stl(asy, raw_stl)
    print(f"[paramub] STL faces={len(ub_raw.faces):,}")

    # CadQuery's STL emits 2-3 giant triangles for the central flat floor.
    # Subdivide before slicing/trimming so the y=0 cut is clean and the
    # perimeter trim's per-face check is meaningful.
    # Finer = sharper boundary trim (no large triangle straddling the
    # polygon boundary). 25 mm gives a clean cut for the rocker / wheel
    # arch features without blowing up face count too much.
    max_edge = 25.0
    print(f"[subdivide] target edge ≤ {max_edge:.0f} mm ...")
    ub_dense = subdivide_to_edge(ub_raw, max_edge=max_edge)
    print(f"[subdivide] faces: {len(ub_raw.faces):,} → {len(ub_dense.faces):,}")

    # Transform to shell frame, take left half, trim perimeter.
    ub_aligned = align_to_shell_frame(ub_dense, hints["midpoint_x_shell_mm"])
    ub_left = keep_left_half(ub_aligned)
    print(f"[left]  faces={len(ub_left.faces):,}  "
          f"y range=({ub_left.bounds[0,1]:.1f}, {ub_left.bounds[1,1]:.1f})")
    # Build a polygon from the shell's outer open-boundary loop and
    # trim UB faces whose centroid xy lies outside it. This is closer to
    # what the user wanted ("project the shell EDGE onto the UB") than
    # the previous full-mesh ray-up test, which kept floor area under
    # the hood/trunk that extended beyond the body's lower silhouette.
    base_for_debug = args.shell_meta.stem.replace("_meta", "")
    # Extract the shell's 3D open-edge loops for visualisation. trimesh
    # breaks the outer silhouette into chained fragments at the wheel
    # arches, so a single closed outer loop isn't directly available —
    # the 2D trim polygon below still comes from the concave hull of
    # all open-edge endpoints, which handles the broken outer cleanly.
    loops3d = extract_shell_boundary_loops_3d(shell_mm)
    print(f"[boundary] {len(loops3d)} open-edge loops in 3D")
    # 3D curtain STL — the actual shell edges in 3D, going up around
    # the wheel arches / following the rocker line / etc. Open this
    # alongside the underbody STL in Blender to see the trim boundary
    # in its true 3D location.
    export_loops_as_curtain_stl(
        loops3d,
        args.output_dir / f"{base_for_debug}_boundary_3d.stl",
        height=10.0,
    )
    # 2D trim polygon: concave hull of open-edge endpoints (handles the
    # broken outer chain by sampling all points). The UB already gets
    # wheelhouse openings from build_wheelhouse_solid, so we don't need
    # holes here for the wheel arches — the outer silhouette alone is
    # what drives the trim.
    boundary = extract_outer_boundary_polygon(shell_mm)
    _dump_boundary_debug(
        shell_mm, boundary,
        args.output_dir / f"{base_for_debug}_boundary.png")
    export_boundary_polygon_stl(
        boundary,
        args.output_dir / f"{base_for_debug}_boundary_xy.stl",
        z_base=0.0, z_height=5.0)
    ub_trim = trim_to_shell_boundary(ub_left, boundary)
    print(f"[trim]  faces={len(ub_trim.faces):,}  "
          f"y range=({ub_trim.bounds[0,1]:.1f}, {ub_trim.bounds[1,1]:.1f})")

    base = args.shell_meta.stem.replace("_meta", "")
    out_ub_stl = args.output_dir / f"{base}_underbody_left.stl"
    out_combined_stl = args.output_dir / f"{base}_combined_left.stl"
    out_render = args.output_dir / f"{base}_combined.png"

    ub_trim.export(str(out_ub_stl), file_type="stl")
    print(f"[out] {out_ub_stl}")

    combined = trimesh.util.concatenate([shell_mm, ub_trim])
    combined.export(str(out_combined_stl), file_type="stl")
    print(f"[out] {out_combined_stl}")

    if not args.no_render:
        try:
            render_combined(shell_mm, ub_trim, out_render, scratch)
            print(f"[out] {out_render}")
        except Exception as exc:
            print(f"[warn] render_combined failed: {exc}")
        out_render_ub = args.output_dir / f"{base}_underbody_only.png"
        try:
            render_underbody_only(ub_trim, out_render_ub, scratch)
            print(f"[out] {out_render_ub}")
        except Exception as exc:
            print(f"[warn] render_underbody_only failed: {exc}")

    # Dump hints + final spec for debugging.
    debug_meta = {
        "hints": hints,
        "shell_anchors": anchors,
        "spec": _spec_to_jsonable(spec),
        "layout": layout,
        "underbody_paramub_frame_faces": int(len(ub_raw.faces)),
        "underbody_left_faces": int(len(ub_left.faces)),
        "underbody_trimmed_faces": int(len(ub_trim.faces)),
        "out_underbody_stl": str(out_ub_stl),
        "out_combined_stl": str(out_combined_stl),
    }
    (args.output_dir / f"{base}_integrate_meta.json").write_text(
        json.dumps(debug_meta, indent=2, default=float))
    return 0


def _dump_boundary_debug(shell_mm, polygon, out_png: Path) -> None:
    """Save a top-down PNG: shell xy footprint + boundary polygon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    # Shell vertices (xy scatter)
    v = shell_mm.vertices
    ax.scatter(v[::50, 0], v[::50, 1], s=0.5, c="#888", alpha=0.3,
                label="shell verts (subsampled)")
    # Boundary polygon
    if hasattr(polygon, "exterior"):
        ex = np.asarray(polygon.exterior.coords)
        ax.plot(ex[:, 0], ex[:, 1], "r-", linewidth=1.5,
                 label="boundary polygon (concave hull of open edges)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Shell xy footprint + trim boundary polygon")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[debug] wrote {out_png}")


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
