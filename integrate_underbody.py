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

Usage::

    python integrate_underbody.py
    python integrate_underbody.py --shell-meta outputs/shell/<name>_meta.json
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


def diffuser_angle_from_hints(hints: dict) -> float:
    """Pick a diffuser ramp angle so the trailing edge of the diffuser
    sits at the shell's actual rear-edge bottom — i.e. the lowest shell
    vertex within a small band of the rearmost x. This places the diffuser
    exit flush with the bumper TIP, not the much lower point at the
    underside of the body well in front of the bumper.

    Rise needed = rear_edge_min_z - ride_height.
    Run available = rear_overhang_mm (distance from rear axle to body tail).
    angle = atan2(rise, run)  (in degrees), clamped to [0°, 25°].
    """
    rise = max(0.0, hints["rear_edge_min_z_mm"] - hints["ride_height_mm"])
    run = max(1.0, hints["rear_overhang_mm"])
    angle = degrees(atan2(rise, run))
    angle = float(np.clip(angle, 0.0, 25.0))
    print(f"[diffuser] rear-edge z={hints['rear_edge_min_z_mm']:.1f} mm  "
          f"ride_height={hints['ride_height_mm']:.1f} mm  → "
          f"rise={rise:.1f} mm  run={run:.1f} mm  →  angle={angle:.2f}°")
    return angle


def lateral_clearance_overrides(hints: dict) -> dict:
    """Per-wheel lateral_clearance for the wheelhouse so its outboard
    edge reaches the shell's rocker line at that wheel.

    The wheelhouse arch is symmetric in y about y_track with half-length
    = (tire_width + 2 * lateral_clearance) / 2 = tire_width/2 +
    lateral_clearance. So to make the outboard edge land at the shell's
    outboard rocker y (= ``wheel_outboard_y``), we need:

        lateral_clearance = wheel_outboard_y - track/2 - tire_width/2

    Inboard side is extended symmetrically (the wheelhouse builder
    doesn't support per-side asymmetry) — typically this just makes the
    inboard wall slightly wider, which is harmless against the floor.
    """
    overrides: dict[tuple[str, str], float] = {}
    track_half = hints["track_mm"] / 2.0
    tire_half = hints["tire_width_mm"] / 2.0
    for axle_label in ("front", "rear"):
        outboard = hints["wheel_outboard_y_mm"].get(axle_label)
        if outboard is None:
            continue
        need = max(0.0, outboard - track_half - tire_half)
        # Add a small margin so the wheelhouse fully reaches the rocker.
        need += 5.0
        overrides[(axle_label, "left")] = need
        overrides[(axle_label, "right")] = need
        print(f"[wheelhouse] {axle_label}: shell rocker y={outboard:.0f}  "
              f"→ lateral_clearance={need:.1f} mm  "
              f"(track/2={track_half:.0f}, tire_half={tire_half:.0f})")
    return overrides


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_spec(hints: dict, diffuser_angle: float,
                lat_overrides: dict | None = None):
    """Construct an UnderbodySpec from the shell-derived hints."""
    from paramub import UnderbodySpec, WheelSpec, TireSpec, SpokeSpec

    # Tire: hit target OD + section width while keeping the standard
    # 25.4*rim_in + 2*sidewall = OD relation. We pick rim_diameter_in to
    # land aspect_ratio in a plausible band (30..70). 18" rim is a safe
    # starting point for ~640 mm OD with section width ~245 mm.
    target_od = hints["tire_od_mm"]
    target_sw = hints["tire_width_mm"]
    # section_width chosen to match the extracted axial width. Aspect ratio
    # is then forced by (target_od/2 - rim_radius_mm) / section_width * 100.
    rim_in = 18.0
    rim_r = rim_in * 25.4 / 2.0
    aspect = max(25.0, min(75.0,
                            (target_od / 2.0 - rim_r) / target_sw * 100.0))
    # Re-derive section width so OD lands exactly on target.
    section_w = (target_od / 2.0 - rim_r) / (aspect / 100.0)

    tire = TireSpec(
        section_width_mm=section_w,
        aspect_ratio=aspect,
        rim_diameter_in=rim_in,
        # Crown radius must satisfy crown_radius > tread_width/2 + 5; clamp.
        tread_width_mm=max(60.0, 0.85 * target_sw),
        crown_radius_mm=max(0.85 * target_sw / 2 + 50.0, 0.6 * target_od),
        # Cosmetic tire features at reasonable defaults.
        sidewall_bulge_mm=6.0,
        shoulder_radius_mm=20.0,
        rim_flange_mm=14.0,
    )

    # Spoke: keep light; the user explicitly said no need to extract.
    spoke = SpokeSpec(
        wheel_width_mm=target_sw,
        rim_diameter_in=rim_in,
        num_spokes=5,
    )

    spec = UnderbodySpec(
        wheelbase_mm=hints["wheelbase_mm"],
        front_overhang_mm=hints["front_overhang_mm"],
        rear_overhang_mm=hints["rear_overhang_mm"],
        track_front_mm=hints["track_mm"],
        track_rear_mm=hints["track_mm"],
        ride_height_mm=hints["ride_height_mm"],
        floor_width_mm=hints["floor_width_mm"],
        diffuser_angle_deg=diffuser_angle,
        diffuser_radius_mm=min(250.0, 0.4 * hints["rear_overhang_mm"]),
        wheel_house_axial_clearance_mm=20.0,
        wheel_house_lateral_clearance_mm=25.0,
        front_steering_clearance_mm=0.0,    # we don't model steering
        rear_steering_clearance_mm=0.0,
        front_wheel_house_fillet_mm=30.0,
        rear_wheel_house_fillet_mm=30.0,
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

def extract_outer_boundary_polygon(shell_mm: trimesh.Trimesh,
                                     ignore_y_eps: float = 5.0,
                                     concave_ratio: float = 0.04):
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


def trim_to_shell_boundary(underbody: trimesh.Trimesh,
                            boundary_polygon) -> trimesh.Trimesh:
    """Trim UB faces whose centroid xy lies outside the shell's outer
    boundary polygon (projected to xy)."""
    try:
        from shapely import contains_xy
    except ImportError:
        from shapely.vectorized import contains as contains_xy
    centroids = underbody.vertices[underbody.faces].mean(axis=1)
    if hasattr(boundary_polygon, "geom_type"):
        keep = contains_xy(boundary_polygon, centroids[:, 0], centroids[:, 1])
    else:
        keep = np.ones(len(centroids), dtype=bool)
    keep_idx = np.flatnonzero(keep)
    print(f"[trim] boundary-polygon  -> keep {int(keep.sum()):,}/"
          f"{len(keep):,} faces (dropped {int((~keep).sum()):,})")
    if len(keep_idx) == 0:
        # nothing kept — return an empty trimesh to avoid crashing the rest
        return trimesh.Trimesh(vertices=underbody.vertices[:0],
                                faces=np.zeros((0, 3), dtype=np.int64),
                                process=False)
    kept = underbody.submesh([keep_idx], append=True)
    if isinstance(kept, list):
        kept = kept[0]
    return kept


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
    not loaded in the same interpreter as cadquery)."""
    code = """
import sys
import pyvista as pv
src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
mesh = pv.read(src)
n = mesh.n_cells
if n <= target:
    mesh.save(dst, binary=True)
else:
    mesh.decimate(1.0 - target / float(n)).save(dst, binary=True)
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
    diffuser_angle = diffuser_angle_from_hints(hints)
    lat_overrides = lateral_clearance_overrides(hints)
    spec = build_spec(hints, diffuser_angle, lat_overrides=lat_overrides)

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
    max_edge = 60.0
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
    boundary = extract_outer_boundary_polygon(shell_mm)
    # Dump the boundary polygon as a small debug PNG so we can see what
    # the trim is actually using.
    _dump_boundary_debug(shell_mm, boundary,
                         args.output_dir / f"{args.shell_meta.stem.replace('_meta', '')}_boundary.png")
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
        render_combined(shell_mm, ub_trim, out_render, scratch)
        print(f"[out] {out_render}")
        out_render_ub = args.output_dir / f"{base}_underbody_only.png"
        render_underbody_only(ub_trim, out_render_ub, scratch)
        print(f"[out] {out_render_ub}")

    # Dump hints + final spec for debugging.
    debug_meta = {
        "hints": hints,
        "diffuser_angle_deg": diffuser_angle,
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
