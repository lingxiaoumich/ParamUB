"""Extract the upper-body shell from a reconstructed car mesh.

The input is a watertight mesh (typically a "shadowfill" reconstruction
from an UltraShape / Hunyuan3D pipeline) covering the entire vehicle
in any axis convention. The output is a single-component upper-body
shell ready to be merged with a parametric underbody.

Pipeline overview
=================

Each step modifies (or labels faces of) the working mesh and exports
an STL in debug mode.

Step 1 — canonicalize
    Load the mesh; auto-detect the input axis convention (which axis
    is "up" and which is the longitudinal length); rotate and translate
    so the canonical frame holds:
        +x = rearward (so the front of the car sits at −x)
        +y = lateral (left / right)
        +z = up
    The body bounding box is centred on x = 0 and z = 0 is snapped to
    the lowest mesh vertex (so the four wheels rest on the ground
    plane). The lateral midline (y = 0) is the car's symmetry plane.

Step 2 — clean planar cut at y = 0
    Slice the canonical mesh at the y = 0 symmetry plane using
    ``trimesh.intersections.slice_mesh_plane`` and keep the y ≤ 0
    half (the left side). Faces that straddle the plane are split:
    portions on the +y side are dropped and new vertices are inserted
    exactly on y = 0 so the cut edge is a clean planar boundary loop.

Step 3 — wheel (x, y) footprints
    Take a thin z-slice near the ground (z < ``z_band_height``) and
    cluster the resulting (x, y) centroids along x into ``n_expected``
    groups (default 2 — front-left and rear-left, since the body is
    already half-cropped). Each cluster's mean gives the wheel's
    longitudinal and lateral hub position.

Step 4 — wheel hub Z + radius
    For each wheel-xy footprint:
    1. Take a thin Y-plane slab at ``y = w.y`` (the wheel's lateral
       position). The Y-slab cuts cleanly through the tire's circular
       silhouette in X-Z without picking up the wheelhouse arch above
       (an X-slab at constant x would).
    2. Run mesh-adjacency-based connected-component analysis on the
       slab's face subset. The tire is one component; the wheelhouse
       walls / suspension / brake disc are separate components.
    3. Pick the component that contains the most "ground-contact"
       faces (centroid z ≤ ``contact_z_max`` and |x − w.x| ≤
       ``contact_x_window``) — that's the tire.
    4. Fit a circle in X-Z to that component's centroids. The fit's
       (cx, cz) is the hub position. Radius = ``max(r_fit, cz)`` —
       the ground-contact constraint that the wheel touches z = 0
       so the tire OD equals cz.
    5. Tire half-width is the 95th-percentile y-extent of the ground-
       contact patch, clamped to [0.20, 0.40] × radius.

Step 5 — wheel cylinder removal
    For each wheel, mark every body face whose centroid sits inside
    the tight cylinder (radius = r × ``radius_factor``, axial-half
    = axial_half × ``axial_factor``; defaults 1.00 / 1.00) as
    REMOVE. The cylinder axis is +y (the wheel spin axis); the
    cross-section is a circle in X-Z. After marking, drop any
    debris — connected components of KEEP faces that got
    disconnected from the main body by the cylinder cut. A protect
    mask preserves any face touching y = 0 so the symmetry edge
    stays clean.

Step 6a — wheel-center ray visibility
    Cast Fibonacci-sphere rays from a structured set of origins
    inside each wheel cylinder (hub centre + 6 points on the upper
    half of the cylinder's inboard edge). The intersector is built
    from the post-s5 KEEP submesh so rays travel through the empty
    cylinder region and first-hit the wheelhouse-interior walls /
    suspension. All hits are collected first; REMOVE is applied in
    one step at the end (order-independent). Hits outside a
    wheelhouse zone (2.5 × r × 2.0 × axial_half) or above a z-cap
    (2.2 × hub_z) are rejected to prevent upper-body penetration.

Step 6b — wheelhouse-zone geometric backstop
    For every face whose centroid sits inside an expanded cylinder
    zone (1.4 × r × 1.7 × axial_half) AND whose outward normal
    points sufficiently toward the wheel hub (cos ≥ 0.30), mark
    REMOVE. This catches inward-facing wheelhouse-interior surfaces
    that the ray pass cannot reach because suspension / brake-disc
    geometry occludes every ray. The zone is asymmetric — the
    outboard side keeps the symmetric ``axial_factor × axial_half``
    extent; the inboard side extends to y = 0 to cover the inboard
    wheelhouse wall and any inboard suspension at any depth.

Step 7 — lower-perimeter cut (visibility classifier with cut zone)
    Per-face two-hemisphere visibility classification:
    * For each face, fire one ray from its (epsilon-nudged) centroid
      toward every Fibonacci-sphere direction whose outward elevation
      lies outside the ±15° equator band and whose hemisphere is on
      the triangle's outward side.
    * A ray that hits no other triangle proves the face is visible
      from "+∞·d". Label:
        KEEP   if any escape with elevation > +15° (seen from above)
        REMOVE if no upper escape but some lower escape (seen only
               from below)
        DROP   otherwise (interior / sealed cavity)
    A cheap geometric prefilter pre-marks "obvious underbody" faces
    (bottom 31 % of z-envelope AND downward normal) so they skip the
    lower-pass casting.

    Crucially the classifier is constrained to a **cut zone**:
        cut_zone = (floor band: z < floor_z_max)
                 ∪ (wheel cylinders, enlarged + outboard end extended
                    to the body's outboard y-limit so the cylinder is
                    flush with the floor slab on the outboard side)
    Any face OUTSIDE the cut zone is forced to KEEP regardless of the
    classifier verdict. This prevents the classifier from punching
    holes in distant body parts (bumper intake, mirror underside,
    headlight, vents) — those features sit outside the cut zone and
    are structurally protected.

    Finally, ``keep_largest_component`` is called on the KEEP set
    (without the y=0 protect mask) so the output is a single
    connected upper shell, with tiny y=0-edge slivers dropped along
    with the rest of the debris.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh


# Public label constants used by ``s7_classify_visibility``.
LABEL_DROP = 0
LABEL_REMOVE = 1
LABEL_KEEP = 2


# ===========================================================================
# Step 1: canonicalize axes + centering + ground snap
# ===========================================================================

@dataclass
class CanonicalFrame:
    """Records the transform applied to the input mesh. The forward
    transform is ``v_canonical = (v_input @ R.T) + t``."""
    R: np.ndarray
    t: np.ndarray
    original_bounds: np.ndarray
    canonical_bounds: np.ndarray

    def inverse_apply(self, v_canon: np.ndarray) -> np.ndarray:
        return (v_canon - self.t) @ self.R


def _load_mesh(path: Path) -> trimesh.Trimesh:
    """Load STL/glb regardless of file extension (UltraShape outputs are
    glb-with-.stl filename)."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    ft = "glb" if magic == b"glTF" else None
    m = trimesh.load(str(path), file_type=ft, force="mesh")
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"Failed to load a usable mesh from {path}")
    return m


def _detect_orig_axes(bounds: np.ndarray) -> tuple[int, int, int]:
    """Return (up, length, lateral) axis indices in the original frame."""
    extents = bounds[1] - bounds[0]
    rel_min = np.abs(bounds[0]) / (extents + 1e-9)
    cands = np.flatnonzero(rel_min < 0.05)
    if len(cands) == 0:
        up = int(np.argmin(extents))
    elif len(cands) == 1:
        up = int(cands[0])
    else:
        up = int(cands[np.argmin(extents[cands])])
    horiz = [a for a in (0, 1, 2) if a != up]
    length = horiz[int(np.argmax([extents[a] for a in horiz]))]
    lateral = [a for a in horiz if a != length][0]
    return up, length, lateral


def _build_rotation(up: int, length: int, lateral: int,
                    flip_length: bool) -> np.ndarray:
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, length] = -1.0 if flip_length else 1.0
    R[1, lateral] = 1.0
    R[2, up] = 1.0
    if np.linalg.det(R) < 0:
        R[1] = -R[1]
    return R


def s1_canonicalize(in_path: Path | str,
                    flip_length: bool = True,
                    verbose: bool = True
                    ) -> tuple[trimesh.Trimesh, CanonicalFrame]:
    """Load the mesh and rotate it into the canonical frame:
    +x rearward, +y lateral, +z up, body centered on x = 0,
    lowest mesh vertex snapped to z = 0."""
    in_path = Path(in_path)
    full = _load_mesh(in_path)
    original_bounds = full.bounds.copy()
    up, length, lateral = _detect_orig_axes(full.bounds)
    R = _build_rotation(up, length, lateral, flip_length=flip_length)
    rotated = full.vertices @ R.T
    mid_x = 0.5 * (rotated[:, 0].min() + rotated[:, 0].max())
    z_floor = rotated[:, 2].min()
    mid_y = 0.5 * (rotated[:, 1].min() + rotated[:, 1].max())
    t = np.array([-mid_x, -mid_y, -z_floor], dtype=np.float64)
    canonical_verts = rotated + t
    new = trimesh.Trimesh(vertices=canonical_verts,
                          faces=full.faces.copy(),
                          process=False)
    new.face_normals = None
    frame = CanonicalFrame(R=R, t=t,
                           original_bounds=original_bounds,
                           canonical_bounds=new.bounds.copy())
    if verbose:
        print(f"[s1] input axes: up={'XYZ'[up]} length={'XYZ'[length]} "
              f"lateral={'XYZ'[lateral]}  flip_length={flip_length}")
        print(f"[s1] canonical bounds:\n{new.bounds}")
        print(f"[s1] canonical extent: {(new.bounds[1] - new.bounds[0]).round(4).tolist()}")
    return new, frame


# ===========================================================================
# Step 2: clean planar cut at y = 0
# ===========================================================================

def clean_y0_boundary(mesh: trimesh.Trimesh,
                       verbose: bool = True) -> trimesh.Trimesh:
    """Snap "tooth-tip" vertices to exactly y=0.

    Step 2 (``s2_keep_left``) gives a clean planar cut, but the later
    cylinder / wheelhouse / visibility steps re-cut the mesh and can
    produce open boundary edges where ONE endpoint sits exactly at y=0
    (a survivor from step 2) and the OTHER drifts a few millimetres
    inboard. The resulting boundary looks jagged in 3D viewers and
    prevents downstream tools (e.g. integrate_underbody's y=0 cap) from
    finding a single clean polygon to triangulate.

    Strategy:
      1. Collect open boundary edges (edges in exactly one face).
      2. Pick "tooth" edges: exactly one endpoint at y=0 (within 1e-9).
      3. Snap the off-axis endpoint to y=0.
      4. Re-build the mesh with ``process=True`` so any zero-area face
         that resulted from a tip and base coinciding is cleaned out.

    Call right before the final STL export.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64).copy()
    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_pairs = unique[counts == 1]
    y_at_0 = np.abs(v[:, 1]) < 1e-9
    mixed = (y_at_0[boundary_pairs[:, 0]] ^ y_at_0[boundary_pairs[:, 1]])
    teeth = boundary_pairs[mixed]
    if len(teeth) == 0:
        if verbose:
            print("[y0-clean] no tooth edges at y=0; nothing to do")
        return mesh
    off_axis = np.where(
        y_at_0[teeth[:, 0]], teeth[:, 1], teeth[:, 0])
    off_axis = np.unique(off_axis)
    if verbose:
        off_y = np.abs(v[off_axis, 1])
        print(f"[y0-clean] {len(teeth):,} tooth edges, {len(off_axis):,} "
              f"unique off-axis tips (|y| mean={off_y.mean():.5f}, "
              f"max={off_y.max():.5f}); snapping to y=0")
    v[off_axis, 1] = 0.0
    cleaned = trimesh.Trimesh(
        vertices=v, faces=np.asarray(mesh.faces, dtype=np.int64),
        process=True)
    if verbose:
        new_at_0 = int(np.sum(np.abs(cleaned.vertices[:, 1]) < 1e-9))
        print(f"[y0-clean] verts at y=0: {int(np.sum(y_at_0)):,} → "
              f"{new_at_0:,}; faces: {len(mesh.faces):,} → "
              f"{len(cleaned.faces):,}")
    return cleaned


def s2_keep_left(mesh: trimesh.Trimesh,
                 verbose: bool = True
                 ) -> trimesh.Trimesh:
    """Slice the mesh at the y = 0 symmetry plane and keep the y ≤ 0
    half. Faces straddling the plane are split with new vertices
    inserted exactly on y = 0 so the cut edge is a clean planar
    boundary loop. Uses :func:`trimesh.intersections.slice_mesh_plane`
    with ``cap=False`` (the boundary stays open, not filled with a
    planar cap)."""
    cut = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=np.array([0.0, -1.0, 0.0]),
        plane_origin=np.array([0.0, 0.0, 0.0]),
        cap=False,
    )
    cut.face_normals = None
    if verbose:
        n_before = len(mesh.faces)
        n_after = len(cut.faces)
        n_on_plane = int(np.sum(np.abs(cut.vertices[:, 1]) < 1e-8))
        print(f"[s2] sliced at y=0: {n_before:,} -> {n_after:,} faces "
              f"({100.0 * n_after / n_before:.1f}%)")
        print(f"[s2] cropped bounds:\n{cut.bounds}")
        print(f"[s2] vertices on y=0 plane: {n_on_plane:,} "
              f"(clean planar boundary)")
    return cut


# ===========================================================================
# Step 3: wheel (x, y) footprints via z-slice + 1D k-means
# ===========================================================================

@dataclass
class WheelXY:
    """Footprint of one wheel from the ground-z slab."""
    x: float
    y: float
    n_points: int


def s3_locate_wheel_xy(mesh: trimesh.Trimesh,
                       z_band_height: float = 0.02,
                       n_expected: int = 2,
                       verbose: bool = True
                       ) -> list[WheelXY]:
    """Cluster face centroids in a thin z-slice near the ground
    (z ∈ [0, ``z_band_height``]) into ``n_expected`` x-groups using a
    1D Lloyd iteration (k-means). Each cluster centroid in (x, y) is
    one wheel's contact-patch footprint.

    For a left-half mesh expect ``n_expected = 2`` (front-left, rear-
    left). Results are sorted by descending x (most rearward first)."""
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    z = centroids[:, 2]
    mask = (z >= 0.0) & (z <= z_band_height)
    pts = centroids[mask][:, :2]
    if len(pts) < 10 * n_expected:
        raise RuntimeError(f"s3: only {len(pts)} face centroids in the "
                           f"z<={z_band_height} band — not enough to "
                           f"locate {n_expected} wheels.")
    x = pts[:, 0]
    x_min, x_max = float(x.min()), float(x.max())
    centers = np.linspace(x_min, x_max, n_expected)
    assign = np.zeros(len(x), dtype=np.int64)
    for _ in range(50):
        d = np.abs(x[:, None] - centers[None, :])
        assign = np.argmin(d, axis=1)
        new_centers = np.array([
            x[assign == k].mean() if (assign == k).any() else centers[k]
            for k in range(n_expected)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    wheels: list[WheelXY] = []
    for k in range(n_expected):
        cluster = pts[assign == k]
        if len(cluster) == 0:
            continue
        wheels.append(WheelXY(x=float(cluster[:, 0].mean()),
                              y=float(cluster[:, 1].mean()),
                              n_points=int(len(cluster))))
    wheels.sort(key=lambda w: -w.x)
    if verbose:
        print(f"[s3] z-band [0.0, {z_band_height}] -> {len(pts):,} pts "
              f"clustered into {len(wheels)} wheel footprints:")
        for i, w in enumerate(wheels):
            print(f"     wheel {i}: x={w.x:+.4f}  y={w.y:+.4f}  "
                  f"npts={w.n_points}")
    return wheels


# ===========================================================================
# Step 4: wheel hub Z + radius via Y-slab + component-from-contact + circle fit
# ===========================================================================

@dataclass
class WheelCenter:
    """Refined 3D wheel hub + radius + half-width."""
    x: float
    y: float
    z: float
    radius: float
    axial_half: float
    n_section_points: int


def _circle_fit_2d(pts: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit (Kasa, 1976). Returns (cx, cy, r)."""
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx * cx + cy * cy, 1e-12)))
    return float(cx), float(cy), r


def s4_locate_wheel_z(mesh: trimesh.Trimesh,
                      wheels_xy: Sequence[WheelXY],
                      y_slab_half: float = 0.008,
                      contact_z_max: float = 0.010,
                      contact_x_window: float = 0.06,
                      verbose: bool = True
                      ) -> list[WheelCenter]:
    """For each wheel-xy footprint, take a thin Y-plane slab at
    ``y = w.y``, find the mesh-adjacency-connected face component
    inside the slab that contains the ground-contact patch, and fit
    a circle in the X-Z plane to that component's centroids.

    The Y-slab cuts cleanly across the tire's circular silhouette
    in X-Z. The tire is one connected component; wheelhouse walls /
    suspension / brake disc are separate components (their surfaces
    do not share edges with the tire). By restricting the fit to the
    component that actually touches the ground we exclude every
    non-tire structure, no robust trimming required.

    Hub position: ``(cx_fit, w.y, cz_fit)``. Radius:
    ``max(r_fit, cz_fit)`` — the ground-contact constraint that the
    wheel touches z=0 makes the tire OD equal cz.

    Tire half-width: 95th-percentile y-spread of the contact-patch
    faces, clamped to ``[0.20, 0.40] × radius`` for plausibility.
    """
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    out: list[WheelCenter] = []
    for i, w in enumerate(wheels_xy):
        slab_mask = np.abs(centroids[:, 1] - w.y) < y_slab_half
        slab_idx = np.flatnonzero(slab_mask)
        if len(slab_idx) < 50:
            raise RuntimeError(
                f"s4: wheel {i}: only {len(slab_idx)} faces in Y-slab "
                f"(half-thickness {y_slab_half}); slab too thin.")
        in_slab = np.zeros(len(mesh.faces), dtype=bool)
        in_slab[slab_idx] = True
        both = in_slab[adj[:, 0]] & in_slab[adj[:, 1]]
        slab_edges = adj[both]
        remap = -np.ones(len(mesh.faces), dtype=np.int64)
        remap[slab_idx] = np.arange(len(slab_idx))
        remapped_edges = (remap[slab_edges] if len(slab_edges)
                          else np.empty((0, 2), dtype=np.int64))
        comps_local = trimesh.graph.connected_components(
            remapped_edges, nodes=np.arange(len(slab_idx)),
            min_len=1, engine="scipy",
        )
        best_idx_global = None
        best_contact_count = 0
        for comp_loc in comps_local:
            faces_global = slab_idx[comp_loc]
            c = centroids[faces_global]
            contact = ((c[:, 2] < contact_z_max)
                       & (np.abs(c[:, 0] - w.x) < contact_x_window))
            n_contact = int(contact.sum())
            if n_contact > best_contact_count:
                best_contact_count = n_contact
                best_idx_global = faces_global
        if best_idx_global is None or best_contact_count == 0:
            raise RuntimeError(
                f"s4: wheel {i}: no Y-slab component reaches the "
                f"ground at z<{contact_z_max} near x≈{w.x:.3f}.")
        tire_pts = centroids[best_idx_global][:, [0, 2]]
        cx_fit, cz_fit, r_fit = _circle_fit_2d(tire_pts)
        r_geom = float(max(cz_fit, 0.0))
        radius = max(r_fit, r_geom)
        contact_mask = ((centroids[:, 2] < contact_z_max)
                        & (np.abs(centroids[:, 0] - w.x) < contact_x_window)
                        & (np.abs(centroids[:, 1] - w.y) < 0.30 * radius
                                                          + 0.05))
        contact_y = centroids[contact_mask, 1]
        if len(contact_y) >= 10:
            ax_half_raw = float(np.percentile(np.abs(contact_y - w.y), 95))
        else:
            ax_half_raw = 0.40 * radius
        ax_half = float(np.clip(ax_half_raw,
                                0.20 * radius, 0.40 * radius))
        out.append(WheelCenter(x=cx_fit, y=w.y, z=cz_fit,
                               radius=radius, axial_half=ax_half,
                               n_section_points=int(len(best_idx_global))))
        if verbose:
            print(f"[s4] wheel {i}: slab_faces={len(slab_idx):,}  "
                  f"components={len(comps_local)}  "
                  f"tire_comp_size={len(best_idx_global)}  "
                  f"contact_n={best_contact_count}")
            print(f"        center=({cx_fit:+.4f}, {w.y:+.4f}, "
                  f"{cz_fit:+.4f})  radius=max(fit={r_fit:.4f}, "
                  f"ground-z={r_geom:.4f})={radius:.4f}  "
                  f"axial_half={ax_half:.4f}")
    return out


# ===========================================================================
# Step 5: wheel cylinder removal (tight envelope)
# ===========================================================================

def s5_cylinder_remove(mesh: trimesh.Trimesh,
                       wheels: Sequence[WheelCenter],
                       radius_factor: float = 1.00,
                       axial_factor: float = 1.00,
                       verbose: bool = True
                       ) -> np.ndarray:
    """Return a boolean ``remove`` mask. A face is marked REMOVE iff its
    centroid lies inside any wheel's cylinder:
        ``sqrt((x − cx)² + (z − cz)²) ≤ radius_factor × r``
        AND ``|y − cy| ≤ axial_factor × axial_half``
    Cylinder axis is +y (the wheel spin axis); cross-section is a
    circle in the X-Z plane.

    Defaults are 1.00 × 1.00 — an exact tire-hugging fit. The s4 hub
    z is the geometric tire radius (ground-contact constraint) so no
    safety margin is needed; bigger factors would cut into the
    rocker / underbody.
    """
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    remove = np.zeros(len(mesh.faces), dtype=bool)
    for i, w in enumerate(wheels):
        dxz = np.sqrt((centroids[:, 0] - w.x) ** 2 +
                      (centroids[:, 2] - w.z) ** 2)
        dy = np.abs(centroids[:, 1] - w.y)
        inside = (dxz <= radius_factor * w.radius) & \
                 (dy  <= axial_factor * w.axial_half)
        added = int(inside.sum() - (inside & remove).sum())
        remove |= inside
        if verbose:
            print(f"[s5] wheel {i}: inside={int(inside.sum()):,}  "
                  f"new={added:,}  cumulative_remove={int(remove.sum()):,}")
    return remove


# ===========================================================================
# Step 6a: wheel-center ray visibility
# ===========================================================================

def _fibonacci_directions(n: int) -> np.ndarray:
    """``(n, 3)`` quasi-uniform unit vectors on the sphere via the
    Fibonacci lattice."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + np.sqrt(5.0))
    theta = golden * i
    z = np.cos(phi)
    r = np.sin(phi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def _build_intersector(mesh: trimesh.Trimesh):
    """Prefer embree (fast); fall back to rtree."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector  # type: ignore
        return RayMeshIntersector(mesh), "embree"
    except Exception:
        return mesh.ray, "rtree"


def _wheel_origin_set(w: "WheelCenter",
                      n_upper_edge: int = 6,
                      edge_radial_frac: float = 1.0,
                      include_hub: bool = True,
                      ) -> list[np.ndarray]:
    """Origin set used by s6a: hub centre + ``n_upper_edge`` points on
    the upper half of the cylinder's inboard edge. The inboard edge
    is the circle at ``y = w.y + axial_half`` (cabin side); upper
    half means θ ∈ [0, π] so z ≥ w.z."""
    origins: list[np.ndarray] = []
    if include_hub:
        origins.append(np.array([w.x, w.y, w.z], dtype=np.float64))
    if n_upper_edge < 1:
        return origins
    thetas = (np.linspace(0.0, np.pi, n_upper_edge)
              if n_upper_edge > 1 else np.array([np.pi / 2]))
    rp = edge_radial_frac * w.radius
    y_edge = w.y + w.axial_half
    for th in thetas:
        origins.append(np.array([w.x + rp * np.cos(th),
                                 float(y_edge),
                                 w.z + rp * np.sin(th)],
                                dtype=np.float64))
    return origins


def s6a_wheel_center_visibility(mesh: trimesh.Trimesh,
                                wheels: Sequence[WheelCenter],
                                n_rays_per_origin: int = 16384,
                                n_upper_edge: int = 6,
                                edge_radial_frac: float = 1.0,
                                include_hub_origin: bool = True,
                                zone_radius_factor: float = 2.5,
                                zone_axial_factor: float = 2.0,
                                z_max_factor: float = 2.2,
                                exclude_mask: np.ndarray | None = None,
                                verbose: bool = True
                                ) -> np.ndarray:
    """Cast Fibonacci-sphere rays from the hub centre plus
    ``n_upper_edge`` points on the upper half of the cylinder's
    inboard edge. Accumulate all first-hit faces across all origins,
    apply zone + z-cap filters, then return one boolean ``remove``
    mask.

    With the s5 cylinder excluded from the intersector, rays travel
    through the empty cylinder region and first-hit the wheelhouse-
    interior walls / suspension on the other side. The inboard-edge
    origins (which sit on the cabin side of the cylinder) have a
    clear line of sight to the inboard wheelhouse wall and inboard
    suspension — surfaces a central hub origin cannot see because
    the brake disc / wheel spokes occlude the view in that direction.

    Filters:
        * **Zone**: hit accepted only if
          ``sqrt((x-cx)²+(z-cz)²) < zone_radius_factor × r`` AND
          ``|y − cy| < zone_axial_factor × axial_half``.
        * **Z-cap**: hit accepted only if ``z ≤ z_max_factor × hub_z``.
          Default 2.2 × hub_z protects the door panel / fender outer
          skin above the wheel arch from upward-escaping rays.

    Removal is **one-step**: all rays are cast and all per-wheel hits
    accumulated into a single mask before any face is marked REMOVE.
    The result is independent of origin ordering, and (since the
    intersector is built once) there is no within-wheel coupling
    where one origin's rays would be blocked by faces another origin
    marked REMOVE.
    """
    base_dirs = _fibonacci_directions(n_rays_per_origin)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    in_any_zone = np.zeros(len(mesh.faces), dtype=bool)
    for w in wheels:
        dxz = np.sqrt((centroids[:, 0] - w.x) ** 2 +
                      (centroids[:, 2] - w.z) ** 2)
        dy = np.abs(centroids[:, 1] - w.y)
        in_any_zone |= (dxz < zone_radius_factor * w.radius) & \
                       (dy < zone_axial_factor * w.axial_half)
    z_cap = (z_max_factor * max(w.z for w in wheels)) if wheels else np.inf
    z_cap_mask = centroids[:, 2] <= z_cap

    if exclude_mask is None:
        intersect_mesh = mesh
        local_to_global = np.arange(len(mesh.faces), dtype=np.int64)
    else:
        keep_idx = np.flatnonzero(~exclude_mask)
        sub = mesh.submesh([keep_idx], append=True)
        if isinstance(sub, list):
            sub = sub[0]
        intersect_mesh = sub
        local_to_global = keep_idx.astype(np.int64)
    intersector, backend = _build_intersector(intersect_mesh)
    n_origins_per_wheel = int(include_hub_origin) + n_upper_edge
    if verbose:
        excl = (int(exclude_mask.sum()) if exclude_mask is not None
                else 0)
        print(f"[s6a] {len(wheels)} wheel(s); zone {zone_radius_factor}× r "
              f"× {zone_axial_factor}× ax_half; z-cap z ≤ {z_cap:.4f} "
              f"({z_max_factor}× hub_z); intersector excludes "
              f"{excl:,} faces ({len(intersect_mesh.faces):,} in submesh); "
              f"{n_origins_per_wheel} origins/wheel × "
              f"{n_rays_per_origin} rays = "
              f"{n_origins_per_wheel * n_rays_per_origin:,} rays/wheel")

    hits_collected = np.zeros(len(mesh.faces), dtype=bool)
    for i, w in enumerate(wheels):
        origins_pts = _wheel_origin_set(
            w, n_upper_edge=n_upper_edge,
            edge_radial_frac=edge_radial_frac,
            include_hub=include_hub_origin,
        )
        wheel_raw_hits = wheel_escapes = 0
        wheel_mask = np.zeros(len(mesh.faces), dtype=bool)
        for op in origins_pts:
            origins = np.tile(op, (n_rays_per_origin, 1))
            local_hits = intersector.intersects_first(origins, base_dirs)
            valid = local_hits >= 0
            global_hits = local_to_global[local_hits[valid]]
            wheel_mask[global_hits] = True
            wheel_raw_hits += int(valid.sum())
            wheel_escapes += int((~valid).sum())
        wheel_mask &= in_any_zone & z_cap_mask
        wheel_kept = int(wheel_mask.sum())
        hits_collected |= wheel_mask
        if verbose:
            print(f"[s6a] wheel {i}: origins={len(origins_pts)}  "
                  f"raw_rays={n_origins_per_wheel * n_rays_per_origin:,}  "
                  f"raw_hit_faces={wheel_raw_hits:,}  "
                  f"escapes={wheel_escapes:,}  "
                  f"kept_after_filters={wheel_kept:,}")
    if verbose:
        print(f"[s6a] one-step REMOVE = {int(hits_collected.sum()):,} faces")
    return hits_collected


# ===========================================================================
# Step 6b: wheelhouse-zone geometric backstop
# ===========================================================================

def s6b_wheelhouse_zone_inward(mesh: trimesh.Trimesh,
                                wheels: Sequence[WheelCenter],
                                radius_factor: float = 1.4,
                                axial_factor: float = 1.7,
                                inward_cos_threshold: float = 0.3,
                                extend_inboard_to_y: float | None = 0.0,
                                verbose: bool = True
                                ) -> np.ndarray:
    """Mark every body face whose centroid lies inside the cylindrical
    wheelhouse zone of any wheel AND whose outward normal points
    sufficiently toward the hub (cos ≥ ``inward_cos_threshold``).

    Geometric backstop for s6a: ray casting can miss wheelhouse-
    interior surfaces occluded from every hub origin. Such surfaces
    are still inward-facing (their normals point at the hub — they
    ARE the inside skin of the wheel arch) and inside the wheelhouse
    zone, so this purely geometric test catches them.

    With ``extend_inboard_to_y`` set (default 0.0), the zone is
    asymmetric: outboard side keeps the symmetric
    ``axial_factor × axial_half`` extent; inboard side extends to
    ``extend_inboard_to_y`` (the symmetry plane). This covers any
    inboard wheelhouse wall + inboard suspension at any depth.
    """
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    for i, w in enumerate(wheels):
        d_xz = np.sqrt((centroids[:, 0] - w.x) ** 2 +
                       (centroids[:, 2] - w.z) ** 2)
        radial_ok = d_xz < radius_factor * w.radius
        if extend_inboard_to_y is None:
            y_ok = np.abs(centroids[:, 1] - w.y) < axial_factor * w.axial_half
            y_lo = w.y - axial_factor * w.axial_half
            y_hi = w.y + axial_factor * w.axial_half
        else:
            half = axial_factor * w.axial_half
            if w.y < extend_inboard_to_y:
                y_lo = w.y - half
                y_hi = extend_inboard_to_y
            else:
                y_lo = extend_inboard_to_y
                y_hi = w.y + half
            y_ok = (centroids[:, 1] >= y_lo) & (centroids[:, 1] <= y_hi)
        in_zone = radial_ok & y_ok
        hub = np.array([w.x, w.y, w.z], dtype=np.float64)
        vec_to_hub = hub[None, :] - centroids
        norm = np.linalg.norm(vec_to_hub, axis=1) + 1e-12
        cos_in = (normals * vec_to_hub / norm[:, None]).sum(axis=1)
        inward = in_zone & (cos_in > inward_cos_threshold)
        added = int((inward & ~mask).sum())
        mask |= inward
        if verbose:
            print(f"[s6b] wheel {i}: zone y ∈ [{y_lo:+.3f}, {y_hi:+.3f}]  "
                  f"in-zone={int(in_zone.sum()):,}  "
                  f"inward={int(inward.sum()):,}  new={added:,}  "
                  f"cumulative={int(mask.sum()):,}")
    return mask


# ===========================================================================
# Step 7: visibility classifier (with cut zone)
# ===========================================================================

def _wheel_cyl_y_bounds(w: "WheelCenter",
                         axial_factor: float,
                         outboard_y_limit: float | None,
                         ) -> tuple[float, float]:
    """Per-wheel (y_lo, y_hi) for the cut-zone cylinder. Symmetric
    if ``outboard_y_limit`` is None; otherwise the outboard end is
    pulled out to ``outboard_y_limit``."""
    half = axial_factor * w.axial_half
    if outboard_y_limit is None:
        return w.y - half, w.y + half
    if w.y < 0:
        return outboard_y_limit, w.y + half
    else:
        return w.y - half, outboard_y_limit


def build_cut_zone(mesh: trimesh.Trimesh,
                   wheels: Sequence["WheelCenter"],
                   floor_z_max: float,
                   wheel_radius_factor: float = 1.3,
                   wheel_axial_factor: float = 1.3,
                   outboard_y_limit: float | None = None,
                   verbose: bool = True
                   ) -> np.ndarray:
    """Return a boolean face-mask: True = the face is *eligible* to be
    marked REMOVE by step 7. Outside-zone faces are forced to KEEP.

    Cut zone = (floor band: z < ``floor_z_max``) ∪ (wheel cylinders,
    ``wheel_radius_factor × r`` in XZ, asymmetric y-bounds per
    :func:`_wheel_cyl_y_bounds`).
    """
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    cut = np.zeros(len(mesh.faces), dtype=bool)
    in_floor = centroids[:, 2] < floor_z_max
    cut |= in_floor
    in_wheel_total = np.zeros(len(mesh.faces), dtype=bool)
    for w in wheels:
        d_xz = np.sqrt((centroids[:, 0] - w.x) ** 2 +
                       (centroids[:, 2] - w.z) ** 2)
        radial_ok = d_xz < wheel_radius_factor * w.radius
        y_lo, y_hi = _wheel_cyl_y_bounds(w, wheel_axial_factor,
                                          outboard_y_limit)
        y_ok = (centroids[:, 1] >= y_lo) & (centroids[:, 1] <= y_hi)
        in_wheel = radial_ok & y_ok
        in_wheel_total |= in_wheel
        cut |= in_wheel
    if verbose:
        outboard_str = (f", outboard end → y={outboard_y_limit:+.4f}"
                        if outboard_y_limit is not None else "")
        print(f"[cut-zone] floor (z < {floor_z_max:.4f}): "
              f"{int(in_floor.sum()):,} faces")
        print(f"[cut-zone] wheel cylinders ({wheel_radius_factor}× r, "
              f"inboard {wheel_axial_factor}× ax_half{outboard_str}) "
              f"× {len(wheels)}: {int(in_wheel_total.sum()):,} faces")
        print(f"[cut-zone] total in zone: {int(cut.sum()):,} faces "
              f"({100 * cut.mean():.1f} % of {len(mesh.faces):,})")
    return cut


def build_cut_zone_volume_mesh(wheels: Sequence["WheelCenter"],
                                floor_z_max: float,
                                floor_x_range: tuple[float, float],
                                floor_y_range: tuple[float, float],
                                wheel_radius_factor: float = 1.3,
                                wheel_axial_factor: float = 1.3,
                                outboard_y_limit: float | None = None,
                                sections: int = 48,
                                ) -> trimesh.Trimesh:
    """Build a debug STL showing the cut-zone volume: floor box +
    wheel cylinders. Overlay against the body STL in MeshLab /
    Blender to confirm the cut region."""
    parts: list[trimesh.Trimesh] = []
    box_extents = (
        floor_x_range[1] - floor_x_range[0],
        floor_y_range[1] - floor_y_range[0],
        floor_z_max,
    )
    box_center = (
        0.5 * (floor_x_range[0] + floor_x_range[1]),
        0.5 * (floor_y_range[0] + floor_y_range[1]),
        0.5 * floor_z_max,
    )
    box = trimesh.creation.box(extents=box_extents)
    box.apply_translation(box_center)
    parts.append(box)
    for w in wheels:
        y_lo, y_hi = _wheel_cyl_y_bounds(w, wheel_axial_factor,
                                          outboard_y_limit)
        cyl_length = y_hi - y_lo
        cyl_center_y = 0.5 * (y_lo + y_hi)
        cyl = trimesh.creation.cylinder(
            radius=wheel_radius_factor * w.radius,
            height=cyl_length,
            sections=sections,
        )
        R = trimesh.transformations.rotation_matrix(np.pi / 2.0,
                                                    [1.0, 0.0, 0.0])
        cyl.apply_transform(R)
        cyl.apply_translation([w.x, cyl_center_y, w.z])
        parts.append(cyl)
    return trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()


def s7_classify_visibility(mesh: trimesh.Trimesh,
                            n_dirs: int = 96,
                            equator_band_deg: float = 15.0,
                            min_outward_cos: float = 0.05,
                            eps_rel: float = 1e-5,
                            prefilter: bool = True,
                            prefilter_z_band_frac: float = 0.31,
                            prefilter_normal_z_max: float = -0.30,
                            cut_zone: np.ndarray | None = None,
                            batch_size: int = 2_000_000,
                            verbose: bool = True,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify each face as KEEP / REMOVE / DROP via two-hemisphere
    visibility ray casting.

    For each face, cast one ray from its (eps-nudged) centroid toward
    every Fibonacci-sphere direction satisfying:
        * elevation outside the ±``equator_band_deg`` band
          (prevents near-horizontal rays sneaking under the rocker
          through the ground-clearance gap)
        * direction on the triangle's outward side
          (``normal · direction > min_outward_cos``)

    A ray that hits no other triangle proves the face is visible
    from "+∞·d". Final labels (KEEP > REMOVE > DROP precedence):
        KEEP   = any escape with elevation > +equator_band_deg
        REMOVE = no upper escape but some lower escape
        DROP   = neither

    A cheap geometric prefilter pre-marks "obvious underbody" faces
    (bottom ``prefilter_z_band_frac`` of z-envelope AND downward
    normal) so they skip the lower-pass casting.

    If ``cut_zone`` is given, any face whose cut_zone flag is False
    is *forced* to LABEL_KEEP — preventing the classifier from
    putting holes in distant body parts (bumper intake, mirror
    underside, vents).

    Returns ``(labels, upper_visible, lower_visible)``.
    """
    n_faces = len(mesh.faces)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    eps = eps_rel * diag
    origins_all = centroids + eps * normals

    upper_visible = np.zeros(n_faces, dtype=bool)
    lower_visible = np.zeros(n_faces, dtype=bool)

    if prefilter:
        z_lo, z_hi = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
        z_thresh = z_lo + prefilter_z_band_frac * (z_hi - z_lo)
        pre_mask = ((centroids[:, 2] < z_thresh) &
                    (normals[:, 2] < prefilter_normal_z_max))
        lower_visible |= pre_mask
        if verbose:
            print(f"[s7] prefilter: z < {z_thresh:.4f} "
                  f"({prefilter_z_band_frac:.0%} of envelope) AND "
                  f"normal_z < {prefilter_normal_z_max:+.2f} -> "
                  f"{int(pre_mask.sum()):,} faces pre-marked lower_visible")

    intersector, backend = _build_intersector(mesh)
    if verbose:
        print(f"[s7] intersector backend = {backend}  "
              f"({n_faces:,} faces, eps={eps:.4g})")

    all_dirs = _fibonacci_directions(n_dirs)
    elev_deg = np.degrees(np.arcsin(all_dirs[:, 2]))
    upper_idx = np.flatnonzero(elev_deg >= equator_band_deg)
    lower_idx = np.flatnonzero(elev_deg <= -equator_band_deg)
    n_eq_skipped = n_dirs - len(upper_idx) - len(lower_idx)
    if verbose:
        print(f"[s7] {n_dirs} Fibonacci dirs: upper={len(upper_idx)}, "
              f"lower={len(lower_idx)}, equator-band skipped="
              f"{n_eq_skipped} (±{equator_band_deg}°)")

    def _shoot(active_mask, d):
        active_idx = np.flatnonzero(active_mask)
        n = len(active_idx)
        if n == 0:
            return active_idx, np.zeros(0, dtype=bool)
        origs = origins_all[active_idx]
        dirs_b = np.broadcast_to(d, origs.shape).astype(np.float64)
        hits = np.empty(n, dtype=np.int64)
        for b0 in range(0, n, batch_size):
            b1 = min(n, b0 + batch_size)
            hits[b0:b1] = intersector.intersects_first(
                origs[b0:b1], dirs_b[b0:b1])
        return active_idx, (hits < 0)

    upper_rays = upper_escapes = 0
    for di in upper_idx:
        d = all_dirs[di]
        cos_nd = normals @ d
        active = (cos_nd > min_outward_cos) & (~upper_visible)
        if not active.any():
            continue
        ai, esc = _shoot(active, d)
        upper_visible[ai[esc]] = True
        upper_rays += int(active.sum())
        upper_escapes += int(esc.sum())
    if verbose:
        print(f"[s7] UPPER pass: rays={upper_rays:,}  "
              f"escapes={upper_escapes:,}  "
              f"upper_visible={int(upper_visible.sum()):,}")

    lower_rays = lower_escapes = 0
    for di in lower_idx:
        d = all_dirs[di]
        cos_nd = normals @ d
        active = (cos_nd > min_outward_cos) & (~lower_visible)
        if not active.any():
            continue
        ai, esc = _shoot(active, d)
        lower_visible[ai[esc]] = True
        lower_rays += int(active.sum())
        lower_escapes += int(esc.sum())
    if verbose:
        print(f"[s7] LOWER pass: rays={lower_rays:,}  "
              f"escapes={lower_escapes:,}  "
              f"lower_visible={int(lower_visible.sum()):,}")

    labels = np.full(n_faces, LABEL_DROP, dtype=np.int8)
    labels[lower_visible] = LABEL_REMOVE
    labels[upper_visible] = LABEL_KEEP
    if cut_zone is not None:
        n_forced = int(((~cut_zone) & (labels != LABEL_KEEP)).sum())
        labels[~cut_zone] = LABEL_KEEP
        if verbose:
            print(f"[s7] cut-zone constraint: "
                  f"{int((~cut_zone).sum()):,} faces outside zone forced to KEEP "
                  f"({n_forced:,} re-classified from REMOVE/DROP)")
    if verbose:
        nk, nr, nd = (int((labels == L).sum()) for L in
                      (LABEL_KEEP, LABEL_REMOVE, LABEL_DROP))
        print(f"[s7] final: KEEP={nk:,} ({100*nk/n_faces:.1f}%)  "
              f"REMOVE={nr:,} ({100*nr/n_faces:.1f}%)  "
              f"DROP={nd:,} ({100*nd/n_faces:.1f}%)")
    return labels, upper_visible, lower_visible


# ===========================================================================
# Helpers — boundary edges, tube viz, debris cleanup
# ===========================================================================

def trace_boundary_edges(mesh: trimesh.Trimesh,
                          mask: np.ndarray,
                          verbose: bool = True
                          ) -> np.ndarray:
    """Return (N, 2) vertex-index pairs for edges shared by exactly one
    masked face and one non-masked face — i.e. the boundary of the
    ``mask`` region. These form closed loops on a closed manifold."""
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adj_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    on_boundary = mask[adj[:, 0]] ^ mask[adj[:, 1]]
    edges = adj_edges[on_boundary]
    if verbose:
        print(f"[boundary] {len(edges):,} edges (mask ↔ ¬mask)")
    return edges


def edges_to_tube_mesh(vertices: np.ndarray,
                        edges: np.ndarray,
                        radius: float = 0.001,
                        sections: int = 4
                        ) -> trimesh.Trimesh:
    """Visualize a set of edges as a STL-friendly mesh by extruding a
    thin cylinder along each (v0, v1) edge."""
    if len(edges) == 0:
        return trimesh.Trimesh()
    parts: list[trimesh.Trimesh] = []
    z_axis = np.array([0.0, 0.0, 1.0])
    for e in edges:
        p0 = vertices[e[0]]
        p1 = vertices[e[1]]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=L,
                                        sections=sections)
        d_hat = d / L
        R = trimesh.geometry.align_vectors(z_axis, d_hat)
        if R is not None:
            cyl.apply_transform(R)
        cyl.apply_translation(0.5 * (p0 + p1))
        parts.append(cyl)
    return trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()


def keep_largest_component(mesh: trimesh.Trimesh,
                           remove_mask: np.ndarray,
                           protect_mask: np.ndarray | None = None,
                           verbose: bool = True
                           ) -> np.ndarray:
    """After applying ``remove_mask``, find the connected components
    of the kept faces (via the mesh's face-adjacency graph restricted
    to KEEP-KEEP edges) and mark all faces NOT in the largest
    component as REMOVE.

    ``protect_mask`` (optional): faces in this mask are exempt from
    being dropped even if they sit in a small component. Useful to
    preserve clean planar symmetry-edge fragments.
    """
    kept_mask = ~remove_mask
    kept_idx = np.flatnonzero(kept_mask)
    if len(kept_idx) == 0:
        return remove_mask.copy()
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    both_kept = kept_mask[adj[:, 0]] & kept_mask[adj[:, 1]]
    kept_edges = adj[both_kept]
    remap = -np.ones(len(mesh.faces), dtype=np.int64)
    remap[kept_idx] = np.arange(len(kept_idx))
    remapped_edges = (remap[kept_edges] if len(kept_edges)
                       else np.empty((0, 2), dtype=np.int64))
    comps = trimesh.graph.connected_components(
        remapped_edges, nodes=np.arange(len(kept_idx)),
        min_len=1, engine="scipy",
    )
    if len(comps) <= 1:
        if verbose:
            print(f"[keep-largest] 1 component, no debris")
        return remove_mask.copy()
    sizes = np.array([len(c) for c in comps], dtype=np.int64)
    largest = int(np.argmax(sizes))
    new_remove = remove_mask.copy()
    debris_total = protected_total = 0
    for i, comp in enumerate(comps):
        if i == largest:
            continue
        comp_face_ids = kept_idx[comp]
        if protect_mask is not None:
            keep_these = protect_mask[comp_face_ids]
            drop_these = comp_face_ids[~keep_these]
            protected_total += int(keep_these.sum())
        else:
            drop_these = comp_face_ids
        new_remove[drop_these] = True
        debris_total += len(drop_these)
    if verbose:
        top = sorted(sizes.tolist(), reverse=True)[:6]
        msg = (f"[keep-largest] {len(comps)} kept-components, "
               f"top-6={top}, dropped {debris_total:,} debris faces")
        if protect_mask is not None:
            msg += f" (kept {protected_total:,} protected faces)"
        print(msg)
    return new_remove


def split_keep_remove(mesh: trimesh.Trimesh,
                      remove: np.ndarray
                      ) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Return ``(kept_submesh, removed_submesh)`` from a per-face mask."""
    keep_idx = np.flatnonzero(~remove)
    rm_idx = np.flatnonzero(remove)
    kept = mesh.submesh([keep_idx], append=True) if len(keep_idx) else None
    rm = mesh.submesh([rm_idx], append=True) if len(rm_idx) else None
    if isinstance(kept, list):
        kept = kept[0]
    if isinstance(rm, list):
        rm = rm[0]
    return kept, rm
