"""Watertight body via voxel flood-fill + projection back onto the input.

Two-stage pipeline:

  Stage A — voxel pipeline (close_body_watertight_voxel.py logic)
    Rasterize input triangles to a 3-D voxel grid, morphologically close,
    flood-fill from the outside, marching cubes on the solid mask. Output
    is a watertight mesh with the right topology (wheel cavities preserved
    as concavities, y=0 cap closed) but it has lost all of the input's
    surface detail to the voxel staircase.

  Stage B — exact closest-point projection
    For every vertex of the voxel mesh, find the exact closest point on
    the input mesh (libigl's point_mesh_squared_distance). If within
    ``--max-proj`` mm, snap the vertex to it. Voxel topology is left
    unchanged so the mesh stays watertight, while the surface geometry
    now follows the original input wherever input has triangles.

In the "true holes" (concatenation seams the input never sealed, e.g.
between wheelhouse dimples and side caps), no input is within reach
and those vertices stay where the voxel pipeline put them — a smooth
local cap. Everywhere else, the input's panel lines and cavity ribs
come back at sub-voxel accuracy.

Trade-off
---------
Vertex density is set by the voxel pitch (~6 mm). Features finer than
that — e.g. a 2 mm raised character line — get softened to the nearest
6 mm vertex spacing. This is the cost of guaranteeing watertightness.

Usage::

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python close_body_watertight_hybrid.py \\
        --input  outputs/integrate/<stem>_combined_body.stl \\
        --output outputs/integrate/<stem>_watertight.stl \\
        --pitch 6 --close-iters 10 --max-proj 12
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Stage A: voxel pipeline (identical to close_body_watertight_voxel.py).
# ---------------------------------------------------------------------------

def voxel_watertight(input_path: Path, pitch: float, close_iters: int,
                      pad: float, samples_per_voxel: float,
                      smooth_sigma: float):
    """Return a watertight trimesh.Trimesh built from a voxel flood-fill of
    the input. Mesh is topologically correct but loses input detail."""
    import trimesh
    import scipy.ndimage as ndi
    from skimage import measure

    t0 = time.time()
    m = trimesh.load(str(input_path), force="mesh", process=False)
    print(f"[voxel] input {len(m.faces):,} tris ({time.time()-t0:.1f}s)")
    print(f"[voxel] bounds: {m.bounds[0]} -> {m.bounds[1]}")

    bmin = m.bounds[0] - pad
    bmax = m.bounds[1] + pad
    size = bmax - bmin
    nx, ny, nz = (np.ceil(size / pitch).astype(int) + 1).tolist()
    print(f"[voxel] grid {nx} x {ny} x {nz} = {nx*ny*nz:,} voxels @ {pitch}mm")

    n_samples = max(int(m.area / (pitch * pitch) * samples_per_voxel),
                    200_000)
    pts, _ = trimesh.sample.sample_surface(m, n_samples)
    pts = np.vstack([pts, np.asarray(m.vertices)])
    idx = np.floor((pts - bmin) / pitch).astype(np.int64)
    np.clip(idx[:, 0], 0, nx - 1, out=idx[:, 0])
    np.clip(idx[:, 1], 0, ny - 1, out=idx[:, 1])
    np.clip(idx[:, 2], 0, nz - 1, out=idx[:, 2])
    surface = np.zeros((nx, ny, nz), dtype=bool)
    surface[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    print(f"[voxel] surface voxels: {surface.sum():,}")

    # Seal y=0 cap.
    y0_idx = int(np.floor((0.0 - bmin[1]) / pitch))
    if 0 <= y0_idx < ny:
        footprint = surface.any(axis=1)
        surface[:, y0_idx, :] |= footprint
        if y0_idx + 1 < ny:
            surface[:, y0_idx + 1, :] |= footprint

    struct = ndi.generate_binary_structure(3, 1)
    closed = ndi.binary_closing(surface, structure=struct,
                                  iterations=close_iters)
    seed = np.zeros_like(closed)
    for corner in [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
                    (-1, -1, 0), (-1, 0, -1), (0, -1, -1), (-1, -1, -1)]:
        seed[corner] = True
    outside = ndi.binary_dilation(seed, structure=struct,
                                    mask=~closed, iterations=-1)
    solid = ~outside
    print(f"[voxel] solid={solid.sum():,} ({solid.sum()*pitch**3/1e9:.2f} m^3)")

    padded = np.pad(solid.astype(np.uint8), 2, mode="constant",
                     constant_values=0)
    if smooth_sigma > 0:
        padded = ndi.gaussian_filter(padded.astype(np.float32),
                                       sigma=smooth_sigma)
        level = 0.5
    else:
        level = 0
    verts, faces, _, _ = measure.marching_cubes(
        padded, level=level, spacing=(pitch, pitch, pitch))
    verts -= 2 * pitch
    verts += bmin
    print(f"[voxel] MC {len(faces):,} tris")

    mv = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    parts = mv.split(only_watertight=False)
    parts.sort(key=lambda p: len(p.faces), reverse=True)
    main = parts[0]
    try:
        if main.volume < 0:
            main.invert()
    except Exception as exc:
        print(f"[voxel] orient skipped: {exc}")
    print(f"[voxel] main component {len(main.faces):,} tris, "
          f"watertight={main.is_watertight}, "
          f"vol={main.volume/1e9:.3f} m^3")
    return main, m


# ---------------------------------------------------------------------------
# Stage B: project voxel-mesh vertices onto the input surface.
# ---------------------------------------------------------------------------

def project_onto_input(voxel_mesh, input_mesh, max_proj: float):
    """Snap each vertex of ``voxel_mesh`` to the closest point on
    ``input_mesh`` if within ``max_proj`` mm. Topology is preserved so
    the mesh stays watertight."""
    import igl
    import trimesh

    V_in = np.asarray(input_mesh.vertices, dtype=np.float64)
    F_in = np.asarray(input_mesh.faces, dtype=np.int32)
    Q = np.asarray(voxel_mesh.vertices, dtype=np.float64)

    t0 = time.time()
    sqd, _, closest = igl.point_mesh_squared_distance(Q, V_in, F_in)
    dist = np.sqrt(sqd)
    print(f"[project] {len(Q):,} closest-point queries "
          f"({time.time()-t0:.1f}s)")
    print(f"[project] dist median={np.median(dist):.2f}mm "
          f"mean={dist.mean():.2f}mm max={dist.max():.2f}mm")

    mask = dist < max_proj
    print(f"[project] snapping {mask.sum():,}/{len(Q):,} verts "
          f"within {max_proj}mm of input")

    v = voxel_mesh.vertices.copy()
    v[mask] = closest[mask]
    out = trimesh.Trimesh(vertices=v, faces=voxel_mesh.faces.copy(),
                            process=False)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pitch", type=float, default=6.0,
                        help="voxel pitch in mm (smaller=more detail, "
                             "more memory)")
    parser.add_argument("--close-iters", type=int, default=10,
                        help="morphological closing iters (keep small "
                             "enough that wheel arches stay open)")
    parser.add_argument("--pad", type=float, default=72.0,
                        help="bbox padding in mm; must exceed "
                             "close_iters * pitch")
    parser.add_argument("--samples-per-voxel", type=float, default=16.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Gaussian sigma in voxels before MC")
    parser.add_argument("--max-proj", type=float, default=None,
                        help="max projection distance in mm. Default = "
                             "2 * pitch.")
    args = parser.parse_args()
    if args.max_proj is None:
        args.max_proj = 2.0 * args.pitch
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage A: voxel watertight base ===")
    voxel_mesh, input_mesh = voxel_watertight(
        args.input, args.pitch, args.close_iters, args.pad,
        args.samples_per_voxel, args.smooth_sigma)

    print(f"\n=== Stage B: project onto input ===")
    out = project_onto_input(voxel_mesh, input_mesh, args.max_proj)

    out.export(str(args.output), file_type="stl")
    print(f"\n[out] {args.output}")
    print(f"  faces:      {len(out.faces):,}")
    print(f"  vertices:   {len(out.vertices):,}")
    print(f"  watertight: {out.is_watertight}")
    try:
        print(f"  volume:     {out.volume/1e9:.3f} m^3")
    except Exception as exc:
        print(f"  volume:     failed ({exc})")
    print(f"  bounds:     {out.bounds[0]} -> {out.bounds[1]}")


if __name__ == "__main__":
    main()
