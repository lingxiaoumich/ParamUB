"""Voxel + flood-fill pipeline for making ``combined_body.stl`` watertight.

Why this approach
-----------------
Pipelines that depend on consistent normal orientation (Poisson, generalized
winding number) struggle on our concatenated body STL because parts have
inconsistent normals and the wheelhouse cavities are open at the wheel arch.
A purely geometric approach sidesteps that:

  1. Rasterize all triangles into a 3-D binary voxel grid (the *surface mask*).
  2. Morphological closing with N iterations to bridge sub-N-voxel gaps
     (set N small enough that wheel arch openings remain open).
  3. Flood-fill from the bbox corner. Everything reachable is *outside*.
     Wheelhouse cavities reach through the arch opening, so they stay outside.
  4. ``solid = NOT outside``.  Marching cubes at level 0.5 over ``solid``.
  5. Largest connected component, fix orientation, write STL.

Knobs
-----
  --pitch         voxel size in mm (8 = ~70M voxels at 4.5x1.0x1.3 m bbox)
  --close-iters   morphological closing radius (in voxels). 2-4 typical.
  --pad           bbox padding (mm). Must be > closing radius * pitch.
  --samples-per-voxel  surface oversampling factor.

Requires::

    pip install trimesh scikit-image scipy mapbox_earcut
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def voxelize_surface(mesh, bmin: np.ndarray, pitch: float,
                     shape: tuple[int, int, int],
                     samples_per_voxel: float = 6.0) -> np.ndarray:
    """Dense surface sampling -> binary voxel grid."""
    import trimesh

    n_samples = max(int(mesh.area / (pitch * pitch) * samples_per_voxel),
                    200_000)
    print(f"[voxelize] sampling {n_samples:,} surface points "
          f"({samples_per_voxel:.1f}x per voxel face)")
    t0 = time.time()
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    print(f"[voxelize] sampled in {time.time() - t0:.1f}s")

    # Include the mesh vertices themselves so thin features don't get missed.
    pts = np.vstack([pts, np.asarray(mesh.vertices)])

    t0 = time.time()
    idx = np.floor((pts - bmin) / pitch).astype(np.int64)
    np.clip(idx[:, 0], 0, shape[0] - 1, out=idx[:, 0])
    np.clip(idx[:, 1], 0, shape[1] - 1, out=idx[:, 1])
    np.clip(idx[:, 2], 0, shape[2] - 1, out=idx[:, 2])
    surface = np.zeros(shape, dtype=bool)
    surface[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    print(f"[voxelize] filled {surface.sum():,} voxels "
          f"({time.time() - t0:.1f}s)")
    return surface


def close_and_flood(surface: np.ndarray, close_iters: int) -> np.ndarray:
    """Morphological close + flood-fill outside.  Returns solid mask."""
    import scipy.ndimage as ndi

    t0 = time.time()
    struct = ndi.generate_binary_structure(3, 1)  # 6-connected
    closed = ndi.binary_closing(surface, structure=struct,
                                 iterations=close_iters)
    print(f"[close] {closed.sum():,} voxels after closing(iters={close_iters}) "
          f"({time.time() - t0:.1f}s)")

    # Flood from corner. We mask on ~closed so flood stops at the closed shell.
    t0 = time.time()
    seed = np.zeros_like(closed)
    seed[0, 0, 0] = True
    seed[-1, 0, 0] = True
    seed[0, -1, 0] = True
    seed[0, 0, -1] = True
    seed[-1, -1, 0] = True
    seed[-1, 0, -1] = True
    seed[0, -1, -1] = True
    seed[-1, -1, -1] = True
    outside = ndi.binary_dilation(seed, structure=struct,
                                   mask=~closed, iterations=-1)
    print(f"[flood] outside = {outside.sum():,} voxels "
          f"({time.time() - t0:.1f}s)")

    solid = ~outside
    print(f"[solid] {solid.sum():,} voxels "
          f"(closed wall + true interior + cavities-not-connected-to-outside)")
    return solid


def marching_cubes(solid: np.ndarray, bmin: np.ndarray, pitch: float,
                   smooth_sigma: float = 1.0):
    """skimage marching_cubes with a zero-padded volume so the iso-surface
    closes at the grid boundary.

    If smooth_sigma > 0, Gaussian-smooths the indicator field in-place
    before MC, which removes the voxel staircase at the cost of slightly
    rounding sharp edges (we want this for a car body). Uses uint8 input
    -> float32 only inside the smoothing call to stay under memory."""
    from skimage import measure
    import trimesh

    t0 = time.time()
    padded_u8 = np.pad(solid.astype(np.uint8),
                        pad_width=2, mode="constant", constant_values=0)
    if smooth_sigma > 0:
        import scipy.ndimage as ndi
        # gaussian_filter accepts integer input and returns same dtype if
        # output isn't specified. We promote to float32 just for MC.
        smoothed = ndi.gaussian_filter(padded_u8.astype(np.float32),
                                        sigma=smooth_sigma)
        del padded_u8
        print(f"[mc] gaussian sigma={smooth_sigma} applied "
              f"({time.time() - t0:.1f}s)")
        t0 = time.time()
        field = smoothed
        level = 0.5
    else:
        field = padded_u8
        level = 0
    verts, faces, _, _ = measure.marching_cubes(
        field, level=level,
        spacing=(pitch, pitch, pitch))
    del field
    # Padding moved origin by 2 voxels in each dim
    verts -= 2 * pitch
    verts += bmin
    print(f"[mc] {len(faces):,} tris, {len(verts):,} verts "
          f"({time.time() - t0:.1f}s)")
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def keep_largest_component(mesh, fix_orient: bool = True):
    import trimesh
    parts = mesh.split(only_watertight=False)
    parts.sort(key=lambda p: len(p.faces), reverse=True)
    print(f"[components] {len(parts)} (top: "
          f"{[len(p.faces) for p in parts[:5]]})")
    main = parts[0]
    if fix_orient:
        try:
            if main.volume < 0:
                main.invert()
        except Exception as exc:
            print(f"[orient] skipped: {exc}")
    return main


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pitch", type=float, default=8.0,
                        help="voxel size, mm")
    parser.add_argument("--close-iters", type=int, default=8,
                        help="morphological closing iterations (voxels). "
                             "Each iter bridges a ~pitch-mm gap. Keep small "
                             "enough that wheel arch openings stay open "
                             "(arches are ~300 mm, closing bridges <2*N*pitch).")
    parser.add_argument("--pad", type=float, default=80.0,
                        help="bbox padding in mm. Must exceed "
                             "close_iters * pitch.")
    parser.add_argument("--samples-per-voxel", type=float, default=20.0,
                        help="surface oversampling. Higher = fewer pinholes "
                             "in the rasterized shell, more memory.")
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Gaussian sigma (voxels) applied to the solid "
                             "indicator before marching cubes. Removes the "
                             "voxel staircase. 0 to disable.")
    parser.add_argument("--no-flood-y0", action="store_true",
                        help="By default we add a y=0 wall so outside can't "
                             "leak through the symmetry-plane cap. Disable "
                             "if input already has a fully sealed y=0 face.")
    args = parser.parse_args()

    import trimesh

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[input] {args.input}")
    t0 = time.time()
    m = trimesh.load(str(args.input), force="mesh", process=False)
    print(f"[input] {len(m.faces):,} tris ({time.time() - t0:.1f}s)")
    print(f"[input] bounds: {m.bounds[0]} -> {m.bounds[1]}")

    bmin = m.bounds[0] - args.pad
    bmax = m.bounds[1] + args.pad
    size = bmax - bmin
    nx, ny, nz = (np.ceil(size / args.pitch).astype(int) + 1).tolist()
    print(f"[grid] {nx} x {ny} x {nz} = {nx*ny*nz:,} voxels "
          f"@ pitch={args.pitch} mm")

    surface = voxelize_surface(m, bmin, args.pitch, (nx, ny, nz),
                                args.samples_per_voxel)

    # Safety: explicitly fill the y=0 plane voxels under the input footprint.
    # If the symmetry-plane cap has any sub-voxel gaps the flood will spill
    # below it. This is a flat slab so a single layer is enough.
    if not args.no_flood_y0:
        y0_idx = int(np.floor((0.0 - bmin[1]) / args.pitch))
        if 0 <= y0_idx < ny:
            # union the input footprint slab with the surface
            footprint = surface.any(axis=1)  # XZ presence
            surface[:, y0_idx, :] |= footprint
            # also seal one voxel above (positive y) so we don't depend on it
            if y0_idx + 1 < ny:
                surface[:, y0_idx + 1, :] |= footprint
            print(f"[y0-cap] sealed y=0 slab at j={y0_idx} "
                  f"({footprint.sum():,} XZ cells)")

    solid = close_and_flood(surface, args.close_iters)
    out = marching_cubes(solid, bmin, args.pitch, args.smooth_sigma)
    out = keep_largest_component(out, fix_orient=True)

    out.export(str(args.output), file_type="stl")
    print(f"\n[out] {args.output}  ({len(out.faces):,} faces, "
          f"{len(out.vertices):,} verts)")
    print(f"  watertight:    {out.is_watertight}")
    try:
        print(f"  volume:        {out.volume:.0f} mm^3")
    except Exception as exc:
        print(f"  volume:        failed ({exc})")
    print(f"  bbox:          {out.bounds[0]} -> {out.bounds[1]}")


if __name__ == "__main__":
    main()
