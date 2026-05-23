"""End-to-end pipeline: make ``combined_body.stl`` watertight.

Stages:
  1. PyMeshLab → Screened Poisson surface reconstruction.
  2. Crop Poisson's extrapolation past the input bbox + pad.
  3. PyMeshLab → ``meshing_close_holes`` (max hole = 1e6 edges).
  4. Trimesh → ``submesh`` keep largest connected component, fix orient.
  5. Custom earcut → triangulate any remaining open boundary loops
     (projects each to its best-fit 2D plane via PCA, then earcut).
  6. Write watertight STL.

Depth knob:
  d=8  → ~256 voxels along longest axis (~18 mm on 4.5 m car)  fast
  d=9  → 512   (~9 mm)
  d=10 → 1024  (~4 mm)             ← good default
  d=11 → 2048  (~2 mm)             slow (~15 min Poisson)
  d=12 → 4096  (~1 mm)             very slow

Requires:
  pip install open3d pymeshlab mapbox_earcut libigl scikit-image
  LD_LIBRARY_PATH includes conda's libstdc++.so.6 (for open3d/pymeshlab)

Usage::

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python close_body_watertight_pipeline.py \\
        --input  outputs/integrate/<stem>_combined_body.stl \\
        --output outputs/integrate/<stem>_watertight.stl \\
        --depth 10 --n-samples 500000
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def stage_poisson(input_path: Path, depth: int, n_samples: int,
                   pointweight: float, samplespernode: float):
    """SPSR via PyMeshLab. Returns the MeshSet with the reconstructed
    mesh as the current mesh."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    t0 = time.time()
    ms.load_new_mesh(str(input_path))
    print(f"[load] {ms.current_mesh().face_number():,} faces "
          f"({time.time() - t0:.1f}s)")
    ms.compute_normal_per_vertex()

    t0 = time.time()
    ms.generate_simplified_point_cloud(samplenum=n_samples)
    print(f"[sample] {ms.current_mesh().vertex_number():,} pts "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    ms.generate_surface_reconstruction_screened_poisson(
        depth=depth,
        fulldepth=5,
        pointweight=pointweight,
        samplespernode=samplespernode,
        scale=1.05,
        iters=8,
        confidence=False,
        preclean=True,
    )
    print(f"[poisson d={depth}] {ms.current_mesh().face_number():,} faces "
          f"({time.time() - t0:.1f}s)")
    return ms


def stage_cleanup(ms, input_bbox_min, input_bbox_max, crop_pad_mm: float):
    """Remove duplicates, crop to input bbox + pad, close holes."""
    import pymeshlab

    t0 = time.time()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_null_faces()
    print(f"[cleanup] {ms.current_mesh().face_number():,} faces "
          f"({time.time() - t0:.1f}s)")

    # Crop to input bbox + pad
    mn = input_bbox_min - crop_pad_mm
    mx = input_bbox_max + crop_pad_mm
    cond = (f"(x < {mn[0]}) || (x > {mx[0]}) || "
            f"(y < {mn[1]}) || (y > {mx[1]}) || "
            f"(z < {mn[2]}) || (z > {mx[2]})")
    t0 = time.time()
    try:
        ms.compute_selection_by_condition_per_vertex(condselect=cond)
        ms.compute_selection_transfer_vertex_to_face(inclusive=False)
        ms.meshing_remove_selected_vertices_and_faces()
    except Exception as exc:
        print(f"[crop] failed: {exc}")
    print(f"[crop] kept {ms.current_mesh().face_number():,} faces "
          f"({time.time() - t0:.1f}s)")

    # Close holes
    t0 = time.time()
    try:
        ms.meshing_close_holes(maxholesize=1_000_000)
    except Exception as exc:
        print(f"[close-holes] failed: {exc}")
    print(f"[close-holes] {ms.current_mesh().face_number():,} faces "
          f"({time.time() - t0:.1f}s)")


def stage_finalise(ms, output_path: Path, scratch_dir: Path) -> "trimesh.Trimesh":
    """Save PyMeshLab output, reload in trimesh, keep largest component,
    fix orientation, fill any remaining open loops with earcut, write
    final STL."""
    import trimesh

    intermediate = scratch_dir / (output_path.stem + "_pml.stl")
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(intermediate), binary=True)

    m = trimesh.load(str(intermediate), force="mesh", process=True)
    parts = m.split(only_watertight=False)
    parts.sort(key=lambda p: len(p.faces), reverse=True)
    print(f"[components] {len(parts)} (top: "
          f"{[len(p.faces) for p in parts[:5]]})")
    main = parts[0]
    if main.volume < 0:
        main.invert()

    # Fill any remaining open loops.
    main = fill_open_loops_earcut(main)

    main.export(str(output_path), file_type="stl")
    return main


# ---------------------------------------------------------------------------
# Custom open-loop fill via earcut
# ---------------------------------------------------------------------------

def _traverse_loop(comp, adj):
    comp_set = set(comp)
    sub_adj = {i: [j for j in adj[i] if j in comp_set] for i in comp}
    start = next((i for i in comp if len(sub_adj[i]) >= 1), comp[0])
    path = [start]
    seen = {start}
    cur = start
    while True:
        nxts = [n for n in sub_adj[cur] if n not in seen]
        if not nxts:
            break
        cur = nxts[0]
        seen.add(cur)
        path.append(cur)
    return path


def _best_2d_projection(pts3: np.ndarray) -> np.ndarray:
    """PCA: drop smallest principal component, project onto the other two."""
    centered = pts3 - pts3.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def fill_open_loops_earcut(mesh) -> "trimesh.Trimesh":
    import mapbox_earcut
    import trimesh

    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return mesh
    adj = defaultdict(set)
    for i, j in boundary:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    visited = set()
    loops = []
    for s in list(adj.keys()):
        if s in visited:
            continue
        comp = []
        stack = [s]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            stack.extend(adj[n] - visited)
        if len(comp) >= 3:
            loops.append(comp)
    print(f"[fill] {len(loops)} open loops to triangulate")
    if not loops:
        return mesh

    v = np.asarray(mesh.vertices).copy()
    new_faces = list(mesh.faces)
    n_added = 0
    for li, comp in enumerate(loops):
        path = _traverse_loop(comp, adj)
        if len(path) < 3:
            continue
        pts3 = v[path]
        pts2 = _best_2d_projection(pts3)
        rings = np.asarray([len(path)], dtype=np.uint32)
        try:
            tri = mapbox_earcut.triangulate_float64(
                np.ascontiguousarray(pts2, dtype=np.float64), rings)
        except Exception as exc:
            print(f"[fill] loop {li} ({len(path)} pts) earcut failed: {exc}")
            continue
        if len(tri) == 0:
            print(f"[fill] loop {li} ({len(path)} pts) earcut returned 0")
            continue
        for t in range(0, len(tri), 3):
            new_faces.append([
                path[int(tri[t])],
                path[int(tri[t + 1])],
                path[int(tri[t + 2])]])
            n_added += 1
        print(f"[fill] loop {li} ({len(path)} pts) -> {len(tri)//3} tris")
    out = trimesh.Trimesh(vertices=v,
                           faces=np.asarray(new_faces, dtype=np.int64),
                           process=True)
    print(f"[fill] added {n_added} tris -> {len(out.faces):,} faces")
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
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--pointweight", type=float, default=4.0)
    parser.add_argument("--samples-per-node", type=float, default=1.5)
    parser.add_argument("--crop-pad", type=float, default=50.0)
    parser.add_argument("--scratch", type=Path, default=None)
    args = parser.parse_args()
    if args.scratch is None:
        args.scratch = args.output.parent / "_pml_scratch"
    args.scratch.mkdir(parents=True, exist_ok=True)

    # Stage 1: Poisson
    ms = stage_poisson(args.input, args.depth, args.n_samples,
                        args.pointweight, args.samples_per_node)

    # Need input bbox for crop — reload input.
    import pymeshlab
    ms2 = pymeshlab.MeshSet()
    ms2.load_new_mesh(str(args.input))
    ibm = ms2.current_mesh().bounding_box()
    in_min = np.array([ibm.min()[0], ibm.min()[1], ibm.min()[2]])
    in_max = np.array([ibm.max()[0], ibm.max()[1], ibm.max()[2]])
    del ms2

    # Stage 2: cleanup, crop, close holes
    stage_cleanup(ms, in_min, in_max, args.crop_pad)

    # Stage 3: finalise (largest comp + fill leftovers + save)
    main = stage_finalise(ms, args.output, args.scratch)

    print(f"\n[out] {args.output}  ({len(main.faces):,} faces, "
          f"{len(main.vertices):,} verts)")
    print(f"  watertight:    {main.is_watertight}")
    print(f"  bodies:        {main.body_count}")
    print(f"  volume:        {main.volume:.0f} mm^3")
    print(f"  bbox:          {main.bounds[0]} → {main.bounds[1]}")


if __name__ == "__main__":
    main()
