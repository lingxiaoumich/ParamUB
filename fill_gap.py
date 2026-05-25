"""Fill the gap between the extracted upper shell and the trimmed
parametric underbody with shared-slant ribbon triangles.

Pipeline position
-----------------
This is step 3 of the reconstructed-car body pipeline:

    1.  run_shell.py            -- extract upper-body shell
    2.  integrate_underbody.py  -- drop in parametric UB, trim to shell
    3.  fill_gap.py             -- close the residual shell <-> UB
                                   rim ribbon  (this script)
    4.  make_watertight.py      -- Blender smooth-remesh + trimesh
                                   extract -> strictly watertight body

See ``paramub/docs/fill_gap.html`` for the full HTML guide for this
script, ``paramub/docs/watertight.html`` for the stage-4 finisher,
and ``paramub/docs/index.html`` for the pipeline overview.

Approach: per-UB-vert slant + bipartite-zip ribbon strip
--------------------------------------------------------
1. Extract the longest open boundary of each input mesh: shell rim
   (~3.7k verts on the example car) and UB rim (~1.9k verts).
2. For each UB rim vert ``u_j``, pick its apex ``apex_j`` = nearest
   shell-rim vert (3-D distance). Subdivide the slant ``u_j → apex_j``
   into ``K_j`` segments at ``target_edge_mm`` spacing (default: auto
   from shell median edge length, typically ~6 mm). The slant verts
   are stored ONCE and reused by both adjacent ribbons.
3. For each UB rim edge ``(u_j, u_{j+1})``, build the ribbon between
   ``slant[j]`` and ``slant[j+1]`` by bipartite-zip triangulation:
   at each step advance whichever slant pointer yields the shorter
   next diagonal. This handles per-slant K mismatches automatically.
   Adjacent ribbons share their slant verts → no wedge holes; the
   gap is watertight on the ribbon (UB) side.
4. The shell side keeps "intentional" polygon holes whose boundary is
   the apex chain plus the shell-rim arcs between consecutive apices
   — the apex chain only samples the shell rim sparsely. Use
   ``--repair`` to feed those holes to MeshLab's ``close_holes`` (+
   optional isotropic remesh) afterwards.

Usage (recommended end-to-end recipe for a watertight result)::

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python fill_gap.py \\
        --shell outputs/integrate/<base>_shell.stl \\
        --ub    outputs/integrate/<base>_underbody_trimmed.stl \\
        --out          outputs/integrate/<base>_gap.stl \\
        --combined-out outputs/integrate/<base>_combined.stl \\
        --repair --repair-close-holes-max 10000 \\
        --repair-remesh-mm 6 --repair-remesh-iters 3

Minimal usage (ribbon mesh only, no repair, no combined STL)::

    python fill_gap.py \\
        --shell outputs/integrate/<base>_shell.stl \\
        --ub    outputs/integrate/<base>_underbody_trimmed.stl \\
        --out   outputs/integrate/<base>_gap.stl

``LD_LIBRARY_PATH`` is only needed when ``--repair`` is set (PyMeshLab
wants the conda env's libstdc++).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_BASE = (
    "outputs/integrate/"
    "alfa_romeo_giuliazhuliye_2025_image10_71415_shadowfill"
)


def extract_longest_boundary(mesh):
    """Return ``(vertex_indices, edges, perimeter)`` of the longest open
    boundary component of ``mesh``."""
    from paramub.shell.recut import find_open_boundary_components
    _, comps = find_open_boundary_components(mesh)
    if not comps:
        raise RuntimeError("mesh has no open-boundary components")
    return comps[0]


def walk_ub_loop(edges) -> list[int]:
    """Walk a closed UB-rim boundary into an ordered vertex loop. The
    UB rim is a clean degree-2 loop (no T-junctions), so a naive walk
    is sufficient."""
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    start = next(iter(adj))
    loop = [start]; seen = {start}; cur = start
    while True:
        nxts = [n for n in adj[cur] if n not in seen]
        if not nxts:
            break
        cur = nxts[0]
        loop.append(cur); seen.add(cur)
    return loop


def build_shared_slant_ribbons(ub_loop: list[int],
                                  shell_rim_idx: np.ndarray,
                                  shell_v3d: np.ndarray,
                                  ub_v3d: np.ndarray,
                                  target_edge_mm: float
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Shared-slant ribbon strip — no wedge holes on the ribbon side.

    For each UB rim vert ``u_j`` we build ONE subdivided slant going
    to its nearest shell-rim vert ``apex_j``, with K_j subdivisions
    based on slant length / target_edge_mm. The slant verts are
    stored ONCE and reused by both adjacent ribbons.

    For each UB edge ``(u_j, u_{j+1})`` the ribbon between
    ``slant[j]`` and ``slant[j+1]`` is built by a bipartite-zip
    triangulation: at every step we advance whichever slant pointer
    yields the shorter next diagonal, emitting one triangle per
    step. This handles per-slant K mismatches automatically and
    produces a clean strip whose right slant is bit-shared with the
    next ribbon's left slant.

    Returns ``(verts, faces)``; verts = [shell rim verts | UB loop
    verts | interior slant verts]. Boundary edges of the resulting
    gap mesh are only:
      - UB base edges (u_j, u_{j+1}) — shared with UB rim
      - "Top" edges between adjacent apices when apex_j ≠ apex_{j+1}.
    All shell-rim verts that aren't used as an apex stay ON the
    shell rim, so the shell side has intentional polygon holes whose
    boundary is the apex chain plus the shell-rim arcs between them.
    """
    from scipy.spatial import cKDTree
    n_s = len(shell_rim_idx)
    n_u = len(ub_loop)
    s_pos = shell_v3d[shell_rim_idx]
    u_pos = ub_v3d[ub_loop]
    SH_OFF = 0
    UB_OFF = n_s
    INT_OFF = n_s + n_u

    s_tree = cKDTree(s_pos)
    _, apex_per_u = s_tree.query(u_pos)   # (n_u,) shell-local

    # Build slants once per UB vert.
    slants: list[list[int]] = []
    interior_verts: list[np.ndarray] = []
    int_count = 0
    for j in range(n_u):
        u = u_pos[j]
        apex_local = int(apex_per_u[j])
        apex = s_pos[apex_local]
        L = float(np.linalg.norm(apex - u))
        K = max(1, int(round(L / target_edge_mm)))
        chain = [UB_OFF + j]
        if K > 1:
            ts = np.linspace(0.0, 1.0, K + 1)[1:-1]
            ints = (u[None, :] * (1 - ts[:, None])
                    + apex[None, :] * ts[:, None])
            interior_verts.append(ints)
            for k in range(K - 1):
                chain.append(INT_OFF + int_count + k)
            int_count += K - 1
        chain.append(SH_OFF + apex_local)
        slants.append(chain)

    if interior_verts:
        all_int = np.vstack(interior_verts)
    else:
        all_int = np.zeros((0, 3), dtype=np.float64)
    verts = np.vstack([s_pos, u_pos, all_int])

    # Bipartite-zip each consecutive slant pair.
    faces: list[tuple[int, int, int]] = []
    for j in range(n_u):
        L = slants[j]
        R = slants[(j + 1) % n_u]
        li, ri = 0, 0
        while li < len(L) - 1 or ri < len(R) - 1:
            can_l = li < len(L) - 1
            can_r = ri < len(R) - 1
            if can_l and can_r:
                d_l = float(np.linalg.norm(verts[L[li + 1]] - verts[R[ri]]))
                d_r = float(np.linalg.norm(verts[R[ri + 1]] - verts[L[li]]))
                advance_l = d_l <= d_r
            else:
                advance_l = can_l
            if advance_l:
                faces.append((L[li], R[ri], L[li + 1]))
                li += 1
            else:
                faces.append((L[li], R[ri], R[ri + 1]))
                ri += 1

    faces_arr = np.asarray(faces, dtype=np.int64)
    Ks = [len(s) - 1 for s in slants]
    print(f"[ribbons] {n_u:,} slants  "
          f"(K min/median/max = {min(Ks)}/{int(np.median(Ks))}/{max(Ks)})  "
          f"→ {len(faces_arr):,} tris  "
          f"({len(all_int):,} shared interior slant verts)")
    return verts, faces_arr


def pymeshlab_repair(combined_mesh,
                       close_holes_max: int = 10000,
                       remesh_target_mm: float = 0.0,
                       remesh_iterations: int = 3,
                       scratch_dir: Path | None = None,
                       close_holes_open_edge_guard: int = 5000):
    """Repair the combined mesh: merge close verts, fix non-manifold
    edges, close_holes, optionally isotropic remesh + re-seal.

    Note: ``close_holes`` (MeshLab's Liepa min-weight DP triangulator)
    can leave holes unfilled when the boundary loop is pinched or
    self-touching — adjacency artifacts at apex-sharing points can
    confuse it. The intentional polygon holes on the shell side are
    well-shaped and usually fill cleanly.

    The DP is O(n³) per hole, so a single large hole on a pathological
    reconstruction (many thousands of open edges) hangs the call for
    hours. When the input open-edge count exceeds
    ``close_holes_open_edge_guard``, the maxholesize is auto-capped to
    a small value (200) so the DP only attempts well-shaped small
    holes; any genuinely large pathological holes are left for the
    downstream stage-4 voxel remesh to close.
    """
    import pymeshlab
    import trimesh
    scratch_dir = scratch_dir or Path("/tmp")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    src = scratch_dir / "repair_in.stl"
    dst = scratch_dir / "repair_out.stl"
    combined_mesh.export(str(src), file_type="stl")

    in_edges = combined_mesh.edges_sorted
    _, in_counts = np.unique(in_edges, axis=0, return_counts=True)
    in_open = int((in_counts == 1).sum())

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    print(f"[repair] in: {ms.current_mesh().face_number():,} tris  "
          f"open_edges={in_open:,}")

    effective_close_holes_max = close_holes_max
    if (close_holes_max > 0
            and in_open > close_holes_open_edge_guard):
        effective_close_holes_max = 0
        print(f"[repair] open_edges {in_open:,} > guard "
              f"{close_holes_open_edge_guard:,} — SKIPPING close_holes "
              f"entirely (capping maxholesize wasn't enough; cumulative "
              f"DP cost across many small holes still hangs). Stage 4's "
              f"voxel remesh will close the remaining holes.")

    try:
        ms.meshing_merge_close_vertices(
            threshold=pymeshlab.Percentage(0.0001))
        print(f"[repair] merge close verts: "
              f"{ms.current_mesh().face_number():,} tris")
    except Exception as exc:
        print(f"[repair] merge close skipped: {exc}")

    try:
        ms.meshing_repair_non_manifold_edges(method=0)
        print(f"[repair] non-manifold fixed: "
              f"{ms.current_mesh().face_number():,} tris")
    except Exception as exc:
        print(f"[repair] non-manifold repair skipped: {exc}")

    if effective_close_holes_max > 0:
        try:
            ms.meshing_close_holes(
                maxholesize=effective_close_holes_max,
                selfintersection=False)
            print(f"[repair] close_holes(max={effective_close_holes_max}): "
                  f"{ms.current_mesh().face_number():,} tris")
        except Exception as exc:
            print(f"[repair] close_holes skipped: {exc}")

    if remesh_target_mm > 0:
        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.AbsoluteValue(remesh_target_mm),
                iterations=remesh_iterations,
                adaptive=True,
                checksurfdist=False)
            print(f"[repair] remesh (target={remesh_target_mm}mm, "
                  f"iters={remesh_iterations}): "
                  f"{ms.current_mesh().face_number():,} tris")
        except Exception as exc:
            print(f"[repair] remesh skipped: {exc}")
        try:
            ms.meshing_repair_non_manifold_edges(method=0)
            ms.meshing_close_holes(
                maxholesize=effective_close_holes_max,
                selfintersection=False)
            print(f"[repair] post-remesh re-seal: "
                  f"{ms.current_mesh().face_number():,} tris")
        except Exception as exc:
            print(f"[repair] post-remesh re-seal skipped: {exc}")

    ms.save_current_mesh(str(dst))
    out = trimesh.load(str(dst), force="mesh", process=True)
    edges = out.edges_sorted
    _, c = np.unique(edges, axis=0, return_counts=True)
    print(f"[repair] final: f={len(out.faces):,} v={len(out.vertices):,}  "
          f"open={int((c == 1).sum()):,}  "
          f"non-manifold={int((c > 2).sum()):,}")
    return out


def main():
    p = argparse.ArgumentParser(
        description="Fill shell ↔ UB rim gap with shared-slant ribbons.")
    p.add_argument("--shell", type=Path,
                   default=Path(f"{DEFAULT_BASE}_shell.stl"))
    p.add_argument("--ub", type=Path,
                   default=Path(f"{DEFAULT_BASE}_underbody_trimmed.stl"))
    p.add_argument("--out", type=Path,
                   default=Path(f"{DEFAULT_BASE}_gap.stl"))
    p.add_argument("--combined-out", type=Path, default=None,
                   help="If set, also write shell + UB + ribbons "
                        "concatenated as a combined STL.")
    p.add_argument("--ribbon-target-mm", type=float, default=0.0,
                   help="Spoke-axis subdivision target edge length. "
                        "0 = auto from shell median edge length.")
    p.add_argument("--repair", action="store_true",
                   help="Run pymeshlab repair on the combined mesh: "
                        "merge close verts, fix non-manifold, "
                        "close_holes (+ optional remesh). Requires "
                        "--combined-out.")
    p.add_argument("--repair-close-holes-max", type=int, default=10000)
    p.add_argument("--repair-remesh-mm", type=float, default=0.0,
                   help="Isotropic remesh target edge length (mm). "
                        "0 = no remesh.")
    p.add_argument("--repair-remesh-iters", type=int, default=3)
    args = p.parse_args()

    import trimesh
    print(f"[load] shell = {args.shell}")
    shell = trimesh.load(str(args.shell), force="mesh", process=True)
    print(f"  f={len(shell.faces):,} v={len(shell.vertices):,}")
    print(f"[load] ub    = {args.ub}")
    ub = trimesh.load(str(args.ub), force="mesh", process=True)
    print(f"  f={len(ub.faces):,} v={len(ub.vertices):,}")

    s_vidx, _,         s_peri = extract_longest_boundary(shell)
    u_vidx, u_edges,   u_peri = extract_longest_boundary(ub)
    print(f"[bnd] shell rim: {len(s_vidx):,} verts, perim {s_peri:.0f} mm")
    print(f"[bnd] ub rim:    {len(u_vidx):,} verts, perim {u_peri:.0f} mm")

    loop = walk_ub_loop(u_edges)
    print(f"[loop] ub loop walked: {len(loop)} / {len(u_vidx)} verts")

    if args.ribbon_target_mm > 0:
        tgt = args.ribbon_target_mm
    else:
        s_edges_lens = np.linalg.norm(
            shell.vertices[shell.edges_sorted[:, 0]]
            - shell.vertices[shell.edges_sorted[:, 1]], axis=1)
        tgt = float(np.median(s_edges_lens))
        print(f"[ribbons] auto target = shell median edge = "
              f"{tgt:.2f} mm")

    gap_v, gap_f = build_shared_slant_ribbons(
        loop, np.asarray(s_vidx),
        shell.vertices, ub.vertices,
        target_edge_mm=tgt)

    gap = trimesh.Trimesh(vertices=gap_v, faces=gap_f, process=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gap.export(str(args.out), file_type="stl")
    print(f"[out] {args.out}  f={len(gap.faces):,} v={len(gap.vertices):,}")

    if args.combined_out is not None:
        combined = trimesh.util.concatenate([shell, ub, gap])
        combined = trimesh.Trimesh(
            vertices=combined.vertices, faces=combined.faces, process=True)
        edges = combined.edges_sorted
        _, c = np.unique(edges, axis=0, return_counts=True)
        print(f"[check] combined: f={len(combined.faces):,} "
              f"v={len(combined.vertices):,}  "
              f"open edges={int((c == 1).sum()):,}  "
              f"non-manifold={int((c > 2).sum()):,}")
        if args.repair:
            combined = pymeshlab_repair(
                combined,
                close_holes_max=args.repair_close_holes_max,
                remesh_target_mm=args.repair_remesh_mm,
                remesh_iterations=args.repair_remesh_iters)
        combined.export(str(args.combined_out), file_type="stl")
        print(f"[out] {args.combined_out}")


if __name__ == "__main__":
    main()
