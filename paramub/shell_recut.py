"""Re-cut the canonical (raw) mesh using the longest open-boundary loop
of the extracted upper shell as a 3-D scissor.

Steps 5–7 of the shell extraction can leave small holes in the upper
shell (over-eager debris drop, wheelhouse zone bleeding into a body
panel, etc.). The OUTER rim of the upper shell — where it meets the
underbody / wheel-arch openings — is by far the longest open-edge loop.
Using that rim as a cutting curve on the original canonical mesh
recovers any faces that were dropped accidentally inside the rim, while
still cleanly excluding everything below the rim (wheels, underbody).

Algorithm
---------
1. Find boundary edges (edges with exactly one face) of the extracted
   upper shell.
2. Trace them into closed loops via vertex adjacency. Each loop is one
   open ring on the shell (outer rim + any internal holes).
3. Compute each loop's perimeter (Euclidean) and pick the longest one.
4. Map that loop's vertex positions back to canonical-mesh vertex
   indices via a KDTree (distances should be ~0 because no vertex
   manipulation happens between s1_canonicalize and the final
   submesh — only face removal).
5. Build a set of "rim edges" in canonical-mesh index space (consecutive
   vertex pairs along the traced loop).
6. Build canonical mesh face adjacency, drop any adjacency pair whose
   shared edge is in the rim-edge set. Now the canonical mesh's face
   graph is split into ≥2 components by the rim.
7. Pick a seed face in the upper region (highest-z face centroid that
   isn't itself touching the rim). Keep its connected component.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _trace_loops(boundary_edges: np.ndarray) -> list[list[int]]:
    """Group boundary edges (Nx2 vertex pairs) into closed/open chains.
    Returns each chain as an ordered list of vertex indices."""
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        ai, bi = int(a), int(b)
        adj[ai].add(bi)
        adj[bi].add(ai)

    visited_in_component: set[int] = set()
    loops: list[list[int]] = []
    for start in list(adj.keys()):
        if start in visited_in_component:
            continue
        # Component (connected via boundary edges)
        comp = []
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited_in_component:
                continue
            visited_in_component.add(n)
            comp.append(n)
            stack.extend(adj[n] - visited_in_component)
        if len(comp) < 3:
            continue
        # Walk the component in order (assumes clean ring; pick any vertex
        # of degree-2 as the start, fall back to any vertex).
        comp_set = set(comp)
        sub_adj = {v: [n for n in adj[v] if n in comp_set] for v in comp}
        ring_start = next(
            (v for v in comp if len(sub_adj[v]) == 2),
            comp[0])
        path = [ring_start]
        seen = {ring_start}
        cur = ring_start
        while True:
            nxts = [n for n in sub_adj[cur] if n not in seen]
            if not nxts:
                break
            cur = nxts[0]
            seen.add(cur)
            path.append(cur)
        if len(path) >= 3:
            loops.append(path)
    return loops


def _loop_perimeter(loop_idx: list[int], verts: np.ndarray) -> float:
    pts = verts[loop_idx]
    diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def find_open_boundary_components(mesh):
    """Group boundary edges into connected components and rank by total
    edge length.

    A component is the set of boundary edges and their endpoint vertices
    reachable via boundary-edge adjacency. Robust against T-junctions
    (vertices of degree ≥ 3 along the boundary, which appear when two
    surface sheets share a single vertex but not an edge) — the whole
    branching ring stays one component instead of being fragmented by a
    naive single-path walker.

    Returns
    -------
    boundary_edges : np.ndarray (M, 2)
        All open boundary edges (each edge appears in exactly one face),
        sorted (smaller vertex index first).
    components : list of (vertex_indices np.ndarray, edges np.ndarray,
                          perimeter float)
        Sorted by ``perimeter`` descending.
    """
    import numpy as np
    edges = mesh.edges_sorted
    u, c = np.unique(edges, axis=0, return_counts=True)
    boundary = u[c == 1]
    if len(boundary) == 0:
        return boundary, []

    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    # BFS each connected vertex component.
    visited: set[int] = set()
    vert_comps: list[set[int]] = []
    for start in list(adj.keys()):
        if start in visited:
            continue
        comp: set[int] = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            stack.extend(adj[n] - visited)
        if len(comp) >= 2:
            vert_comps.append(comp)

    # Bucket boundary edges by component using vertex -> component-id map.
    vert_to_comp = np.full(int(boundary.max()) + 1, -1, dtype=np.int64)
    for cid, vs in enumerate(vert_comps):
        for v in vs:
            vert_to_comp[v] = cid
    edge_cid = vert_to_comp[boundary[:, 0]]
    # (sanity: both endpoints are in the same component by construction)

    components = []
    verts_arr = np.asarray(mesh.vertices)
    for cid, vs in enumerate(vert_comps):
        edges_cid = boundary[edge_cid == cid]
        if len(edges_cid) == 0:
            continue
        d = np.linalg.norm(
            verts_arr[edges_cid[:, 0]] - verts_arr[edges_cid[:, 1]],
            axis=1)
        components.append((
            np.fromiter(vs, dtype=np.int64, count=len(vs)),
            edges_cid,
            float(d.sum()),
        ))
    components.sort(key=lambda c: c[2], reverse=True)
    return boundary, components


def find_open_boundary_loops(mesh):
    """Backwards-compatible thin wrapper. Returns ``(boundary_edges,
    loop_vertex_rings)`` — but each ring is now just the largest
    spanning path inside one component, so callers wanting a robust
    "longest loop" should use :func:`find_open_boundary_components`
    instead."""
    boundary, components = find_open_boundary_components(mesh)
    loops: list[list[int]] = []
    for vs, edges, _peri in components:
        sub_loops = _trace_loops(edges)
        loops.extend(sub_loops)
    loops.sort(key=lambda lp: _loop_perimeter(lp, mesh.vertices), reverse=True)
    return boundary, loops


def export_edges_as_tubes(mesh, edges: np.ndarray, out_path,
                            radius_rel: float = 0.001,
                            sections: int = 4) -> int:
    """Visualize a set of boundary edges as a tube-mesh STL.

    ``radius_rel`` is a fraction of the mesh diagonal.
    Returns the number of edges drawn.
    """
    import numpy as np
    from .shell_extract import edges_to_tube_mesh
    if len(edges) == 0:
        return 0
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    radius = radius_rel * diag
    tube = edges_to_tube_mesh(
        np.asarray(mesh.vertices), np.asarray(edges),
        radius=radius, sections=sections)
    tube.export(str(out_path), file_type="stl")
    return int(len(edges))


def loop_to_edges(loop: list[int]) -> "np.ndarray":
    """Convert an ordered vertex-index ring to ``(N, 2)`` edge pairs
    (closes the loop with the trailing back-to-start edge)."""
    import numpy as np
    n = len(loop)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(loop, dtype=np.int64)
    return np.column_stack([arr, np.roll(arr, -1)])


def recut_with_longest_rim(canonical_mesh,
                            final_mesh,
                            snap_tol: float | None = None,
                            verbose: bool = True):
    """Re-cut ``canonical_mesh`` using the longest open-boundary
    component of ``final_mesh`` as a 3-D fence.

    "Component" rather than "loop": we use *all* edges of the largest
    connected boundary-edge component (vertices reachable from each
    other through boundary edges). That's robust to T-junctions, where
    a naive single-path walk would otherwise truncate.

    Returns ``(cut_mesh, info)`` where ``cut_mesh`` is a trimesh.Trimesh
    of the kept faces and ``info`` is a dict with diagnostic counts.
    """
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    info: dict = {}

    # ---- 1. Boundary edges + components ------------------------------
    boundary, components = find_open_boundary_components(final_mesh)
    info["final_mesh_boundary_edges"] = int(len(boundary))
    info["component_count"] = len(components)
    if not components:
        if verbose:
            print("[recut] final_mesh has no open boundary; nothing to cut")
        return final_mesh, info

    if verbose:
        peri_top = [round(c[2], 4) for c in components[:5]]
        size_top = [len(c[0]) for c in components[:5]]
        print(f"[recut] {len(components)} boundary components; "
              f"top-5 perimeter = {peri_top}  vert counts = {size_top}")

    largest_verts, largest_edges, largest_peri = components[0]
    info["rim_perimeter"] = largest_peri
    info["rim_vertex_count"] = int(len(largest_verts))
    info["rim_edge_count"] = int(len(largest_edges))
    info["other_component_perimeters"] = [c[2] for c in components[1:]]

    # ---- 2. Snap rim verts back to canonical-mesh indices ------------
    if snap_tol is None:
        try:
            edge_len = canonical_mesh.edges_unique_length
            snap_tol = 2.0 * float(np.median(edge_len))
        except Exception:
            snap_tol = 1e-3
    rim_positions = np.asarray(final_mesh.vertices)[largest_verts]
    tree = cKDTree(np.asarray(canonical_mesh.vertices))
    dist, nn = tree.query(rim_positions, distance_upper_bound=snap_tol)
    valid = np.isfinite(dist)
    info["rim_snap_tol"] = float(snap_tol)
    info["rim_snapped"] = int(valid.sum())
    info["rim_total"] = int(len(valid))
    if verbose:
        max_d = float(dist[valid].max()) if valid.any() else float("nan")
        print(f"[recut] snap rim to canonical: "
              f"{valid.sum()}/{len(valid)} within {snap_tol:.4g} "
              f"(max d among snapped = {max_d:.4g})")
    if valid.sum() < 3:
        if verbose:
            print("[recut] not enough snapped rim verts; returning final_mesh")
        return final_mesh, info

    # ---- 3. Build rim edges in canonical index space -----------------
    # Per-final-vertex -> canonical-vertex map (only valid entries).
    final_to_canon = np.full(len(rim_positions), -1, dtype=np.int64)
    final_to_canon[valid] = nn[valid]
    # final-mesh-index -> rim-position-index (within largest_verts)
    final_idx_to_rim_pos = {int(v): i for i, v in enumerate(largest_verts)}

    rim_edges: set[tuple[int, int]] = set()
    for fa, fb in largest_edges:
        ia = final_idx_to_rim_pos.get(int(fa), -1)
        ib = final_idx_to_rim_pos.get(int(fb), -1)
        if ia < 0 or ib < 0:
            continue
        ca, cb = final_to_canon[ia], final_to_canon[ib]
        if ca < 0 or cb < 0 or ca == cb:
            continue
        rim_edges.add((int(min(ca, cb)), int(max(ca, cb))))
    info["rim_edges_built"] = len(rim_edges)
    if not rim_edges:
        if verbose:
            print("[recut] no usable rim edges; returning final_mesh")
        return final_mesh, info

    # ---- 6. Canonical face adjacency with rim edges removed -----------
    fa = np.asarray(canonical_mesh.face_adjacency, dtype=np.int64)
    fae = np.asarray(canonical_mesh.face_adjacency_edges, dtype=np.int64)
    fae_sorted = np.sort(fae, axis=1)
    fae_tuples = list(map(tuple, fae_sorted.tolist()))
    keep_pair = np.fromiter(
        (t not in rim_edges for t in fae_tuples),
        dtype=bool, count=len(fae_tuples))
    fa_kept = fa[keep_pair]
    info["adjacency_pairs_total"] = int(len(fa))
    info["adjacency_pairs_after_rim_cut"] = int(len(fa_kept))

    n_faces = len(canonical_mesh.faces)
    if len(fa_kept) == 0:
        if verbose:
            print("[recut] no face adjacency left after cut (degenerate)")
        return final_mesh, info
    rows = np.concatenate([fa_kept[:, 0], fa_kept[:, 1]])
    cols = np.concatenate([fa_kept[:, 1], fa_kept[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    adj_mat = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    n_comp, labels = connected_components(adj_mat, directed=False)
    info["face_components"] = int(n_comp)

    # ---- 7. Pick a seed face: highest-z centroid that isn't on the rim
    centroids = canonical_mesh.vertices[canonical_mesh.faces].mean(axis=1)
    rim_vert_set = set(int(nn[i]) for i in range(len(nn)) if valid[i])
    if rim_vert_set:
        rim_vert_arr = np.fromiter(rim_vert_set, dtype=np.int64,
                                    count=len(rim_vert_set))
        in_rim = np.zeros(n_faces, dtype=bool)
        f = canonical_mesh.faces
        in_rim |= np.isin(f[:, 0], rim_vert_arr)
        in_rim |= np.isin(f[:, 1], rim_vert_arr)
        in_rim |= np.isin(f[:, 2], rim_vert_arr)
    else:
        in_rim = np.zeros(n_faces, dtype=bool)
    z_scores = centroids[:, 2].astype(np.float64)
    z_scores[in_rim] -= 1e12
    seed_face = int(np.argmax(z_scores))
    seed_label = int(labels[seed_face])
    info["seed_face"] = seed_face
    info["seed_label"] = seed_label
    info["seed_centroid_z"] = float(centroids[seed_face, 2])

    keep_mask = labels == seed_label
    keep_idx = np.flatnonzero(keep_mask)
    info["kept_faces"] = int(len(keep_idx))
    if verbose:
        comp_sizes = np.bincount(labels)
        top5 = sorted(comp_sizes.tolist(), reverse=True)[:5]
        print(f"[recut] cut → {n_comp} components, top-5 sizes = {top5}")
        print(f"[recut] seed face {seed_face} (z={centroids[seed_face, 2]:.4f}) "
              f"in component {seed_label} "
              f"({comp_sizes[seed_label]:,} faces)")

    cut_mesh = canonical_mesh.submesh([keep_idx], append=True)
    if isinstance(cut_mesh, list):
        cut_mesh = cut_mesh[0]
    return cut_mesh, info
