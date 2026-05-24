"""Stage 4 of the reconstructed-car pipeline: turn the
:mod:`fill_gap`-combined STL into a strictly watertight, bilaterally
symmetric, single-body STL ready for CFD / 3-D printing / volume
computation.

Pipeline (all in one ``paramub`` env, optionally in a single process):

  STAGE A — Blender (bpy 4.2):
    1. import STL
    2. Smooth Remesh modifier (dual-contouring, octree depth ``depth``)
    3. keep largest connected component
    4. merge-by-distance, dissolve-degenerate, fill_holes, normals consistent
    5. symmetrize at y=0 (NEGATIVE_Y)
    6. iterative non-manifold edge cleanup (max ``max_nm_passes``)
    7. keep largest connected component again
    8. export STL  (intermediate ``_fixed.stl``)

  STAGE B — trimesh:
    9. split STL by connected components
   10. keep the largest body (drops Blender STL float32 micro-strays)
   11. verify watertight + record metrics
   12. export STL (final ``_clean.stl``) + JSON report

Why smooth dual-contouring as the core:
  Smooth Remesh produces a closed iso-surface from any input triangle
  soup, and merges nearby disconnected parts when they sit within ~bbox
  / 2^depth of each other. The symmetrize pass folds left/right
  asymmetries and provides a free Y-symmetry guarantee; the
  delete-non-manifold-and-refill loop handles the few 3-face edges the
  remesh leaves behind.

Depth / fidelity / RAM (measured on a ~4.5 m car, ~1.09 M input tris):

  depth   raw tris   raw error   clean tris   clean error  approx peak RAM
   8       198 K     0.53 mm     199 K        0.82 mm      ~4 GB
   9       798 K     0.16 mm     ≈800 K       ≈0.5 mm      ~8 GB
  10       3.2 M     0.045 mm    ≈3.2 M       ≈0.4 mm      ~24 GB
  11       12.8 M    0.014 mm    12.8 M       0.38 mm      ~96 GB

Depth 11 is the recommended default and needs a Slurm node (see
``make_watertight.sbatch`` for the 128 GB / 8 CPU / 3 h wrapper).

Public API
----------
:func:`make_watertight` — single-call high-level entrypoint that runs
the full pipeline and writes ``<stem>_fixed.stl`` + ``<stem>_clean.stl``
+ ``<stem>_clean.json``. Used by :mod:`make_watertight` CLI.

Stage A (bpy):
  :func:`smooth_remesh`           full Blender pipeline (steps 1-8)

Stage B (trimesh):
  :func:`extract_biggest_body`    step 9-10
  :func:`verify_watertight`       step 11 (builds a report dict)
  :func:`write_report`            JSON helper

bpy is imported lazily inside :func:`smooth_remesh` so importing this
module from a context that doesn't need the Blender stage (e.g. a small
post-hoc verify script) stays cheap.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Stage A — Blender (bpy 4.2)
# ---------------------------------------------------------------------------

def _bpy_modules():
    """Import bpy + bmesh lazily so this module is cheap to import in
    contexts that only need the trimesh-side helpers."""
    import bpy
    import bmesh
    return bpy, bmesh


def _reset_scene():
    bpy, _ = _bpy_modules()
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_stl(path: str):
    bpy, _ = _bpy_modules()
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _export_stl(path: str):
    bpy, _ = _bpy_modules()
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=False,
                              ascii_format=False)
    else:
        bpy.ops.export_mesh.stl(filepath=path, use_selection=False, ascii=False)


def _stats(obj, tag: str):
    """Print ``(non_manifold_edges, boundary_edges, non_manifold_verts)``
    and return the tuple."""
    _, bmesh = _bpy_modules()
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    nm_edges = sum(1 for e in bm.edges if not e.is_manifold)
    boundary_edges = sum(1 for e in bm.edges if len(e.link_faces) < 2)
    nm_verts = sum(1 for v in bm.verts if not v.is_manifold)
    print(f"[stats:{tag}] faces={len(me.polygons)} verts={len(me.vertices)} "
          f"non_manifold_edges={nm_edges} boundary_edges={boundary_edges} "
          f"non_manifold_verts={nm_verts}", flush=True)
    bm.free()
    return nm_edges, boundary_edges, nm_verts


def _edit_op(obj, fn):
    """Run ``fn`` inside Edit mode with everything selected."""
    bpy, _ = _bpy_modules()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    fn()
    bpy.ops.object.mode_set(mode="OBJECT")


def _keep_largest_component(obj):
    """Delete every connected component except the largest (by face count)."""
    bpy, bmesh = _bpy_modules()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    seen = [False] * len(bm.faces)
    components = []
    for i, f in enumerate(bm.faces):
        if seen[i]:
            continue
        stack, comp = [f], []
        while stack:
            cur = stack.pop()
            if seen[cur.index]:
                continue
            seen[cur.index] = True
            comp.append(cur)
            for e in cur.edges:
                for nb in e.link_faces:
                    if not seen[nb.index]:
                        stack.append(nb)
        components.append(comp)
    components.sort(key=len, reverse=True)
    print(f"[keep] {len(components)} components; top10 "
          f"{[len(c) for c in components[:10]]}", flush=True)
    if len(components) >= 2:
        biggest_idx = set(f.index for f in components[0])
        bpy.ops.mesh.select_all(action="DESELECT")
        for f in bm.faces:
            if f.index not in biggest_idx:
                f.select = True
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.delete(type="FACE")
    bpy.ops.object.mode_set(mode="OBJECT")


def _smooth_remesh_modifier(obj, depth: int, scale: float):
    bpy, _ = _bpy_modules()
    mod = obj.modifiers.new(name="remesh", type="REMESH")
    mod.mode = "SMOOTH"
    mod.octree_depth = depth
    mod.scale = scale
    mod.use_remove_disconnected = False
    mod.use_smooth_shade = False
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


# Effectively-unlimited cap for Blender's fill_holes inside the NM-fix
# loop. The default `fill_sides=16` was way too small — it silently
# skipped any hole with more than 64 boundary edges (e.g. the elongated
# cowl-gutter slots Blender's dual-contour remesh sometimes opens).
# Earcut runtime is O(N²) per loop and we process each loop once per
# pass, so 100 k is "no practical cap" without being insane.
_NM_FILL_HOLES_MAX_SIDES = 100_000


def _fix_nonmanifold_iter(obj, merge_distance: float, fill_sides: int,
                            max_passes: int) -> int:
    """Loop: select-non-manifold → delete those faces → fill holes →
    cleanup. Returns the pass count at convergence (or ``max_passes``).

    ``fill_sides`` here is ignored for the loop's fill_holes call — we
    use a very large cap (``_NM_FILL_HOLES_MAX_SIDES``) so big holes
    aren't silently skipped. ``fill_sides`` still controls the
    non-loop fill_holes calls (one-shot cleanups elsewhere)."""
    bpy, _ = _bpy_modules()
    for i in range(max_passes):
        print(f"[fix-nm] pass {i+1}", flush=True)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_non_manifold(
            extend=False, use_wire=True, use_boundary=True,
            use_multi_face=True, use_non_contiguous=True, use_verts=True)
        bpy.ops.mesh.select_more()
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=_NM_FILL_HOLES_MAX_SIDES)
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        nm_e, b_e, nm_v = _stats(obj, f"nm{i+1}")
        if nm_e == 0 and b_e == 0 and nm_v == 0:
            print(f"[fix-nm] converged after {i+1} pass(es)", flush=True)
            return i + 1
    print(f"[fix-nm] cap of {max_passes} passes reached without convergence",
          flush=True)
    return max_passes


def lightweight_symmetrize(input_stl: str | Path, output_stl: str | Path, *,
                              symmetry_direction: str = "NEGATIVE_Y",
                              merge_distance: float = 1e-3,
                              fill_sides: int = 16) -> None:
    """Lightweight Blender pass — symmetrize an already-watertight STL
    without dual-contour remesh.

    Used when stage 3's combined.stl is already
    ``is_watertight=True`` (which is the normal outcome of
    ``fill_gap.py --repair --repair-remesh-mm 6``). In that case the
    only thing left to add is bilateral symmetry; we don't need the
    heavy smooth-remesh that erases sub-voxel features like the cowl
    gutter and triggers the non-manifold-fix divergence.

    Pipeline:
      1. import STL
      2. symmetrize at y=0  (mirrors the source side to the other,
         welds the seam within ``merge_distance``)
      3. fill_holes / merge / normals consistent
      4. export STL

    Returns the time spent in seconds.
    """
    bpy, _ = _bpy_modules()
    input_stl = str(input_stl)
    output_stl = str(output_stl)
    os.makedirs(os.path.dirname(output_stl) or ".", exist_ok=True)
    t0 = time.time()

    _reset_scene()
    print(f"[lite] import {input_stl}", flush=True)
    obj = _import_stl(input_stl)
    _stats(obj, "lite_in")

    print(f"[lite] symmetrize {symmetry_direction} "
          f"threshold={merge_distance}", flush=True)
    _edit_op(obj, lambda: bpy.ops.mesh.symmetrize(
        direction=symmetry_direction, threshold=merge_distance))
    _edit_op(obj, lambda: bpy.ops.mesh.remove_doubles(
        threshold=merge_distance))
    _edit_op(obj, lambda: bpy.ops.mesh.fill_holes(sides=fill_sides))
    _edit_op(obj, lambda: bpy.ops.mesh.normals_make_consistent(
        inside=False))
    _stats(obj, "lite_out")

    print(f"[lite] export {output_stl}", flush=True)
    _export_stl(output_stl)
    print(f"[lite] stage A (lightweight) total {time.time() - t0:.1f}s",
          flush=True)


def smooth_remesh(input_stl: str | Path, output_stl: str | Path, *,
                    depth: int = 11,
                    scale: float = 0.99,
                    symmetrize: bool = True,
                    symmetry_direction: str = "NEGATIVE_Y",
                    fix_nonmanifold: bool = True,
                    max_nm_passes: int = 8,
                    merge_distance: float = 1e-3,
                    fill_sides: int = 16) -> None:
    """Run the full Blender stage end-to-end on ``input_stl`` and write
    ``output_stl``.

    Parameters
    ----------
    depth:
        Octree depth for the Smooth Remesh modifier (2^depth cells across
        the longest bbox axis). Memory/time scale ~8x per +1. Default 11
        gives ~2 mm/cell on a 4.5 m car.
    scale:
        Remesh modifier grid scale (0..0.99). Lower = more bbox padding.
    symmetrize / symmetry_direction:
        Bisect at y=0 and mirror. ``NEGATIVE_Y`` keeps y<0 and mirrors
        to y>0. Other axes / signs accepted (see Blender's
        ``mesh.symmetrize`` docs). Pass ``symmetrize=False`` to skip.
    fix_nonmanifold / max_nm_passes:
        Iterative delete-NM-edges-then-refill loop. Default cap of 20
        passes; loop exits early on convergence.
    merge_distance:
        Vertex merge threshold (mm). Used for ``remove_doubles``,
        ``symmetrize`` seam weld, and ``dissolve_degenerate``.
    fill_sides:
        Max hole size (edges) for ``fill_holes``. The non-manifold-fix
        loop uses ``4*fill_sides`` for the larger holes it creates.
    """
    bpy, _ = _bpy_modules()
    input_stl = str(input_stl)
    output_stl = str(output_stl)
    os.makedirs(os.path.dirname(output_stl) or ".", exist_ok=True)
    t0 = time.time()

    _reset_scene()
    print(f"[step] import {input_stl}", flush=True)
    obj = _import_stl(input_stl)
    _stats(obj, "in")

    print(f"[step] smooth remesh depth={depth} scale={scale}", flush=True)
    t1 = time.time()
    _smooth_remesh_modifier(obj, depth, scale)
    print(f"[step] remesh done in {time.time() - t1:.1f}s", flush=True)
    _stats(obj, "remesh")

    print("[step] keep-largest (post-remesh)", flush=True)
    _keep_largest_component(obj)
    _stats(obj, "kept_pre")

    print(f"[step] merge-by-distance {merge_distance}", flush=True)
    _edit_op(obj, lambda: bpy.ops.mesh.remove_doubles(threshold=merge_distance))

    print("[step] dissolve-degenerate", flush=True)
    _edit_op(obj, lambda: bpy.ops.mesh.dissolve_degenerate(threshold=merge_distance))

    print(f"[step] fill_holes sides={fill_sides}", flush=True)
    _edit_op(obj, lambda: bpy.ops.mesh.fill_holes(sides=fill_sides))

    print("[step] normals_make_consistent", flush=True)
    _edit_op(obj, lambda: bpy.ops.mesh.normals_make_consistent(inside=False))
    _stats(obj, "cleaned")

    if symmetrize:
        print(f"[step] symmetrize {symmetry_direction} threshold={merge_distance}",
              flush=True)
        _edit_op(obj, lambda: bpy.ops.mesh.symmetrize(
            direction=symmetry_direction, threshold=merge_distance))
        _edit_op(obj, lambda: bpy.ops.mesh.remove_doubles(threshold=merge_distance))
        _edit_op(obj, lambda: bpy.ops.mesh.fill_holes(sides=fill_sides))
        _edit_op(obj, lambda: bpy.ops.mesh.normals_make_consistent(inside=False))
        _stats(obj, "sym")

    if fix_nonmanifold:
        _fix_nonmanifold_iter(obj, merge_distance, fill_sides, max_nm_passes)

    print("[step] keep-largest (post-fix)", flush=True)
    _keep_largest_component(obj)
    _edit_op(obj, lambda: bpy.ops.mesh.normals_make_consistent(inside=False))
    _stats(obj, "out")

    print(f"[step] export {output_stl}", flush=True)
    _export_stl(output_stl)
    print(f"[step] stage A total {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage B — trimesh
# ---------------------------------------------------------------------------

def extract_biggest_body(mesh):
    """Return the largest connected component of ``mesh`` (by face count).

    Blender's STL exporter writes each face as an independent (v0, v1, v2)
    triple in float32. When trimesh re-loads the STL and merges vertices
    by position, a small number of vertices that Blender considered shared
    end up split (float32 precision loss). Those splits leave 2-6 face
    micro-bodies that an ``is_watertight`` check then fails on. Keeping
    only the biggest body reliably strips those strays without touching
    the real surface.
    """
    bodies = sorted(mesh.split(only_watertight=False),
                    key=lambda b: len(b.faces), reverse=True)
    if not bodies:
        raise RuntimeError("Input mesh has no faces.")
    print(f"[extract] {len(bodies)} bodies; top10 "
          f"{[len(b.faces) for b in bodies[:10]]}", flush=True)
    biggest = bodies[0]
    print(f"[extract] biggest: faces={len(biggest.faces)}  "
          f"is_watertight={biggest.is_watertight}  "
          f"winding={biggest.is_winding_consistent}  "
          f"euler={biggest.euler_number}", flush=True)
    return biggest


def verify_watertight(mesh, reference_path: str | Path | None = None,
                       n_samples: int = 30_000) -> dict[str, Any]:
    """Build a JSON-serializable verify report for ``mesh``.

    The report always includes face/vertex counts, ``is_watertight``,
    ``is_winding_consistent``, ``euler_number``, bbox + extents, and
    surface area in m². When ``is_watertight`` is True the report also
    includes ``volume_m3``.

    When ``reference_path`` is given, the report also includes symmetric
    Hausdorff-style point-to-surface distance metrics (mean / p95 / max,
    in mm) computed from ``n_samples`` surface samples per side:

      * ``dist_remesh_to_orig_mm`` — sample ``mesh``, query nearest on ref.
      * ``dist_orig_to_remesh_mm`` — sample ref, query nearest on ``mesh``.
    """
    import numpy as np
    import trimesh

    report: dict[str, Any] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "bbox_min": mesh.bounds[0].tolist(),
        "bbox_max": mesh.bounds[1].tolist(),
        "extents": mesh.extents.tolist(),
        "surface_area_m2": float(mesh.area * 1e-6),
    }
    if mesh.is_watertight:
        report["volume_m3"] = float(mesh.volume * 1e-9)

    if reference_path is not None and Path(reference_path).exists():
        try:
            import rtree  # noqa: F401  — needed by trimesh.proximity
        except ImportError:
            print(f"[verify] surface-distance metric skipped: "
                  f"rtree not installed (`pip install rtree`)",
                  flush=True)
        else:
            print(f"[verify] sampling {n_samples:,} pts/side against "
                  f"{reference_path}", flush=True)
            ref = trimesh.load(str(reference_path), force="mesh")
            pts_a, _ = trimesh.sample.sample_surface(mesh, n_samples)
            pts_b, _ = trimesh.sample.sample_surface(ref, n_samples)
            _, d_ab, _ = trimesh.proximity.closest_point(ref, pts_a)
            _, d_ba, _ = trimesh.proximity.closest_point(mesh, pts_b)
            report["dist_remesh_to_orig_mm"] = {
                "mean": float(np.mean(d_ab)),
                "p95":  float(np.percentile(d_ab, 95)),
                "max":  float(np.max(d_ab)),
            }
            report["dist_orig_to_remesh_mm"] = {
                "mean": float(np.mean(d_ba)),
                "p95":  float(np.percentile(d_ba, 95)),
                "max":  float(np.max(d_ba)),
            }

    return report


def write_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a verify report to ``path`` as pretty-printed JSON. Also
    echoes the report to stdout."""
    txt = json.dumps(report, indent=2)
    print(txt, flush=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(txt)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

WHEEL_CORNERS = ("front_left", "front_right", "rear_left", "rear_right")


def discover_wheel_paths(combined_input: str | Path,
                          wheel_corners: tuple[str, ...] = WHEEL_CORNERS,
                          ) -> dict[str, Path]:
    """Find the 4 wheel STLs that belong to ``combined_input``.

    Convention: the combined STL is named ``<base>_combined.stl`` and
    the wheels are named ``<base>_wheel_<corner>.stl`` in the same
    directory (this is how :mod:`integrate_underbody` writes them).
    Returns a dict ``{corner: path}`` for whichever wheel files exist.
    """
    path = Path(combined_input)
    stem = path.stem
    if stem.endswith("_combined"):
        base = stem[: -len("_combined")]
    else:
        base = stem
    found: dict[str, Path] = {}
    for corner in wheel_corners:
        cand = path.with_name(f"{base}_wheel_{corner}.stl")
        if cand.exists():
            found[corner] = cand
    return found


def make_wheel_watertight(input_stl: str | Path,
                            clean_stl: str | Path,
                            report_json: str | Path | None = None,
                            reference_stl: str | Path | None = None,
                            n_samples: int = 30_000,
                            ) -> dict[str, Any]:
    """Pure-trimesh watertight finisher for a single wheel STL.

    Each CadQuery-tessellated wheel is single-component and consistently
    wound but typically has a small number of open edges from the
    chord-discretization on curved tire surfaces. ``trimesh`` can close
    them in-place with ``repair.fill_holes`` — no Blender remesh needed
    (the wheel is already a clean dual-shell-free surface).

    Pipeline:
      1. load and merge close vertices
      2. fix winding if needed
      3. ``trimesh.repair.fill_holes`` (closes small open boundaries)
      4. take biggest body (drops any micro-strays)
      5. verify watertight + write report
    """
    import trimesh
    from trimesh.repair import fill_holes, fix_normals

    input_stl = str(input_stl)
    clean_stl = str(clean_stl)
    Path(clean_stl).parent.mkdir(parents=True, exist_ok=True)

    print(f"[wheel] load {input_stl}", flush=True)
    m = trimesh.load(input_stl, force="mesh", process=True)
    print(f"[wheel] in: f={len(m.faces):,}  v={len(m.vertices):,}  "
          f"watertight={m.is_watertight}  winding={m.is_winding_consistent}",
          flush=True)

    if not m.is_winding_consistent:
        fix_normals(m)

    if not m.is_watertight:
        fill_holes(m)
        print(f"[wheel] after fill_holes: f={len(m.faces):,}  "
              f"watertight={m.is_watertight}  "
              f"winding={m.is_winding_consistent}", flush=True)

    biggest = extract_biggest_body(m)
    biggest.export(clean_stl)
    size_kb = os.path.getsize(clean_stl) // 1024
    print(f"[wheel] exported {clean_stl} ({size_kb} KB)", flush=True)

    report = verify_watertight(
        biggest, reference_path=reference_stl, n_samples=n_samples)
    report["file"] = clean_stl
    if report_json is not None:
        write_report(report, report_json)
    return report


def _input_is_watertight(input_stl: str | Path) -> bool:
    """Quick load+check whether the input STL is already
    ``is_watertight=True``. Used by :func:`make_watertight` to pick
    between the lightweight and full Blender pipelines."""
    import trimesh
    m = trimesh.load(str(input_stl), force="mesh", process=True)
    return bool(m.is_watertight)


def make_watertight(input_stl: str | Path, *,
                      fixed_stl: str | Path,
                      clean_stl: str | Path,
                      report_json: str | Path | None = None,
                      reference_stl: str | Path | None = None,
                      n_samples: int = 30_000,
                      mode: str = "auto",
                      **remesh_kwargs) -> dict[str, Any]:
    """Run both stages of the watertight pipeline end-to-end.

    Stage A (Blender) reads ``input_stl`` and writes ``fixed_stl``.
    Stage B (trimesh) reads ``fixed_stl``, extracts the biggest body,
    writes ``clean_stl``, and returns a verify report dict.

    ``mode`` selects the stage-A strategy:

      * ``"auto"``  (default) — load the input, check
        ``is_watertight``. If True, use ``"lightweight"``; else use
        ``"full"``. This handles the common case where stage 3's
        ``fill_gap.py --repair --repair-remesh-mm 6`` already produced
        a watertight mesh and we only need to add symmetry, without
        paying the cost (or feature-erosion risk) of the smooth
        dual-contour remesh.
      * ``"lightweight"`` — :func:`lightweight_symmetrize` only.
        Preserves all input detail; cheap; assumes input is already
        watertight or close to it.
      * ``"full"`` — :func:`smooth_remesh` (octree dual-contouring +
        symmetrize + iterative NM-fix). Needed when input has many
        open boundary edges or non-manifold defects that trimesh-side
        repair can't handle.

    When ``report_json`` is given, the report is also written there.
    When ``reference_stl`` is given, the report includes surface
    distance metrics against it (typically the original input).

    Any extra keyword arguments are forwarded to whichever stage-A
    function runs (``depth``, ``symmetrize``, ``max_nm_passes`` for
    full; ``symmetry_direction``, ``fill_sides``, ``merge_distance``
    for both).
    """
    import trimesh

    fixed_stl = str(fixed_stl)
    clean_stl = str(clean_stl)
    Path(clean_stl).parent.mkdir(parents=True, exist_ok=True)

    if mode == "auto":
        wt = _input_is_watertight(input_stl)
        mode = "lightweight" if wt else "full"
        print(f"[stage A] input is_watertight={wt} → mode={mode}", flush=True)

    if mode == "lightweight":
        # Forward only the kwargs lightweight_symmetrize understands.
        lite_kwargs = {
            k: v for k, v in remesh_kwargs.items()
            if k in ("symmetry_direction", "merge_distance", "fill_sides")
        }
        if remesh_kwargs.get("symmetrize") is False:
            print(f"[stage A] lightweight: symmetrize=False requested; "
                  f"skipping symmetrize and just copying input as fixed",
                  flush=True)
            import shutil
            shutil.copyfile(str(input_stl), fixed_stl)
        else:
            lightweight_symmetrize(input_stl, fixed_stl, **lite_kwargs)
    elif mode == "full":
        smooth_remesh(input_stl, fixed_stl, **remesh_kwargs)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected "
                         f"'auto' | 'lightweight' | 'full'")

    print(f"[stage B] load {fixed_stl}", flush=True)
    fixed = trimesh.load(fixed_stl, force="mesh")
    biggest = extract_biggest_body(fixed)
    biggest.export(clean_stl)
    size_mb = os.path.getsize(clean_stl) // 1024 // 1024
    print(f"[stage B] exported {clean_stl} ({size_mb} MB)", flush=True)

    report = verify_watertight(
        biggest, reference_path=reference_stl, n_samples=n_samples)
    report["file"] = clean_stl
    if report_json is not None:
        write_report(report, report_json)
    return report
