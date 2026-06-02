"""Final step: trim/merge the front tire deflectors into the FINISHED
watertight body, via a Blender (bpy) boolean UNION.

Runs AFTER stage 4 (make_watertight). The clean body and the deflector
plates are both closed watertight solids; the plate top was built 50 mm
ABOVE the floor on purpose, so a boolean UNION absorbs the overlapping top
into the body and leaves the 40 mm fin protruding below — i.e. it "trims
the deflector in" and yields a single watertight body.

Boolean engine: Blender as a Python module (``bpy``), the same engine the
stage-4 watertight finisher uses (no trimesh boolean backend / manifold3d
is installed in this env, and the body is ~14 M triangles, far too large to
union on the login node). MUST run on a compute node.

Usage (inside the paramub conda env, on a Slurm node)::

    python trim_deflector.py \
        --body      outputs/<car>_v4/integrate/<car>_clean.stl \
        --deflector outputs/<car>_v4/integrate/<car>_deflector.stl \
        --out       outputs/<car>_v4/integrate/<car>_clean_deflector.stl \
        --solver FAST
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _stl_import(filepath: str):
    import bpy
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=filepath)
    else:
        bpy.ops.import_mesh.stl(filepath=filepath)


def _stl_export(filepath: str):
    import bpy
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=filepath, export_selected_objects=True)
    else:
        bpy.ops.export_mesh.stl(filepath=filepath, use_selection=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", type=Path, required=True)
    p.add_argument("--deflector", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--solver", choices=("FAST", "EXACT"), default="FAST",
                   help="Blender boolean solver (FAST cheap, EXACT robust).")
    args = p.parse_args()

    for f in (args.body, args.deflector):
        if not f.is_file():
            print(f"ERROR: missing input: {f}", file=sys.stderr)
            return 2

    import bpy

    t_all = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    t0 = time.time()
    _stl_import(str(args.body.resolve()))
    body = bpy.context.selected_objects[0]
    body.name = "body"
    print(f"[trim] imported body in {time.time()-t0:.1f}s  "
          f"verts={len(body.data.vertices):,}", flush=True)

    t0 = time.time()
    _stl_import(str(args.deflector.resolve()))
    defl = bpy.context.selected_objects[0]
    defl.name = "deflector"
    print(f"[trim] imported deflector  verts={len(defl.data.vertices):,}  "
          f"in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    mod = body.modifiers.new(name="defl_union", type="BOOLEAN")
    mod.operation = "UNION"
    mod.object = defl
    mod.solver = args.solver
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"[trim] boolean UNION ({args.solver}) in {time.time()-t0:.1f}s  "
          f"verts={len(body.data.vertices):,}", flush=True)

    bpy.data.objects.remove(defl, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.context.view_layer.objects.active = body

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _stl_export(str(args.out.resolve()))
    print(f"[trim] exported {args.out} in {time.time()-t0:.1f}s", flush=True)
    print(f"[trim] total {time.time()-t_all:.1f}s", flush=True)

    try:
        import trimesh
        m = trimesh.load(str(args.out), force="mesh", process=True)
        print(f"[trim] result: {len(m.faces):,} faces  "
              f"watertight={m.is_watertight}  "
              f"components={len(m.split(only_watertight=False))}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[trim] (post-check skipped: {exc})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
