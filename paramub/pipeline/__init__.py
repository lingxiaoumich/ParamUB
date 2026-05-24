"""Post-trim pipeline finishers.

Modules:

  watertight    Stage 4 — Blender (bpy 4.2) smooth-remesh + symmetrize
                + iterative non-manifold fix (full mode) OR Blender
                symmetrize only (lightweight mode for already-watertight
                inputs), then trimesh-side biggest-body extract + verify.
                Also runs a fast trimesh-only fill_holes pass on the
                4 adjacent wheel STLs.
  summary       Consolidated <base>_summary.json writer — bundles every
                geometry knob the parametric UB used (wheel centres,
                tire spec, wheelhouse heights, splitter + diffuser
                sections, floor z) with the geometric outputs of the
                final watertight result (per-part dimensions, triangle
                counts, watertight flags, volumes, totals).

These modules don't depend on cadquery — they're trimesh / bpy based.
"""
