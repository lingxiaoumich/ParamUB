"""Upper-body shell extraction from a reconstructed whole-car mesh.

Modules:

  extract       canonicalize axes, remove wheels/wheelhouses/underbody,
                output a single connected upper-body shell
                (find_open_boundary_components etc.)
  recut         take the longest open boundary loop of the shell as a
                cutter and re-cut the canonical mesh (fills in small
                holes accidentally created by earlier steps)
  render        10-panel matplotlib renderer for the per-step shell
                diagnostics

These modules don't depend on cadquery — they're trimesh /
scipy / shapely based so they work in any env that has those.
"""
