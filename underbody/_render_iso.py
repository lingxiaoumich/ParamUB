"""Render a single isometric PyVista view of an STL file.

Subprocess entry: takes a JSON payload with stl_path + output_path.
Same camera framing as the 'isometric' view in _render_views.py, just
emitted as one PNG instead of as part of the 5-view composite.
"""

from __future__ import annotations

import json
import sys

import pyvista as pv


def render_iso(stl_path: str, output_path: str,
               size=(1400, 900)) -> None:
    pv.global_theme.background = "white"
    mesh = pv.read(stl_path)
    cx, cy, cz = mesh.center
    bx0, bx1, by0, by1, bz0, bz1 = mesh.bounds
    extent_xy = max(bx1 - bx0, by1 - by0)
    extent_z = bz1 - bz0
    r_xy = extent_xy * 0.95
    r_z = max(extent_z, extent_xy * 0.35) * 0.95

    use_pbr = "llvmpipe" not in pv.GPUInfo().renderer.lower()

    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.enable_parallel_projection()
    if use_pbr:
        pl.add_mesh(mesh, color="#b0b8c1", pbr=True,
                    metallic=0.7, roughness=0.45, smooth_shading=True)
        pl.add_light(pv.Light(position=(r_xy, r_xy, r_z * 2.0),
                              focal_point=(cx, cy, cz), intensity=1.2))
        pl.add_light(pv.Light(position=(-r_xy, -r_xy * 0.5, r_z),
                              focal_point=(cx, cy, cz), intensity=0.4))
    else:
        pl.add_mesh(mesh, color="#b0b8c1", smooth_shading=True,
                    specular=0.35, specular_power=15,
                    ambient=0.25, diffuse=0.85)

    pl.camera.position = (cx + r_xy, cy + r_xy * 0.6, cz + r_z * 1.2)
    pl.camera.focal_point = (cx, cy, cz)
    pl.camera.up = (0, 0, 1)
    pl.camera.reset_clipping_range()
    pl.camera.zoom(0.85)
    pl.screenshot(output_path, transparent_background=False)
    pl.close()


if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    render_iso(payload["stl_path"], payload["output_path"])
