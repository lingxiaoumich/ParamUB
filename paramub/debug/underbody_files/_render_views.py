"""Standalone PyVista renderer for the underbody assembly.

Same 5-view contract as ParamUB's _render_views.py (outboard, side,
isometric, top, Y=0 cross-section) but with car-scaled camera framing
and a bottom-up "underside" view in place of the wheel "outboard" view.

Runs as a subprocess from underbody/build.py — must not import CadQuery
(OCP leaves OpenGL state that turns subsequent off-screen renders black
on llvmpipe / OSMesa).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pyvista as pv


def render_views(stl_path: str, meta_lines, output_dir: str,
                 preview_name: str = "underbody_preview.png",
                 keep_views: bool = True):
    pv.global_theme.background = "white"
    mesh_pv = pv.read(stl_path)
    os.makedirs(output_dir, exist_ok=True)

    cx, cy, cz = mesh_pv.center
    bx0, bx1, by0, by1, bz0, bz1 = mesh_pv.bounds
    extent_x = bx1 - bx0
    extent_y = by1 - by0
    extent_z = bz1 - bz0

    # Frame from outside the largest extent so the whole car fits.
    r_xy = max(extent_x, extent_y) * 0.95
    r_z = max(extent_z, max(extent_x, extent_y) * 0.35) * 0.95

    use_pbr = "llvmpipe" not in pv.GPUInfo().renderer.lower()

    views = [
        {"name": "underside",       "label": "Underside (looking up)",
         "camera_pos": (cx, cy, bz0 - r_z),    "view_up": (1, 0, 0), "mode": "solid"},
        {"name": "side_profile",    "label": "Side Profile (+Y)",
         "camera_pos": (cx, by1 + r_xy, cz),   "view_up": (0, 0, 1), "mode": "solid"},
        {"name": "isometric",       "label": "Isometric",
         "camera_pos": (cx + r_xy, cy + r_xy * 0.6, cz + r_z * 1.2),
         "view_up": (0, 0, 1), "mode": "solid"},
        {"name": "top_down",        "label": "Top Down",
         "camera_pos": (cx, cy, bz1 + r_z),    "view_up": (1, 0, 0), "mode": "solid"},
        {"name": "cross_section",   "label": "Cross Section Outline (Y=0)",
         "camera_pos": (cx, by1 + r_xy, cz),   "view_up": (0, 0, 1), "mode": "section_outline"},
    ]

    for view in views:
        pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
        pl.enable_parallel_projection()

        if view["mode"] == "section_outline":
            section = mesh_pv.slice(normal="y", origin=mesh_pv.center)
            outline = section.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            )
            display_mesh = outline if outline.n_points else section
            fx, fy, fz = display_mesh.center
        else:
            display_mesh = mesh_pv
            fx, fy, fz = cx, cy, cz

        if view["mode"] == "section_outline":
            pl.add_mesh(display_mesh, color="#111827", line_width=4,
                        render_lines_as_tubes=True, lighting=False)
        elif use_pbr:
            pl.add_mesh(display_mesh, color="#b0b8c1", pbr=True,
                        metallic=0.7, roughness=0.45, smooth_shading=True)
            pl.add_light(pv.Light(position=(r_xy, r_xy, r_z * 2.0),
                                  focal_point=(cx, cy, cz), intensity=1.2))
            pl.add_light(pv.Light(position=(-r_xy, -r_xy * 0.5, r_z),
                                  focal_point=(cx, cy, cz), intensity=0.4))
        else:
            pl.add_mesh(display_mesh, color="#b0b8c1", smooth_shading=True,
                        specular=0.35, specular_power=15,
                        ambient=0.25, diffuse=0.85)

        pl.camera.position = view["camera_pos"]
        pl.camera.focal_point = (fx, fy, fz)
        pl.camera.up = view["view_up"]
        pl.camera.reset_clipping_range()
        pl.camera.zoom(0.85)
        pl.add_text(view["label"], position="upper_left",
                    font_size=14, color="black")

        image_path = os.path.join(output_dir, f"view_{view['name']}.png")
        pl.screenshot(image_path, transparent_background=False)
        pl.close()
        print(f"  saved {os.path.basename(image_path)}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    image_names = [
        "underside", "side_profile", "isometric",
        "top_down",  "cross_section",
    ]
    for i, name in enumerate(image_names):
        row, col = divmod(i, 3)
        img = mpimg.imread(os.path.join(output_dir, f"view_{name}.png"))
        axes[row][col].imshow(img)
        axes[row][col].axis("off")

    axes[1][2].axis("off")
    axes[1][2].text(0.05, 0.95, "\n".join(meta_lines),
                    transform=axes[1][2].transAxes, fontsize=11,
                    verticalalignment="top", fontfamily="monospace")

    plt.suptitle("Underbody Assembly — Parametric Preview", fontsize=16)
    plt.tight_layout()
    preview_path = os.path.join(output_dir, preview_name)
    plt.savefig(preview_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  composite saved: {os.path.basename(preview_path)}")

    if not keep_views:
        for name in image_names:
            try:
                os.remove(os.path.join(output_dir, f"view_{name}.png"))
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    render_views(
        payload["stl_path"],
        payload["meta_lines"],
        payload["output_dir"],
        payload.get("preview_name", "underbody_preview.png"),
        payload.get("keep_views", True),
    )
