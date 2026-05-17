"""Standalone PyVista renderer.

Run as a subprocess from `tire_assembly.py` because importing CadQuery in
the same process initialises OCP OpenGL state that turns subsequent
PyVista renders into solid black frames on llvmpipe / OSMesa.

Args (positional): <stl_path> <meta_json_path>
"""

import sys
import json
import os

import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def render_views(stl_path: str, meta_lines, output_dir: str,
                 preview_name: str = "tire_preview.png", keep_views: bool = False):
    pv.global_theme.background = 'white'
    mesh_pv = pv.read(stl_path)
    os.makedirs(output_dir, exist_ok=True)
    cx, cy, cz = mesh_pv.center
    # Bump the orbit radius up so the full tire fits in every frame —
    # 0.75 had the mesh overflowing the viewport.
    r = mesh_pv.length * 1.5

    use_pbr = "llvmpipe" not in pv.GPUInfo().renderer.lower()

    views = [
        {"name": "outboard_face",  "label": "Outboard Face",
         "camera_pos": (cx, cy, cz + r),                       "view_up": (0, 1, 0), "mode": "solid"},
        {"name": "side_profile",   "label": "Side Profile",
         "camera_pos": (cx + r, cy, cz),                       "view_up": (0, 0, 1), "mode": "solid"},
        {"name": "isometric",      "label": "Orthographic Corner",
         "camera_pos": (cx + r * 0.7, cy + r * 0.5, cz + r * 0.5),
         "view_up": (0, 0, 1), "mode": "solid"},
        {"name": "top_down",       "label": "Top Down",
         "camera_pos": (cx, cy + r, cz),                       "view_up": (1, 0, 0), "mode": "solid"},
        {"name": "cross_section",  "label": "Cross Section Outline (Y=0)",
         "camera_pos": (cx, cy + r, cz),                       "view_up": (1, 0, 0), "mode": "section_outline"},
    ]

    for view in views:
        pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
        pl.enable_parallel_projection()

        if view["mode"] == "section_outline":
            section = mesh_pv.slice(normal='y', origin=mesh_pv.center)
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
            pl.add_mesh(
                display_mesh,
                color='#111827',
                line_width=4,
                render_lines_as_tubes=True,
                lighting=False,
            )
        elif use_pbr:
            pl.add_mesh(display_mesh,
                        color='#b0b8c1',
                        pbr=True, metallic=0.75, roughness=0.35,
                        smooth_shading=True)
            pl.add_light(pv.Light(position=(r, r, r * 1.5),
                                  focal_point=(cx, cy, cz), intensity=1.2))
            pl.add_light(pv.Light(position=(-r, -r * 0.5, r),
                                  focal_point=(cx, cy, cz), intensity=0.4))
        else:
            pl.add_mesh(display_mesh,
                        color='#b0b8c1',
                        smooth_shading=True,
                        specular=0.4, specular_power=15,
                        ambient=0.25, diffuse=0.85)

        pl.camera.position    = view["camera_pos"]
        pl.camera.focal_point = (fx, fy, fz)
        pl.camera.up          = view["view_up"]
        pl.camera.reset_clipping_range()
        pl.camera.zoom(0.85)   # leave a margin around the mesh
        pl.add_text(view["label"], position='upper_left', font_size=14, color='black')
        image_path = os.path.join(output_dir, f"view_{view['name']}.png")
        pl.screenshot(image_path, transparent_background=False)
        pl.close()
        print(f"  saved {os.path.basename(image_path)}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    image_names = [
        "outboard_face", "side_profile", "isometric",
        "top_down",      "cross_section",
    ]
    for i, name in enumerate(image_names):
        row, col = divmod(i, 3)
        img = mpimg.imread(os.path.join(output_dir, f"view_{name}.png"))
        axes[row][col].imshow(img)
        axes[row][col].axis('off')

    axes[1][2].axis('off')
    axes[1][2].text(0.05, 0.95, "\n".join(meta_lines),
                    transform=axes[1][2].transAxes,
                    fontsize=11, verticalalignment='top',
                    fontfamily='monospace')

    plt.suptitle("Tire Assembly  Parametric Preview", fontsize=16)
    plt.tight_layout()
    preview_path = os.path.join(output_dir, preview_name)
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  composite saved: {os.path.basename(preview_path)}")

    if not keep_views:
        for name in image_names:
            os.remove(os.path.join(output_dir, f"view_{name}.png"))


if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    render_views(
        payload["stl_path"],
        payload["meta_lines"],
        payload["output_dir"],
        payload.get("preview_name", "tire_preview.png"),
        payload.get("keep_views", False),
    )
