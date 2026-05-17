"""Standalone PyVista renderer for the underbody STL.

Builds an 8-panel composite (one PNG) covering:

    iso · top · front elevation · Y=0 longitudinal section
    front-axle section · front-tire detail
    rear-axle section · rear-tire detail

Runs as a subprocess from :mod:`paramub.generate` because PyVista's
off-screen renderer goes blank on llvmpipe / OSMesa once CadQuery / OCP
has initialised OpenGL state in the parent process.

Receives a JSON payload as argv[1]:

    {"stl_path": ..., "output_dir": ..., "meta_lines": [...],
     "preview_name": "underbody_preview.png",
     "axles": {"front_axle_x": ..., "rear_axle_x": ...,
               "track_front_mm": ..., "track_rear_mm": ...,
               "tire_radius_f_mm": ..., "tire_radius_r_mm": ...}}
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


PANEL_SIZE = (900, 700)
PANEL_ASPECT = PANEL_SIZE[0] / PANEL_SIZE[1]   # width / height


def _new_plotter():
    pl = pv.Plotter(off_screen=True, window_size=PANEL_SIZE)
    pl.enable_parallel_projection()
    return pl


def _add_solid(pl, mesh, use_pbr, r_xy, r_z, cx, cy, cz):
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


def _section_outline(mesh, normal, origin):
    """Slice + extract feature edges. Returns the polydata to draw, or None."""
    section = mesh.slice(normal=normal, origin=origin)
    if section.n_points == 0:
        return None
    outline = section.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False,
    )
    return outline if outline.n_points else section


def _add_section_lines(pl, polydata, line_width=6):
    pl.add_mesh(polydata, color="#111827", line_width=line_width,
                render_lines_as_tubes=True, lighting=False)


def _fit_parallel(width_world, height_world, pad=1.15):
    """Pick a parallel_scale (= camera half-height in world units) that fits
    both dimensions inside the panel's aspect ratio."""
    half_h = height_world / 2.0
    half_w = width_world / 2.0
    fit_for_h = half_h
    fit_for_w = half_w / PANEL_ASPECT
    return max(fit_for_h, fit_for_w) * pad


def _shoot(pl, label):
    pl.add_text(label, position="upper_left", font_size=14, color="black")
    img = pl.screenshot(return_img=True, transparent_background=False)
    pl.close()
    return img


def _set_camera(pl, position, focal, up, parallel_scale=None):
    pl.camera.position = position
    pl.camera.focal_point = focal
    pl.camera.up = up
    pl.camera.reset_clipping_range()
    if parallel_scale is not None:
        pl.camera.parallel_scale = parallel_scale


def _render_axle_section(mesh, axle_x, *,
                          focal_y, focal_z, half_width_world,
                          half_height_world, label,
                          big_xy):
    """Render an X-normal section centered at (axle_x, focal_y, focal_z)
    with a parallel projection sized to (half_width_world, half_height_world)
    in (Y, Z)."""
    pl = _new_plotter()
    poly = _section_outline(mesh, normal="x",
                            origin=(axle_x, focal_y, focal_z))
    if poly is None:
        return _shoot(pl, label + " (no section)")
    _add_section_lines(pl, poly)
    scale = _fit_parallel(half_width_world * 2.0, half_height_world * 2.0)
    _set_camera(pl,
                (axle_x + big_xy, focal_y, focal_z),
                (axle_x, focal_y, focal_z),
                (0, 0, 1),
                parallel_scale=scale)
    return _shoot(pl, label)


def _render_panels(stl_path: str, axles: dict):
    pv.global_theme.background = "white"
    mesh = pv.read(stl_path)
    cx, cy, cz = mesh.center
    bx0, bx1, by0, by1, bz0, bz1 = mesh.bounds
    span_x = bx1 - bx0
    span_y = by1 - by0
    span_z = bz1 - bz0
    r_xy = max(span_x, span_y) * 0.95
    r_z = max(span_z, max(span_x, span_y) * 0.35) * 0.95
    big = max(span_x, span_y, span_z) * 2.0
    use_pbr = "llvmpipe" not in pv.GPUInfo().renderer.lower()

    front_x = axles["front_axle_x"]
    rear_x = axles["rear_axle_x"]
    track_f = axles["track_front_mm"]
    track_r = axles["track_rear_mm"]
    tire_r_f = axles["tire_radius_f_mm"]
    tire_r_r = axles["tire_radius_r_mm"]

    panels = []

    # 1. Isometric ----------------------------------------------------------
    pl = _new_plotter()
    _add_solid(pl, mesh, use_pbr, r_xy, r_z, cx, cy, cz)
    _set_camera(pl,
                (cx + r_xy, cy + r_xy * 0.6, cz + r_z * 1.2),
                (cx, cy, cz), (0, 0, 1),
                parallel_scale=_fit_parallel(span_x * 1.05, span_z + span_y))
    pl.camera.zoom(0.9)
    panels.append(_shoot(pl, "Isometric"))

    # 2. Top down — looking straight down; +X forward to the right of image.
    pl = _new_plotter()
    _add_solid(pl, mesh, use_pbr, r_xy, r_z, cx, cy, cz)
    _set_camera(pl, (cx, cy, bz1 + big),
                (cx, cy, cz), (0, 1, 0),
                parallel_scale=_fit_parallel(span_x, span_y))
    panels.append(_shoot(pl, "Top down (+X →)"))

    # 3. Front elevation (camera from +X) -----------------------------------
    pl = _new_plotter()
    _add_solid(pl, mesh, use_pbr, r_xy, r_z, cx, cy, cz)
    _set_camera(pl, (bx1 + big, cy, cz),
                (cx, cy, cz), (0, 0, 1),
                parallel_scale=_fit_parallel(span_y, span_z))
    panels.append(_shoot(pl, "Front elevation"))

    # 4. Y = 0 longitudinal section ----------------------------------------
    pl = _new_plotter()
    poly = _section_outline(mesh, normal="y", origin=(cx, 0, cz))
    if poly is not None:
        _add_section_lines(pl, poly, line_width=10)
        sx0, sx1, _sy0, _sy1, sz0, sz1 = poly.bounds
        # The body section is a wide-but-shallow curve. Pad vertical
        # extent to ~12% of horizontal so the line has visual weight.
        section_w = sx1 - sx0
        section_h = max(sz1 - sz0, section_w * 0.12)
        focal = ((sx0 + sx1) / 2.0, 0, (sz0 + sz1) / 2.0)
        scale = _fit_parallel(section_w, section_h, pad=1.05)
        _set_camera(pl, (focal[0], by1 + big, focal[2]),
                    focal, (0, 0, 1), parallel_scale=scale)
    panels.append(_shoot(pl, "Y = 0 longitudinal section"))

    # 5-8. Axle / tire sections --------------------------------------------
    # Frame "full axle" sections to floor_width + sky over the dome.
    # Use mesh's Y / Z span as proxies, biased down so the body + tire fit.
    full_half_w = span_y / 2.0
    full_half_h = max(span_z * 0.55, max(tire_r_f, tire_r_r) + 100.0)
    full_focal_z = bz0 + full_half_h  # vertically center wheels + arches

    panels.append(_render_axle_section(
        mesh, front_x,
        focal_y=0.0, focal_z=full_focal_z,
        half_width_world=full_half_w, half_height_world=full_half_h,
        label="Front axle section", big_xy=big,
    ))
    # Detail: zoom on right front wheel.
    panels.append(_render_axle_section(
        mesh, front_x,
        focal_y=+track_f / 2.0, focal_z=tire_r_f,
        half_width_world=tire_r_f * 1.35,
        half_height_world=tire_r_f * 1.35,
        label="Front tire detail (right)", big_xy=big,
    ))
    panels.append(_render_axle_section(
        mesh, rear_x,
        focal_y=0.0, focal_z=full_focal_z,
        half_width_world=full_half_w, half_height_world=full_half_h,
        label="Rear axle section", big_xy=big,
    ))
    panels.append(_render_axle_section(
        mesh, rear_x,
        focal_y=+track_r / 2.0, focal_z=tire_r_r,
        half_width_world=tire_r_r * 1.35,
        half_height_world=tire_r_r * 1.35,
        label="Rear tire detail (right)", big_xy=big,
    ))

    return panels


def _split_meta_columns(lines, max_lines_per_col=18):
    """Split ``lines`` into N columns of ~max_lines_per_col lines each.
    Split prefers blank-line boundaries to keep sections together."""
    if len(lines) <= max_lines_per_col:
        return [lines]
    cols = []
    cur = []
    for ln in lines:
        cur.append(ln)
        if len(cur) >= max_lines_per_col and ln.strip() == "":
            cols.append(cur)
            cur = []
    if cur:
        cols.append(cur)
    return cols


def render_composite(stl_path: str, meta_lines, output_dir: str,
                     preview_name: str, axles: dict):
    os.makedirs(output_dir, exist_ok=True)
    panels = _render_panels(stl_path, axles)

    # Figure: 2 rows of panels + a wide bottom strip for metadata.
    fig, axes = plt.subplots(2, 4, figsize=(22, 14))
    for i, img in enumerate(panels):
        row, col = divmod(i, 4)
        ax = axes[row][col]
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle("Parametric Underbody — Preview", fontsize=18, y=0.99)
    # Spread metadata into columns across the bottom strip.
    cols = _split_meta_columns(meta_lines, max_lines_per_col=18)
    n_cols = max(len(cols), 1)
    col_x_start = 0.01
    col_x_end = 0.99
    col_step = (col_x_end - col_x_start) / n_cols
    for i, col_lines in enumerate(cols):
        fig.text(
            col_x_start + i * col_step, 0.01,
            "\n".join(col_lines),
            fontsize=9, fontfamily="monospace",
            verticalalignment="bottom",
        )
    plt.tight_layout(rect=(0.0, 0.22, 1.0, 0.97))
    preview_path = os.path.join(output_dir, preview_name)
    plt.savefig(preview_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  composite saved: {os.path.basename(preview_path)}")


if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    render_composite(
        payload["stl_path"],
        payload["meta_lines"],
        payload["output_dir"],
        payload.get("preview_name", "preview.png"),
        payload["axles"],
    )
