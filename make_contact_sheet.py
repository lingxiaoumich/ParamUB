"""Render a 6x5 contact sheet for the 30 catalog cases.

Per case: one isometric PyVista render + one *filled* 2D top-half section.
The section is built from a trimesh cut through Y=0; only the +X half of
the cross-section (the upper half of the wheel) is drawn, as solid
polygons (so the contact sheet shows just the cut face, not the surface
of the kept half).

Run standalone (it must NOT import CadQuery — OCP leaves OpenGL state
that turns subsequent PyVista off-screen renders into black frames on
llvmpipe).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import MultiPolygon, Polygon, box


ROOT = Path(__file__).resolve().parent
CATALOG_DIR = ROOT / "outputs" / "catalog"
SUMMARY_JSON = CATALOG_DIR / "summary.json"
CONTACT_PATH = CATALOG_DIR / "contact_sheet.png"


def _add_mesh(pl, mesh):
    pl.add_mesh(
        mesh,
        color="#b0b8c1",
        smooth_shading=True,
        specular=0.4,
        specular_power=15,
        ambient=0.25,
        diffuse=0.85,
    )


def render_iso(mesh: pv.DataSet, size=(800, 700)) -> np.ndarray:
    cx, cy, cz = mesh.center
    r = mesh.length * 1.5
    pl = pv.Plotter(off_screen=True, window_size=size)
    _add_mesh(pl, mesh)
    pl.camera.position = (cx + r * 0.7, cy + r * 0.5, cz + r * 0.5)
    pl.camera.focal_point = (cx, cy, cz)
    pl.camera.up = (0, 0, 1)
    pl.camera.reset_clipping_range()
    pl.camera.zoom(0.85)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _top_half_polygons(stl_path: str):
    """Compute the filled top-half cross-section through Y=0 as shapely polygons.

    Returns (polygons, (umin, umax, vmin, vmax)) where the polygons are in
    a 2D coordinate frame with u = 3D Z (axial, horizontal in the plot)
    and v = 3D X (radial, vertical in the plot). Only the v >= 0 half is
    kept (the "top half" of the wheel).
    """
    mesh = trimesh.load(stl_path)
    cx, cy, cz = mesh.centroid

    section_3d = mesh.section(plane_origin=[cx, cy, cz], plane_normal=[0, 1, 0])
    if section_3d is None:
        return [], (0.0, 1.0, 0.0, 1.0)

    # Custom 2D projection that maps 3D (X, Y, Z) -> 2D (Z - cz, X - cx).
    # Putting X on the v axis means "top of the plot" = "top of the wheel".
    to_2D = np.array(
        [
            [0.0, 0.0, 1.0, -cz],
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    section_2d, _ = section_3d.to_planar(to_2D=to_2D)
    polys = list(section_2d.polygons_full)

    if not polys:
        return [], (0.0, 1.0, 0.0, 1.0)

    all_coords = np.vstack([np.asarray(p.exterior.coords) for p in polys])
    umin, vmin = all_coords.min(axis=0)
    umax, vmax = all_coords.max(axis=0)

    # Clip to v >= 0 (top half — points above the spin axis).
    pad = max(umax - umin, vmax - vmin) * 0.1
    clip = box(umin - pad, 0.0, umax + pad, vmax + pad)

    top_polys = []
    for p in polys:
        clipped = p.intersection(clip)
        if clipped.is_empty:
            continue
        if isinstance(clipped, Polygon):
            top_polys.append(clipped)
        elif isinstance(clipped, MultiPolygon):
            top_polys.extend(clipped.geoms)

    if not top_polys:
        return [], (umin, umax, 0.0, vmax)

    return top_polys, (umin, umax, 0.0, vmax)


def render_top_half_section(stl_path: str, size=(800, 700)) -> np.ndarray:
    """Draw the filled top-half cross-section through Y=0 as a 2D image."""
    polys, (umin, umax, vmin, vmax) = _top_half_polygons(stl_path)

    fig = plt.figure(
        figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100, facecolor="white"
    )
    ax = fig.add_subplot(111)

    for poly in polys:
        verts, codes = [], []

        ex = list(poly.exterior.coords)
        verts.append(ex[0])
        codes.append(MplPath.MOVETO)
        for pt in ex[1:]:
            verts.append(pt)
            codes.append(MplPath.LINETO)
        verts.append(ex[0])
        codes.append(MplPath.CLOSEPOLY)

        for interior in poly.interiors:
            inter = list(interior.coords)
            verts.append(inter[0])
            codes.append(MplPath.MOVETO)
            for pt in inter[1:]:
                verts.append(pt)
                codes.append(MplPath.LINETO)
            verts.append(inter[0])
            codes.append(MplPath.CLOSEPOLY)

        path = MplPath(verts, codes)
        ax.add_patch(
            PathPatch(
                path,
                facecolor="#b0b8c1",
                edgecolor="#111827",
                linewidth=1.2,
            )
        )

    pad_u = max((umax - umin) * 0.05, 5.0)
    pad_v = max((vmax - vmin) * 0.05, 5.0)
    ax.set_xlim(umin - pad_u, umax + pad_u)
    ax.set_ylim(vmin - pad_v, vmax + pad_v)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[..., :3].copy()
    plt.close(fig)
    return img


def build_contact_sheet(rows, cols=5):
    pv.global_theme.background = "white"

    cases = sorted(rows, key=lambda r: r["case_name"])
    n = len(cases)
    rows_ct = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows_ct, cols, figsize=(cols * 4.4, rows_ct * 3.0), squeeze=False
    )

    for idx in range(rows_ct * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if idx >= n:
            ax.axis("off")
            continue
        case = cases[idx]
        mesh = pv.read(case["stl_path"])
        iso = render_iso(mesh)
        sec = render_top_half_section(case["stl_path"])
        combined = np.hstack([iso, sec])
        ax.imshow(combined)
        title = f"{case['case_name']}\n{case.get('size', '')}  {case.get('category', '')}".strip()
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        print(f"  rendered {case['case_name']}")

    plt.suptitle(
        "ParamUB Tire Catalog — Isometric and Top-Half Section",
        fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(CONTACT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = json.loads(SUMMARY_JSON.read_text())
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        raise SystemExit("no successful catalog cases found")
    build_contact_sheet(ok_rows)
    print(f"saved {CONTACT_PATH}")


if __name__ == "__main__":
    main()
