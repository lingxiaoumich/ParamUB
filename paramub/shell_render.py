"""Render one step's 10-panel composite via matplotlib.

The layout (10 views) mirrors ``ab-upt`` / ``Image2CFDv3``'s canonical
multi-view layout. The renderer itself is matplotlib's
``Poly3DCollection`` with hand-rolled Lambertian shading. matplotlib is
used (rather than pyvista) because on the OSMesa stack in this conda
env, the combination of an explicit ``camera_position`` + parallel
projection produces black framebuffers and intermittent ``std::bad_alloc``;
matplotlib bypasses VTK entirely and is environment-independent.

Call as::

    python -m paramub.shell_render <payload.json>

Payload schema is documented in :func:`main`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# 10 canonical views, matching the ab-upt convention.
VIEW_SPECS = [
    ("Front",            (-1.0,  0.0,  0.0), (0.0, 0.0, 1.0)),
    ("Rear",             ( 1.0,  0.0,  0.0), (0.0, 0.0, 1.0)),
    ("Top",              ( 0.0,  0.0,  1.0), (0.0, 1.0, 0.0)),
    ("Bottom",           ( 0.0,  0.0, -1.0), (0.0, 1.0, 0.0)),
    ("Left",             ( 0.0, -1.0,  0.0), (0.0, 0.0, 1.0)),
    ("Right",            ( 0.0,  1.0,  0.0), (0.0, 0.0, 1.0)),
    ("Top Front Iso",    (-1.0,  0.8,  0.8), (0.0, 0.0, 1.0)),
    ("Top Rear Iso",     ( 1.0,  0.8,  0.8), (0.0, 0.0, 1.0)),
    ("Bottom Front Iso", (-1.0,  0.8, -0.8), (0.0, 0.0, 1.0)),
    ("Bottom Rear Iso",  ( 1.0,  0.8, -0.8), (0.0, 0.0, 1.0)),
]
# Lighting (Lambertian only). Sun slightly above the +x −y +z corner
# so the iso views pick up nice highlights along the roof.
LIGHT_DIR = np.asarray([-0.4, -0.6, 0.7])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.35
DIFFUSE = 0.65
# Camera framing. Smaller numbers = tighter zoom.
ZOOM_MARGIN = 1.05      # padding around the body's projected extent


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _camera_basis(direction, view_up):
    fwd = _normalize(direction)
    up_seed = _normalize(view_up)
    up = up_seed - np.dot(up_seed, fwd) * fwd
    n = float(np.linalg.norm(up))
    if n < 1e-8:
        raise ValueError(f"view_up {view_up} parallel to direction {direction}")
    up = up / n
    right = np.cross(fwd, up)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    return fwd, up, right


def _view_init_from_dir(direction, view_up):
    d = _normalize(direction)
    elev = float(np.degrees(np.arcsin(d[2])))
    azim = float(np.degrees(np.arctan2(d[1], d[0])))
    return elev, azim


def _setup_view(ax, bounds, direction, view_up, label):
    """Frame the body inside ``ax`` for one camera direction.

    matplotlib's 3D axes uses a cube bounding box. We project the
    body's 8 corner-points onto the camera's right and up axes,
    then size the cube to ``ZOOM_MARGIN × max(half_w, half_h)`` —
    using max(w, h) instead of max(w, h, depth) so the camera
    actually fills the panel rather than the body shrinking to fit
    its longest 3D axis.
    """
    elev, azim = _view_init_from_dir(direction, view_up)
    ax.view_init(elev=elev, azim=azim)
    cx, cy, cz = 0.5 * (bounds[0] + bounds[1])
    _, up, right = _camera_basis(direction, view_up)
    corners = np.array([[bounds[a, 0], bounds[b, 1], bounds[c, 2]]
                        for a in (0, 1) for b in (0, 1) for c in (0, 1)],
                       dtype=np.float64)
    rel = corners - np.array([cx, cy, cz])
    half_w = float(np.max(np.abs(rel @ right)))
    half_h = float(np.max(np.abs(rel @ up)))
    half = ZOOM_MARGIN * max(half_w, half_h)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title(label, fontsize=10, pad=2)


# --- Lambertian-shaded mesh -----------------------------------------------

def _face_normals(verts, faces):
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norms, 1e-12)


def _shaded_colors(normals, base_color):
    rgb = np.asarray(to_rgb(base_color), dtype=np.float64)
    intensity = AMBIENT + DIFFUSE * np.clip(normals @ LIGHT_DIR, 0.0, 1.0)
    out = np.clip(intensity[:, None] * rgb[None, :], 0.0, 1.0)
    rgba = np.empty((len(out), 4), dtype=np.float64)
    rgba[:, :3] = out
    rgba[:, 3] = 1.0
    return rgba


def _add_shaded_mesh(ax, verts, faces, base_color, *, alpha: float = 1.0):
    if len(faces) == 0:
        return
    tris = verts[faces]
    normals = _face_normals(verts, faces)
    facecolors = _shaded_colors(normals, base_color)
    facecolors[:, 3] = alpha
    pc = Poly3DCollection(tris, facecolors=facecolors,
                          edgecolor="none", linewidth=0.0)
    pc.set_zsort("average")
    ax.add_collection3d(pc)


# --- Per-step overlays ----------------------------------------------------

def _add_origin_axes(ax, length: float):
    for vec, col, lab in (
        (np.array([1.0, 0.0, 0.0]), "#dc2626", "+x (rear)"),
        (np.array([0.0, 1.0, 0.0]), "#16a34a", "+y"),
        (np.array([0.0, 0.0, 1.0]), "#2563eb", "+z (up)"),
    ):
        end = vec * length
        ax.plot([0, end[0]], [0, end[1]], [0, end[2]],
                color=col, linewidth=2.5, zorder=5)
        ax.text(*end * 1.10, lab, color=col, fontsize=8, zorder=6)


def _add_y_plane(ax, bounds, color="#22c55e", alpha=0.18):
    x_lo, x_hi = bounds[0, 0], bounds[1, 0]
    z_lo, z_hi = bounds[0, 2], bounds[1, 2]
    quad = np.array([
        [x_lo, 0.0, z_lo], [x_hi, 0.0, z_lo],
        [x_hi, 0.0, z_hi], [x_lo, 0.0, z_hi],
    ])
    poly = Poly3DCollection([quad], facecolors=color, alpha=alpha,
                            edgecolor="#16a34a", linewidth=1.0)
    ax.add_collection3d(poly)


def _add_cylinder(ax, w, *, radius_factor: float = 1.00,
                  axial_factor: float = 1.00,
                  color: str = "#fb923c", alpha: float = 0.40,
                  edges: int = 32):
    """Translucent cylinder with axis = +y."""
    r = w["radius"] * radius_factor
    ax_half = w["axial_half"] * axial_factor
    theta = np.linspace(0.0, 2.0 * np.pi, edges, endpoint=False)
    cx_ring = r * np.cos(theta) + w["x"]
    cz_ring = r * np.sin(theta) + w["z"]
    y_lo = w["y"] - ax_half
    y_hi = w["y"] + ax_half
    sides = []
    for k in range(edges):
        k1 = (k + 1) % edges
        sides.append([
            [cx_ring[k],  y_lo, cz_ring[k]],
            [cx_ring[k1], y_lo, cz_ring[k1]],
            [cx_ring[k1], y_hi, cz_ring[k1]],
            [cx_ring[k],  y_hi, cz_ring[k]],
        ])
    sides.append([[cx_ring[k], y_lo, cz_ring[k]] for k in range(edges)])
    sides.append([[cx_ring[k], y_hi, cz_ring[k]] for k in range(edges)])
    pc = Poly3DCollection(sides, facecolors=color, alpha=alpha,
                          edgecolor="#9a3412", linewidth=1.2)
    ax.add_collection3d(pc)
    for cy in (y_lo, y_hi):
        ax.plot(np.r_[cx_ring, cx_ring[0]],
                np.full(edges + 1, cy),
                np.r_[cz_ring, cz_ring[0]],
                color="#9a3412", linewidth=1.4, zorder=7)


def _add_marker(ax, centre, color, size=40.0, edge="black"):
    ax.scatter([centre[0]], [centre[1]], [centre[2]],
               c=color, s=size, marker="o",
               edgecolors=edge, linewidths=0.7,
               depthshade=False, zorder=10)


def _build_legend(keep_color, remove_color, *, remove_count,
                  show_axes, show_y_plane, has_wheels_xy, has_wheels_3d,
                  wheels_xy_color, wheels_3d_color, show_cylinders,
                  has_extra_origin):
    handles = [Patch(facecolor=keep_color, edgecolor="black", label="KEEP")]
    if remove_count:
        handles.append(Patch(facecolor=remove_color, edgecolor="black",
                             label=f"REMOVE ({remove_count:,})"))
    if show_y_plane:
        handles.append(Patch(facecolor="#22c55e", alpha=0.4,
                             edgecolor="#16a34a", label="y = 0 plane"))
    if show_axes:
        for col, lab in (("#dc2626", "+x (rear)"),
                         ("#16a34a", "+y"),
                         ("#2563eb", "+z (up)")):
            handles.append(plt.Line2D([0], [0], color=col, linewidth=2.5,
                                       label=lab))
    if has_wheels_xy:
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=wheels_xy_color,
                                  markeredgecolor="black", markersize=8,
                                  label="wheel (x,y)"))
    if has_wheels_3d:
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=wheels_3d_color,
                                  markeredgecolor="black", markersize=8,
                                  label="wheel hub"))
    if show_cylinders:
        handles.append(Patch(facecolor="#fb923c", alpha=0.4,
                             edgecolor="#c2410c", label="wheel cylinder"))
    if has_extra_origin:
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor="#2563eb",
                                  markeredgecolor="black", markersize=9,
                                  label="ray origin"))
    return handles


def main(argv):
    """Render a 10-panel composite for one step.

    Payload schema::

        {
          "stl_path": "...",          # body STL to render
          "remove_indices": [int],    # face indices marked REMOVE
          "title": "Step N — ...",
          "out_path": "...png",
          "view_bounds": [[xmin,ymin,zmin],[xmax,ymax,zmax]],
          "keep_color":   "#b8c0c8",
          "remove_color": "#dc2626",
          "show_axes":     bool,
          "show_y_plane":  bool,
          "wheels_xy":     [{"x":.., "y":..}, ...],
          "wheels_xy_color": "#dc2626",
          "wheels_3d":     [{"x":.., "y":.., "z":.., "radius":.., "axial_half":..}, ...],
          "wheels_3d_color": "#dc2626",
          "show_cylinders": bool,
          "cylinder_radius_factor": 1.00,
          "cylinder_axial_factor": 1.00,
          "extra_origin": [x,y,z] | null,
          "body_alpha":   1.0
        }
    """
    import trimesh

    payload = json.loads(Path(argv[1]).read_text())

    body = trimesh.load(payload["stl_path"], file_type="stl",
                        force="mesh", process=False)
    verts = np.asarray(body.vertices, dtype=np.float64)
    faces = np.asarray(body.faces, dtype=np.int64)
    n = len(faces)
    remove_idx = np.asarray(payload["remove_indices"], dtype=np.int64)
    remove_mask = np.zeros(n, dtype=bool)
    if len(remove_idx):
        remove_mask[remove_idx] = True

    keep_faces = faces[~remove_mask]
    remove_faces = faces[remove_mask]

    view_bounds = np.asarray(payload["view_bounds"], dtype=np.float64)
    keep_color = payload.get("keep_color", "#b8c0c8")
    remove_color = payload.get("remove_color", "#dc2626")
    show_axes = bool(payload.get("show_axes", False))
    show_y_plane = bool(payload.get("show_y_plane", False))
    wheels_xy = payload.get("wheels_xy") or []
    wheels_3d = payload.get("wheels_3d") or []
    wheels_xy_color = payload.get("wheels_xy_color", "#dc2626")
    wheels_3d_color = payload.get("wheels_3d_color", "#dc2626")
    show_cylinders = bool(payload.get("show_cylinders", False))
    extra_origin = payload.get("extra_origin")
    diag = float(np.linalg.norm(view_bounds[1] - view_bounds[0]))

    body_alpha = float(payload.get("body_alpha", 1.0))
    if body_alpha >= 1.0 and (
        (wheels_3d and len(remove_faces) == 0 and not show_cylinders
         and extra_origin is None)
        or show_cylinders
    ):
        body_alpha = 0.35

    fig, axes = plt.subplots(2, 5, figsize=(24, 9.5),
                             subplot_kw={"projection": "3d"})
    for ax, (label, direction, view_up) in zip(axes.flat, VIEW_SPECS):
        _setup_view(ax, view_bounds, direction, view_up, label)
        _add_shaded_mesh(ax, verts, keep_faces, base_color=keep_color,
                         alpha=body_alpha)
        _add_shaded_mesh(ax, verts, remove_faces, base_color=remove_color,
                         alpha=0.97)
        if show_axes:
            _add_origin_axes(ax, length=0.18 * diag)
        if show_y_plane:
            _add_y_plane(ax, view_bounds)
        cyl_rfac = float(payload.get("cylinder_radius_factor", 1.00))
        cyl_afac = float(payload.get("cylinder_axial_factor", 1.00))
        for w in wheels_3d:
            if show_cylinders:
                _add_cylinder(ax, w, radius_factor=cyl_rfac,
                              axial_factor=cyl_afac)
            _add_marker(ax, (w["x"], w["y"], w["z"]), wheels_3d_color,
                        size=45)
        for w in wheels_xy:
            _add_marker(ax, (w["x"], w["y"], 0.0), wheels_xy_color, size=45)
        if extra_origin is not None:
            _add_marker(ax, tuple(extra_origin), "#2563eb", size=60)

    handles = _build_legend(
        keep_color, remove_color,
        remove_count=int(remove_mask.sum()),
        show_axes=show_axes, show_y_plane=show_y_plane,
        has_wheels_xy=bool(wheels_xy), has_wheels_3d=bool(wheels_3d),
        wheels_xy_color=wheels_xy_color,
        wheels_3d_color=wheels_3d_color,
        show_cylinders=show_cylinders,
        has_extra_origin=extra_origin is not None,
    )
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(payload["title"], fontsize=14, y=0.995)
    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    out_path = Path(payload["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv)
