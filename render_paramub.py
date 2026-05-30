"""Render a ParamUB car (clean body + 4 clean wheels) from 10 views.

Reuses the camera setup from ab-upt's render_surface_pressure_mesh.py
(the same VIEW_SPECS + parallel-projection auto-fit that produced the
surface_pressure_mesh_renders), but for plain geometry: the body is
shaded light grey, the wheels dark, with no scalar field.

Per car it writes:
    <out>/renders/<car>_10view.png            10-view montage
    <out>/views/<view_slug>/<car>.png          one PNG per view
                                               (used by the contact sheets)

The 4 wheel STLs are <car>_wheel_{front_left,front_right,rear_left,
rear_right}_clean.stl; the body is <car>_clean.stl. All live under
outputs/<car>/integrate/.

Usage:
    python render_paramub.py --car 2UKms
    python render_paramub.py --car 2UKms --out outputs/summary --max-faces 150000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

REPO_ROOT = Path(__file__).resolve().parent

# Materials. Upper shell stays grey (matches the original render); only
# the parametric underbody is blue so it reads as a distinct component
# in the manual visual check.
SHELL_COLOR = "#c4ccd4"   # light grey, as original
UNDERBODY_COLOR = "#3b6fa6"   # medium technical blue
WHEEL_COLOR = "#23262a"   # near-black
SECTION_SHELL_COLOR = "#52606d"
SECTION_UNDERBODY_COLOR = "#1e4e8c"
SECTION_WHEEL_COLOR = "#23262a"
Y_SECTIONS_MM = [0, 200, 400, 600, 800]

# (name, camera direction, view-up) -- identical to ab-upt VIEW_SPECS.
VIEW_SPECS = [
    ("Front", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("Rear", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("Top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("Bottom", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("Left", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ("Right", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("Top Front Iso", (-1.0, 0.8, 0.8), (0.0, 0.0, 1.0)),
    ("Top Rear Iso", (1.0, 0.8, 0.8), (0.0, 0.0, 1.0)),
    ("Bottom Front Iso", (-1.0, 0.8, -0.8), (0.0, 0.0, 1.0)),
    ("Bottom Rear Iso", (1.0, 0.8, -0.8), (0.0, 0.0, 1.0)),
]

# The three views the contact sheets are built from.
CONTACT_VIEWS = ["Bottom Front Iso", "Bottom", "Bottom Rear Iso"]


def view_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _normalize_vector(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"zero-length vector: {vec}")
    return arr / norm


def _camera_basis(direction, view_up):
    forward = _normalize_vector(direction)
    up_seed = _normalize_vector(view_up)
    up = up_seed - np.dot(up_seed, forward) * forward
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-8:
        raise ValueError(f"view_up {view_up} parallel to direction {direction}")
    up = up / up_norm
    right = np.cross(forward, up)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    return forward, up, right


def _compute_parallel_scale(points_xyz, center, direction, view_up,
                            image_aspect, fit_margin, zoom_factor):
    _, up, right = _camera_basis(direction, view_up)
    rel = points_xyz.astype(np.float32) - center.reshape(1, 3).astype(np.float32)
    horiz = rel @ right
    vert = rel @ up
    half_width = float(np.max(np.abs(horiz)))
    half_height = float(np.max(np.abs(vert)))
    parallel_scale = fit_margin * max(half_height, half_width / max(image_aspect, 1e-6))
    return parallel_scale / max(zoom_factor, 1e-6)


def _load_mesh(path: Path) -> "pv.PolyData":
    mesh = pv.read(path.as_posix())
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.triangulate()


def _decimate(mesh: "pv.PolyData", max_faces: int) -> "pv.PolyData":
    if max_faces <= 0 or mesh.n_faces <= max_faces:
        return mesh
    reduction = 1.0 - (float(max_faces) / float(mesh.n_faces))
    reduction = min(max(reduction, 0.0), 0.98)
    out = mesh.decimate(reduction, volume_preservation=True)
    return out.clean()


def load_car(car: str, max_faces: int):
    """Load the watertight clean body as a single mesh (rendered blue)
    plus the wheels. Returns (shell, underbody, wheels, all_points)
    where ``shell`` stays empty and ``underbody`` carries the full
    body -- keeps the existing render/section call signatures unchanged
    so only the colour mapping switches.
    """
    integ = REPO_ROOT / "outputs" / car / "integrate"
    clean_path = integ / f"{car}_clean.stl"
    if not clean_path.is_file():
        raise FileNotFoundError(f"no clean body: {clean_path}")
    underbody = _decimate(_load_mesh(clean_path), max_faces)
    shell = pv.PolyData()

    wheel_meshes = []
    for corner in ("front_left", "front_right", "rear_left", "rear_right"):
        wp = integ / f"{car}_wheel_{corner}_clean.stl"
        if wp.is_file():
            wheel_meshes.append(_decimate(_load_mesh(wp), max(max_faces // 2, 1)))
    if wheel_meshes:
        wheels = wheel_meshes[0]
        for w in wheel_meshes[1:]:
            wheels = wheels.merge(w)
    else:
        wheels = pv.PolyData()

    pts = []
    for m in (shell, underbody, wheels):
        if m.n_points:
            pts.append(np.asarray(m.points, dtype=np.float32))
    all_points = np.concatenate(pts, axis=0)
    return shell, underbody, wheels, all_points


def render_view(shell, underbody, wheels, all_points, direction, view_up,
                window_size=(640, 460), fit_margin=1.08, zoom_factor=1.0):
    # "three lights" + eye-dome lighting (screen-space depth-edge enhancement)
    # makes underbody contours read clearly: the floor/splitter/diffuser
    # creases show up as crisp edges instead of washing out under the flat
    # light-kit ambient. Higher specular + lower ambient also helps form
    # definition without going metallic.
    plotter = pv.Plotter(off_screen=True, window_size=window_size,
                         lighting="three lights")
    plotter.set_background("#f3f4f6")
    plotter.enable_eye_dome_lighting()
    if shell.n_points:
        plotter.add_mesh(shell, color=SHELL_COLOR, smooth_shading=True,
                         specular=0.18, specular_power=14, ambient=0.22,
                         diffuse=0.76, show_scalar_bar=False)
    if underbody.n_points:
        plotter.add_mesh(underbody, color=UNDERBODY_COLOR, smooth_shading=True,
                         specular=0.25, specular_power=20, ambient=0.18,
                         diffuse=0.78, show_scalar_bar=False)
    if wheels.n_points:
        plotter.add_mesh(wheels, color=WHEEL_COLOR, smooth_shading=True,
                         specular=0.25, specular_power=22, ambient=0.15,
                         diffuse=0.72, show_scalar_bar=False)

    # Frame on the union of body+wheel bounds so every view is centered.
    bmin = all_points.min(axis=0)
    bmax = all_points.max(axis=0)
    center = 0.5 * (bmin + bmax)
    radius = 0.5 * float(np.max(bmax - bmin))
    cam_dir = _normalize_vector(direction)
    up = _normalize_vector(view_up)
    distance = max(radius * 3.2 / max(zoom_factor, 1e-3), 1e-3)
    position = center + cam_dir * distance
    plotter.camera_position = [tuple(position.tolist()),
                               tuple(center.tolist()),
                               tuple(up.tolist())]
    plotter.camera.SetViewAngle(24.0)
    plotter.camera.SetParallelProjection(True)
    image_aspect = float(window_size[0]) / float(window_size[1])
    pscale = _compute_parallel_scale(all_points, center, direction, view_up,
                                     image_aspect, fit_margin, zoom_factor)
    plotter.camera.SetParallelScale(pscale)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def _slice_to_segments_xz(mesh, y_mm: float) -> np.ndarray | None:
    """Slice ``mesh`` at the plane Y=y_mm and return its intersection as an
    (M, 2, 2) array of XZ line segments suitable for matplotlib's
    LineCollection. Returns None when the slice is empty (mesh doesn't
    cross the plane, or it does but with no line cells)."""
    if mesh.n_points == 0:
        return None
    try:
        slc = mesh.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, y_mm, 0.0))
    except Exception:
        return None
    if slc is None or slc.n_points == 0:
        return None
    lines = np.asarray(slc.lines)
    if lines.size == 0:
        return None
    # PyVista slice cells are all 2-pt segments: flattened [2, i0, i1, ...]
    lines = lines.reshape(-1, 3)
    seg_idx = lines[:, 1:]
    seg_xyz = np.asarray(slc.points)[seg_idx]   # (M, 2, 3)
    return seg_xyz[..., [0, 2]]                  # XZ only -> (M, 2, 2)


def render_y_section(shell, underbody, wheels, y_mm: float,
                      all_points, ax) -> None:
    """Draw the cross-section at Y=y_mm on a matplotlib axes in the XZ
    plane (X length, Z height). Upper shell in grey, parametric
    underbody in blue, wheels in dark."""
    shell_segs = (_slice_to_segments_xz(shell, y_mm)
                  if shell.n_points else None)
    ub_segs = (_slice_to_segments_xz(underbody, y_mm)
               if underbody.n_points else None)
    wheel_segs = (_slice_to_segments_xz(wheels, y_mm)
                  if wheels.n_points else None)

    have_any = any(s is not None and len(s)
                   for s in (shell_segs, ub_segs, wheel_segs))
    if not have_any:
        ax.text(0.5, 0.5, f"no intersection at Y={y_mm:.0f} mm",
                ha="center", va="center", fontsize=9, color="#52606d",
                transform=ax.transAxes)
        ax.set_facecolor("#f3f4f6")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_facecolor("#f3f4f6")
        if shell_segs is not None and len(shell_segs):
            ax.add_collection(LineCollection(
                shell_segs, colors=SECTION_SHELL_COLOR, linewidths=0.9))
        if ub_segs is not None and len(ub_segs):
            ax.add_collection(LineCollection(
                ub_segs, colors=SECTION_UNDERBODY_COLOR, linewidths=0.9))
        if wheel_segs is not None and len(wheel_segs):
            ax.add_collection(LineCollection(
                wheel_segs, colors=SECTION_WHEEL_COLOR, linewidths=0.9))
        # Lock XZ extent + aspect across all 5 sections so the body shape
        # is comparable panel-to-panel.
        xmin, _, zmin = all_points.min(axis=0)
        xmax, _, zmax = all_points.max(axis=0)
        pad_x = 0.04 * (xmax - xmin)
        pad_z = 0.10 * (zmax - zmin)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(zmin - pad_z, zmax + pad_z)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#d0d6dc", linewidth=0.4, alpha=0.7)
        ax.tick_params(labelsize=6, length=2, width=0.4)
        for spine in ax.spines.values():
            spine.set_color("#b9c0c8")
            spine.set_linewidth(0.5)
    ax.set_title(f"Section Y = {y_mm:.0f} mm", fontsize=10, pad=6)


def render_car(car: str, out_dir: Path, max_faces: int) -> None:
    shell, underbody, wheels, all_points = load_car(car, max_faces)
    print(f"[{car}] shell f={shell.n_faces:,}  "
          f"underbody f={underbody.n_faces:,}  "
          f"wheels f={wheels.n_faces:,}", flush=True)

    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    for name, direction, view_up in VIEW_SPECS:
        images[name] = render_view(shell, underbody, wheels, all_points,
                                    direction, view_up)

    # 3-row montage: rows 0-1 = 10 PyVista views, row 2 = Y-sections.
    fig = plt.figure(figsize=(18, 11.5), dpi=150)
    gs = fig.add_gridspec(3, 5, wspace=0.04, hspace=0.14,
                          height_ratios=[1.0, 1.0, 0.85])
    for idx, (name, _, _) in enumerate(VIEW_SPECS):
        ax = fig.add_subplot(gs[idx // 5, idx % 5])
        ax.imshow(images[name])
        ax.set_axis_off()
        ax.set_title(name, fontsize=10, pad=6)
    for i, y in enumerate(Y_SECTIONS_MM):
        ax = fig.add_subplot(gs[2, i])
        render_y_section(shell, underbody, wheels, float(y), all_points, ax)
    fig.suptitle(f"{car} -- clean body (blue) + wheels",
                 fontsize=14, y=0.995)
    montage = renders_dir / f"{car}_10view.png"
    fig.savefig(montage, bbox_inches="tight")
    plt.close(fig)
    print(f"[{car}] wrote {montage}", flush=True)

    # Per-view PNGs for the contact sheets (unchanged set of views).
    for name in CONTACT_VIEWS:
        vdir = out_dir / "views" / view_slug(name)
        vdir.mkdir(parents=True, exist_ok=True)
        plt.imsave(vdir / f"{car}.png", images[name])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--car", required=True, help="car base name, e.g. 2UKms")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "summary")
    ap.add_argument("--max-faces", type=int, default=150000,
                     help="decimation target for the body (render only).")
    args = ap.parse_args()
    render_car(args.car, args.out, args.max_faces)


if __name__ == "__main__":
    main()
