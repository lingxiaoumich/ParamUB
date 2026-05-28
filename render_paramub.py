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
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

REPO_ROOT = Path(__file__).resolve().parent

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
    """Return (body_mesh, wheels_mesh, all_points) decimated for render."""
    integ = REPO_ROOT / "outputs" / car / "integrate"
    body_path = integ / f"{car}_clean.stl"
    if not body_path.is_file():
        raise FileNotFoundError(f"no clean body: {body_path}")
    body = _decimate(_load_mesh(body_path), max_faces)

    wheel_meshes = []
    for corner in ("front_left", "front_right", "rear_left", "rear_right"):
        wp = integ / f"{car}_wheel_{corner}_clean.stl"
        if wp.is_file():
            # Wheels are already light (~200k faces); cap each at max_faces/2.
            wheel_meshes.append(_decimate(_load_mesh(wp), max(max_faces // 2, 1)))
    # Some cars finish the body but not all 4 wheel cleans; render the
    # body alone rather than crashing.
    if wheel_meshes:
        wheels = wheel_meshes[0]
        for w in wheel_meshes[1:]:
            wheels = wheels.merge(w)
    else:
        wheels = pv.PolyData()

    pts = [np.asarray(body.points, dtype=np.float32)]
    if wheels.n_points:
        pts.append(np.asarray(wheels.points, dtype=np.float32))
    all_points = np.concatenate(pts, axis=0)
    return body, wheels, all_points


def render_view(body, wheels, all_points, direction, view_up,
                window_size=(640, 460), fit_margin=1.08, zoom_factor=1.0):
    plotter = pv.Plotter(off_screen=True, window_size=window_size,
                         lighting="light kit")
    plotter.set_background("#f3f4f6")
    plotter.add_mesh(body, color="#c4ccd4", smooth_shading=True,
                     specular=0.12, specular_power=12, ambient=0.25,
                     diffuse=0.75, show_scalar_bar=False)
    if wheels.n_points:
        plotter.add_mesh(wheels, color="#33373b", smooth_shading=True,
                         specular=0.2, specular_power=18, ambient=0.2,
                         diffuse=0.7, show_scalar_bar=False)

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


def render_car(car: str, out_dir: Path, max_faces: int) -> None:
    body, wheels, all_points = load_car(car, max_faces)
    print(f"[{car}] body f={body.n_faces:,}  "
          f"wheels f={wheels.n_faces:,}", flush=True)

    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    for name, direction, view_up in VIEW_SPECS:
        images[name] = render_view(body, wheels, all_points, direction, view_up)

    # 10-view montage (2 rows x 5).
    fig = plt.figure(figsize=(18, 8), dpi=150)
    gs = fig.add_gridspec(2, 5, wspace=0.02, hspace=0.08)
    for idx, (name, _, _) in enumerate(VIEW_SPECS):
        ax = fig.add_subplot(gs[idx // 5, idx % 5])
        ax.imshow(images[name])
        ax.set_axis_off()
        ax.set_title(name, fontsize=10, pad=6)
    fig.suptitle(f"{car} -- clean body + wheels", fontsize=14, y=0.98)
    montage = renders_dir / f"{car}_10view.png"
    fig.savefig(montage, bbox_inches="tight")
    plt.close(fig)
    print(f"[{car}] wrote {montage}", flush=True)

    # Per-view PNGs for the contact sheets.
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
