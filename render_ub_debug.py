"""Debug-render the underbody integration chain to localise geometry
artifacts (e.g. a stray horizontal spike on a wheelhouse).

For each ARTIFACT/step of the per-car pipeline it writes one composite in
the SAME layout as ``render_paramub.py`` (10 PyVista views + a row of
Y = 0/200/400/600/800 mm cross-sections), overlaying the relevant meshes
with distinct colours and a semi-transparent shell for context:

  [1] detected   shell + placed wheels                  (detected geometry)
  [2] underbody  shell(ctx) + raw parametric underbody  (generated geometry)
  [3] trimmed    shell(ctx) + trimmed UB + cut polygon  (generated + extraction box)
  [4] combined   shell+UB merged (pre-watertight)
  [5] clean      final watertight body + wheels         (what you see)

All meshes are read straight from an existing per-car bundle, so this runs
on a COMPLETED pipeline with no re-run. The Y-section row is the key view
for a "horizontal spike": a section through the front-wheel Y shows the
offending profile, and comparing the same section across steps [2]->[5]
tells you which operation introduced it.

It reuses the camera + section helpers from ``render_paramub`` so the
template matches the production renders exactly.

Usage:
  python render_ub_debug.py --car 0mjLG --bundle outputs/0mjLG_v2
  python render_ub_debug.py --car 0mjLG --bundle outputs/0mjLG_v2 \
      --extra outputs/0mjLG_v2/shell_debug/0mjLG_step6b_kept.stl#step6b#"#16a34a"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

from render_paramub import (
    VIEW_SPECS,
    Y_SECTIONS_MM,
    _compute_parallel_scale,
    _decimate,
    _load_mesh,
    _normalize_vector,
    _slice_to_segments_xz,
)

REPO_ROOT = Path(__file__).resolve().parent

# Component colours (consistent across every composite).
COL_SHELL = "#c4ccd4"      # light grey – shell as the main subject
COL_SHELL_CTX = "#9aa6b2"  # muted grey – shell shown only as context
COL_UB = "#3b6fa6"         # technical blue – generated underbody
COL_WHEEL = "#23262a"      # near-black – wheels
COL_CUT = "#d1495b"        # red – extraction / cut polygon
COL_COMBINED = "#3b6fa6"


def render_view_multi(components, all_points, direction, view_up,
                      window_size=(640, 460), fit_margin=1.08,
                      zoom_factor=1.0, focus=None):
    """One PyVista screenshot of a set of (mesh, color, opacity) layers,
    framed like ``render_paramub.render_view``. If ``focus`` = (center3,
    half) is given, the camera is centred there and zoomed to ``half`` mm
    instead of fitting the whole body — used to inspect one wheelhouse."""
    plotter = pv.Plotter(off_screen=True, window_size=window_size,
                         lighting="three lights")
    plotter.set_background("#f3f4f6")
    plotter.enable_eye_dome_lighting()
    for mesh, color, opacity in components:
        if mesh.n_points:
            plotter.add_mesh(mesh, color=color, opacity=opacity,
                             smooth_shading=True, specular=0.2,
                             specular_power=18, ambient=0.2, diffuse=0.77,
                             show_scalar_bar=False)
    if focus is not None:
        center = np.asarray(focus[0], dtype=np.float64)
        half = float(focus[1])
        radius = half
        pscale = fit_margin * half
    else:
        bmin = all_points.min(axis=0)
        bmax = all_points.max(axis=0)
        center = 0.5 * (bmin + bmax)
        radius = 0.5 * float(np.max(bmax - bmin))
        aspect = float(window_size[0]) / float(window_size[1])
        pscale = _compute_parallel_scale(all_points, center, direction,
                                         view_up, aspect, fit_margin,
                                         zoom_factor)
    cam_dir = _normalize_vector(direction)
    up = _normalize_vector(view_up)
    distance = max(radius * 3.2 / max(zoom_factor, 1e-3), 1e-3)
    position = center + cam_dir * distance
    plotter.camera_position = [tuple(position.tolist()),
                               tuple(center.tolist()),
                               tuple(up.tolist())]
    plotter.camera.SetViewAngle(24.0)
    plotter.camera.SetParallelProjection(True)
    plotter.camera.SetParallelScale(pscale)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def section_multi(ax, components, y_mm, all_points, focus=None):
    """Draw the Y=y_mm cross-section of every component (its colour) in XZ.
    With ``focus``=(center3, half) the XZ window is zoomed to that box."""
    drew = False
    for mesh, color, _opacity in components:
        segs = _slice_to_segments_xz(mesh, y_mm) if mesh.n_points else None
        if segs is not None and len(segs):
            ax.add_collection(LineCollection(segs, colors=color,
                                             linewidths=1.0))
            drew = True
    ax.set_facecolor("#f3f4f6")
    if not drew:
        ax.text(0.5, 0.5, f"no intersection\nY={y_mm:.0f} mm", ha="center",
                va="center", fontsize=8, color="#52606d",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if focus is not None:
            cx, _cy, cz = np.asarray(focus[0], dtype=np.float64)
            half = float(focus[1])
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cz - half, cz + half)
        else:
            xmin, _, zmin = all_points.min(axis=0)
            xmax, _, zmax = all_points.max(axis=0)
            ax.set_xlim(xmin - 0.04 * (xmax - xmin), xmax + 0.04 * (xmax - xmin))
            ax.set_ylim(zmin - 0.10 * (zmax - zmin), zmax + 0.10 * (zmax - zmin))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#d0d6dc", linewidth=0.4, alpha=0.7)
        ax.tick_params(labelsize=6, length=2, width=0.4)
        for spine in ax.spines.values():
            spine.set_color("#b9c0c8")
            spine.set_linewidth(0.5)
    ax.set_title(f"Section Y = {y_mm:.0f} mm", fontsize=10, pad=6)


def render_step(title, components, out_path, focus=None, ys=None):
    """``components``: list of (pv_mesh, color, opacity, label). Writes a
    3-row montage (2 rows of 10 views + 1 row of Y sections). ``focus`` =
    (center3, half) zooms every panel onto that box; ``ys`` overrides the
    section planes (defaults to Y_SECTIONS_MM)."""
    ys = list(ys) if ys is not None else list(Y_SECTIONS_MM)
    comps = [(m, c, o) for (m, c, o, _l) in components]
    pts = [np.asarray(m.points, dtype=np.float32)
           for (m, _c, _o, _l) in components if m.n_points]
    if not pts:
        print(f"[debug-render] SKIP {out_path.name}: no geometry")
        return
    all_points = np.concatenate(pts, axis=0)

    images = {name: render_view_multi(comps, all_points, d, vu, focus=focus)
              for name, d, vu in VIEW_SPECS}

    fig = plt.figure(figsize=(18, 11.5), dpi=150)
    gs = fig.add_gridspec(3, 5, wspace=0.04, hspace=0.16,
                          height_ratios=[1.0, 1.0, 0.85])
    for i, (name, _d, _vu) in enumerate(VIEW_SPECS):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        ax.imshow(images[name])
        ax.set_axis_off()
        ax.set_title(name, fontsize=10, pad=6)
    for i, y in enumerate(ys):
        ax = fig.add_subplot(gs[2, i])
        section_multi(ax, comps, float(y), all_points, focus=focus)

    handles = [Patch(facecolor=c, edgecolor="black", label=l)
               for (_m, c, _o, l) in components]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=14, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug-render] wrote {out_path}", flush=True)


def load_wheel(integ: Path, car: str, corner: str, max_faces: int):
    """Load one wheel, preferring the watertight ``_clean.stl`` (stage 4)
    but falling back to the pre-watertight aligned ``.stl`` (stage 2) so the
    debug renders work even when stage 4 hasn't run / timed out."""
    clean = integ / f"{car}_wheel_{corner}_clean.stl"
    raw = integ / f"{car}_wheel_{corner}.stl"
    return load(clean if clean.is_file() else raw, max_faces)


def load(path: Path, max_faces: int):
    p = Path(path)
    if not p.is_file():
        print(f"[debug-render] (missing) {p}")
        return pv.PolyData()
    m = _load_mesh(p)
    n = int(m.n_faces)
    if max_faces and max_faces > 0 and n > max_faces:
        # decimate_pro is much faster and tolerant of non-manifold /
        # dirty meshes than VTK's quadric decimate(volume_preservation),
        # which can hang on the raw combined/clean bodies.
        try:
            red = min(max(1.0 - float(max_faces) / float(n), 0.0), 0.95)
            m = m.decimate_pro(red, preserve_topology=False).clean()
        except Exception as e:  # noqa: BLE001
            print(f"[debug-render] decimate skipped for {p.name} "
                  f"({n:,} faces): {e}", flush=True)
    print(f"[debug-render] loaded {p.name}: {n:,} -> {int(m.n_faces):,} faces",
          flush=True)
    return m


def merge(meshes):
    out = pv.PolyData()
    for m in meshes:
        if m.n_points:
            out = m if out.n_points == 0 else out.merge(m)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--car", required=True, help="car base name, e.g. 0mjLG")
    ap.add_argument("--bundle", type=Path, required=True,
                    help="per-car bundle dir, e.g. outputs/0mjLG_v2")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for PNGs (default <bundle>/debug)")
    ap.add_argument("--max-faces", type=int, default=0,
                    help="decimation target per mesh (render only). 0 = no "
                         "decimation (VTK renders full-res meshes fine; "
                         "avoids the decimate hang on raw/dirty bodies). "
                         "Set e.g. 200000 to speed up if needed.")
    ap.add_argument("--extra", action="append", default=[],
                    help="extra overlay 'path#label#color' (repeatable); "
                         "rendered on top of the shell context as its own "
                         "step. Handy for shell --debug step STLs.")
    ap.add_argument("--focus", choices=("none", "front", "rear",
                                        "front_left", "front_right",
                                        "rear_left", "rear_right"),
                    default="none",
                    help="zoom every panel onto a wheelhouse instead of the "
                         "whole car (default none = full body). 'front' = "
                         "front-left wheel.")
    ap.add_argument("--zoom-mult", type=float, default=1.8,
                    help="half-window = zoom-mult x wheel radius (focus mode).")
    args = ap.parse_args()

    car = args.car
    integ = args.bundle / "integrate"
    out = args.out if args.out is not None else (args.bundle / "debug")
    mf = args.max_faces

    shell = load(integ / f"{car}_shell.stl", mf)
    ub_raw = load(integ / f"{car}_underbody.stl", mf)
    ub_trim = load(integ / f"{car}_underbody_trimmed.stl", mf)
    combined = load(integ / f"{car}_combined.stl", mf)
    # Final body: prefer watertight _clean.stl, else fall back to the
    # pre-watertight combined (shell+UB) so step [5] still renders.
    clean_p = integ / f"{car}_clean.stl"
    clean = load(clean_p, mf) if clean_p.is_file() else combined
    clean_is_fallback = not clean_p.is_file()
    cutwall = load(integ / f"{car}_cut_polygon.stl", mf)
    wheels = merge([load_wheel(integ, car, c, mf // 2)
                    for c in ("front_left", "front_right",
                              "rear_left", "rear_right")])

    # Optional wheelhouse zoom: centre + section planes from one wheel.
    focus = None
    ys = None
    sfx = ""
    if args.focus != "none":
        corner = "front_left" if args.focus == "front" else (
            "rear_left" if args.focus == "rear" else args.focus)
        wm = load_wheel(integ, car, corner, mf)
        if wm.n_points:
            b = wm.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
            center = np.array([0.5 * (b[0] + b[1]),
                               0.5 * (b[2] + b[3]),
                               0.5 * (b[4] + b[5])], dtype=np.float64)
            radius = 0.5 * float(max(b[1] - b[0], b[5] - b[4]))
            half = args.zoom_mult * radius
            focus = (center, half)
            cy = center[1]
            ys = [cy - 0.6 * radius, cy - 0.3 * radius, cy,
                  cy + 0.3 * radius, cy + 0.6 * radius]
            sfx = f"_{args.focus}zoom"
            print(f"[debug-render] focus={corner} center={center.round(1)} "
                  f"radius={radius:.0f} half={half:.0f}", flush=True)
        else:
            print(f"[debug-render] focus wheel not found ({corner}); "
                  f"rendering full body")

    zoom_note = f"  (zoom: {args.focus} wheelhouse)" if focus else ""

    def emit(title, comps, name):
        render_step(f"{car}  {title}{zoom_note}", comps,
                    out / f"{car}_{name}{sfx}.png", focus=focus, ys=ys)

    emit("[1] detected geometry: extracted shell + placed wheels",
         [(shell, COL_SHELL, 1.0, "shell (extracted)"),
          (wheels, COL_WHEEL, 1.0, "wheels (detected/placed)")],
         "dbg1_detected")
    emit("[2] generated geometry: raw parametric underbody (shell=ctx)",
         [(shell, COL_SHELL_CTX, 0.22, "shell (context)"),
          (ub_raw, COL_UB, 1.0, "underbody RAW (generated)")],
         "dbg2_underbody_raw")
    emit("[3] trim: underbody trimmed to shell rim + cut polygon box",
         [(shell, COL_SHELL_CTX, 0.22, "shell (context)"),
          (cutwall, COL_CUT, 0.7, "cut polygon (extraction box)"),
          (ub_trim, COL_UB, 1.0, "underbody TRIMMED")],
         "dbg3_trimmed")
    emit("[4] combined: shell + underbody merged (pre-watertight)",
         [(combined, COL_COMBINED, 1.0, "combined (shell+UB)")],
         "dbg4_combined")
    clean_lbl = ("combined (pre-watertight FALLBACK)" if clean_is_fallback
                 else "clean body (watertight)")
    clean_title = ("[5] final body + wheels "
                   + ("(watertight stage missing -> combined fallback)"
                      if clean_is_fallback else "(clean watertight)"))
    emit(clean_title,
         [(clean, COL_UB, 1.0, clean_lbl),
          (wheels, COL_WHEEL, 1.0, "wheels")],
         "dbg5_clean")

    # Optional ad-hoc overlays (e.g. shell --debug per-step STLs).
    for i, spec in enumerate(args.extra):
        parts = spec.split("#")
        path = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else Path(path).stem
        color = parts[2] if len(parts) > 2 and parts[2] else "#16a34a"
        m = load(Path(path), mf)
        render_step(
            f"{car}  [extra] {label}",
            [(shell, COL_SHELL_CTX, 0.22, "shell (context)"),
             (m, color, 1.0, label)],
            out / f"{car}_dbgX{i}_{label}.png")

    print(f"[debug-render] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
