"""Tile per-car view PNGs into one contact sheet per view.

Reads the per-view PNGs that render_paramub.py writes under
    outputs/summary/views/<view_slug>/<car>.png
and produces one contact sheet per view:
    outputs/summary/contact_<view_slug>.png

Default views: bottom_front_iso, bottom, bottom_rear_iso.

Each tile is labelled with the car name and (when available) its
watertight flag from the car's summary.json, so the sheet doubles as
a quick pass/fail visual audit.

Usage:
    python make_contact_sheets.py
    python make_contact_sheets.py --cols 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VIEWS = ["bottom_front_iso", "bottom", "bottom_rear_iso"]


def watertight_flag(car: str) -> str:
    summary = REPO_ROOT / "outputs" / car / "integrate" / f"{car}_summary.json"
    if not summary.is_file():
        return "?"
    try:
        d = json.loads(summary.read_text())
        wt = d.get("outputs", {}).get("body", {}).get("is_watertight")
        return "wt" if wt else "open" if wt is False else "?"
    except (json.JSONDecodeError, OSError):
        return "?"


def build_sheet(view_slug: str, out_dir: Path, cols: int) -> Path | None:
    view_dir = out_dir / "views" / view_slug
    pngs = sorted(view_dir.glob("*.png"))
    if not pngs:
        print(f"[contact] no PNGs in {view_dir} -- skipping {view_slug}")
        return None

    n = len(pngs)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 2.4, rows * 1.9), dpi=130)
    gs = fig.add_gridspec(rows, cols, wspace=0.04, hspace=0.18)
    for i, png in enumerate(pngs):
        car = png.stem
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(mpimg.imread(png))
        ax.set_axis_off()
        ax.set_title(f"{car} [{watertight_flag(car)}]", fontsize=6, pad=2)
    title = view_slug.replace("_", " ").title()
    fig.suptitle(f"ParamUB contact sheet -- {title}  ({n} cars)",
                 fontsize=14, y=0.997)
    out_path = out_dir / f"contact_{view_slug}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[contact] wrote {out_path}  ({n} cars, {rows}x{cols} grid)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "summary")
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS,
                     help="view slugs to build sheets for")
    ap.add_argument("--cols", type=int, default=10, help="tiles per row")
    args = ap.parse_args()
    for v in args.views:
        build_sheet(v, args.out, args.cols)


if __name__ == "__main__":
    main()
