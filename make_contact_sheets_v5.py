"""Contact sheet generator for batch_v5.

Tiles per-car view PNGs from
    outputs/batch_v5_summary/views/<view_slug>/<car>.png
into paginated contact sheets:
    outputs/batch_v5_summary/contact_<view_slug>_page<N>.png

Layout: 5 columns × 20 rows = 100 cars per page.
Resolution: 4× linear scale vs the original contact sheets
    (old: figsize tile 2.4×1.9 in at 130 dpi → 312×247 px/tile;
     new: same figsize at 520 dpi → 1248×988 px/tile → 6240×19760 px/page).

Each tile is labelled with the car name and watertight flag from its
<car>_clean.json (if present).

Usage:
    python make_contact_sheets_v5.py
    python make_contact_sheets_v5.py --views bottom_front_iso bottom
    python make_contact_sheets_v5.py --cols 5 --rows 20 --dpi 520
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
DEFAULT_BATCH_OUT = REPO_ROOT / "outputs" / "batch_v5"
DEFAULT_SUMMARY   = REPO_ROOT / "outputs" / "batch_v5_summary"


def watertight_flag(car: str, batch_out: Path) -> str:
    report = batch_out / car / "integrate" / f"{car}_clean.json"
    if not report.is_file():
        return "?"
    try:
        d = json.loads(report.read_text())
        wt = d.get("is_watertight")
        return "wt" if wt else "open" if wt is False else "?"
    except (json.JSONDecodeError, OSError):
        return "?"


def build_sheets(
    view_slug: str,
    summary_dir: Path,
    batch_out: Path,
    cols: int,
    rows: int,
    dpi: int,
) -> list[Path]:
    view_dir = summary_dir / "views" / view_slug
    pngs = sorted(view_dir.glob("*.png"))
    if not pngs:
        print(f"[contact] no PNGs in {view_dir} -- skipping {view_slug}")
        return []

    per_page = cols * rows
    pages = [pngs[i:i + per_page] for i in range(0, len(pngs), per_page)]
    total = len(pngs)
    out_paths = []

    # tile size in inches — same as the old script so that × dpi gives the
    # per-tile pixel count (old dpi=130 → 312×247 px; new dpi=520 → 1248×988).
    tile_w, tile_h = 2.4, 1.9

    for page_idx, page_pngs in enumerate(pages):
        page_rows = (len(page_pngs) + cols - 1) // cols
        fig = plt.figure(
            figsize=(cols * tile_w, page_rows * tile_h),
            dpi=dpi,
        )
        gs = fig.add_gridspec(page_rows, cols, wspace=0.04, hspace=0.18)
        for i, png in enumerate(page_pngs):
            car = png.stem
            ax = fig.add_subplot(gs[i // cols, i % cols])
            ax.imshow(mpimg.imread(png))
            ax.set_axis_off()
            wt = watertight_flag(car, batch_out)
            ax.set_title(f"{car} [{wt}]", fontsize=6, pad=2)

        n_page = len(page_pngs)
        n_pages = len(pages)
        title = view_slug.replace("_", " ").title()
        fig.suptitle(
            f"ParamUB v5 -- {title}   page {page_idx+1}/{n_pages}"
            f"  (cars {page_idx*per_page+1}–{page_idx*per_page+n_page}"
            f" of {total})",
            fontsize=14, y=0.997,
        )

        suffix = f"_page{page_idx+1:03d}" if len(pages) > 1 else ""
        out_path = summary_dir / f"contact_{view_slug}{suffix}.png"
        if out_path.exists():
            print(f"[contact] skip {out_path.name}  (already exists)")
            plt.close(fig)
            out_paths.append(out_path)
            continue
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[contact] wrote {out_path}  "
              f"({n_page} cars, {page_rows}×{cols} grid, {dpi} dpi)")
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY,
                    help="Dir with views/ subdir (default outputs/batch_v5_summary).")
    ap.add_argument("--batch-out", type=Path, default=DEFAULT_BATCH_OUT,
                    help="Batch root for watertight flag lookup "
                         "(default outputs/batch_v5).")
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)
    ap.add_argument("--cols", type=int, default=5, help="Tiles per row (default 5).")
    ap.add_argument("--rows", type=int, default=20,
                    help="Max rows per page / cars per sheet (default 20 → 100/page).")
    ap.add_argument("--dpi", type=int, default=520,
                    help="Output DPI (default 520 = 4× the original 130).")
    args = ap.parse_args()

    for v in args.views:
        build_sheets(v, args.summary_dir, args.batch_out,
                     args.cols, args.rows, args.dpi)


if __name__ == "__main__":
    main()
