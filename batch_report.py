"""Scan outputs/ and emit batch_report.csv + batch_report.md.

For each STL in DATA_DIR (default data/data_upload/), classify the
pipeline result by looking at:

    outputs/<base>/integrate/<base>_summary.json   -- definitive,
        present only if all 4 stages finished

    outputs/<base>/                                -- partial state if
                                                     summary missing

    (no outputs/<base>/)                            -- never started

States:
  complete_watertight  summary present + body is_watertight True
  complete_open        summary present + body is_watertight False
                       (clean.stl exists but mesh still has open edges)
  partial              outputs dir exists but no summary -- a stage
                       failed or is still running
  not_started          input STL has no outputs dir yet

Usage:
    python batch_report.py
    python batch_report.py --data-dir data/data_upload --csv my.csv

Re-runnable; safe to invoke while jobs are still in flight (partial
rows reflect current state).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Stage marker files (paths relative to REPO_ROOT). Detecting which
# of these exist tells us which stage produced its primary artifact
# before the run died -- "what we got out of it" rather than "where
# did it crash".
STAGE_MARKERS = [
    ("shell",       "outputs/{base}/shell/{base}_meta.json"),
    ("integrate",   "outputs/{base}/integrate/{base}_underbody_trimmed.stl"),
    ("fill_gap",    "outputs/{base}/integrate/{base}_combined.stl"),
    ("watertight",  "outputs/{base}/integrate/{base}_clean.stl"),
]


def classify(base: str) -> dict:
    summary_path = REPO_ROOT / "outputs" / base / "integrate" / f"{base}_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            return {"state": "partial", "stages_done": "summary_corrupt"}
        out = summary.get("outputs", {})
        body = out.get("body", {}) or {}
        wheels = out.get("wheels", {}) or {}
        body_wt = body.get("is_watertight")
        wheels_wt = (all(w.get("is_watertight") for w in wheels.values())
                     if wheels else None)
        return {
            "state": "complete_watertight" if body_wt else "complete_open",
            "body_watertight": body_wt,
            "body_faces": body.get("faces"),
            "body_volume_m3": body.get("volume_m3"),
            "wheels_watertight": wheels_wt,
            "n_wheels": len(wheels),
            "stages_done": "1,2,3,4",
        }

    out_dir = REPO_ROOT / "outputs" / base
    if not out_dir.exists():
        return {"state": "not_started", "stages_done": ""}

    done = []
    for s, fmt in STAGE_MARKERS:
        if (REPO_ROOT / fmt.format(base=base)).exists():
            done.append(s)
    return {"state": "partial", "stages_done": ",".join(done) or "(none)"}


def duration_min(base: str) -> float | None:
    """Pull '... pipeline complete in Xs (Y min)' out of master log."""
    log = REPO_ROOT / "outputs" / base / "logs" / f"{base}_pipeline.log"
    if not log.exists():
        return None
    # Only tail the last 4 KB -- the completion line is at EOF.
    try:
        size = log.stat().st_size
        with log.open("rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"pipeline complete in [\d.]+s.*?\(([\d.]+) min\)", tail)
    return float(m.group(1)) if m else None


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "base", "state", "body_watertight", "wheels_watertight",
        "body_faces", "body_volume_m3", "duration_min", "stages_done",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def yn(v) -> str:
    """yes / no / - for the markdown table."""
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return "-"


def write_markdown(rows: list[dict], counts: dict, path: Path,
                    show_per_car: bool) -> None:
    total = len(rows)
    with path.open("w") as f:
        f.write(f"# Batch report -- {total} cars\n\n")

        f.write("## Roll-up\n\n")
        f.write("| State | Count | % |\n|---|---:|---:|\n")
        for s in ("complete_watertight", "complete_open",
                  "partial", "not_started"):
            n = counts.get(s, 0)
            pct = (100.0 * n / total) if total else 0.0
            f.write(f"| {s} | {n} | {pct:.1f}% |\n")

        completed = counts["complete_watertight"] + counts["complete_open"]
        durs = [r["duration_min"] for r in rows
                if r.get("duration_min") is not None]
        if durs:
            durs_sorted = sorted(durs)
            n = len(durs_sorted)
            median = durs_sorted[n // 2]
            f.write(f"\nFinished pipelines: {completed}/{total}. "
                    f"Median duration: {median:.1f} min "
                    f"(min={min(durs):.1f}, max={max(durs):.1f}).\n")

        if not show_per_car:
            return

        f.write("\n## Per-car detail\n\n")
        f.write("| Car | State | Body wt | Wheels wt | "
                "Body faces | Vol m³ | Min | Stages done |\n")
        f.write("|---|---|---|---|---:|---:|---:|---|\n")
        for r in rows:
            faces_s = (f"{r['body_faces']:,}"
                       if r.get("body_faces") else "-")
            vol = r.get("body_volume_m3")
            vol_s = f"{vol:.3f}" if isinstance(vol, (int, float)) else "-"
            dur = r.get("duration_min")
            dur_s = f"{dur:.1f}" if dur is not None else "-"
            f.write(
                f"| {r['base']} | {r['state']} | "
                f"{yn(r.get('body_watertight'))} | "
                f"{yn(r.get('wheels_watertight'))} | "
                f"{faces_s} | {vol_s} | {dur_s} | "
                f"{r.get('stages_done', '')} |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/data_upload",
                     help="Directory of input STLs (default data/data_upload)")
    ap.add_argument("--csv", default="batch_report.csv",
                     help="CSV output path")
    ap.add_argument("--md", default="batch_report.md",
                     help="Markdown output path")
    ap.add_argument("--rollup-only", action="store_true",
                     help="Skip the per-car table in the Markdown "
                          "(useful when there are hundreds of cars).")
    args = ap.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    if not data_dir.is_dir():
        ap.error(f"data dir not found: {data_dir}")

    stls = sorted(data_dir.glob("*.stl"))
    if not stls:
        ap.error(f"no .stl in {data_dir}")

    rows: list[dict] = []
    counts = {"complete_watertight": 0, "complete_open": 0,
              "partial": 0, "not_started": 0}
    for stl in stls:
        base = stl.stem
        info = classify(base)
        info["base"] = base
        info["duration_min"] = duration_min(base)
        rows.append(info)
        counts[info["state"]] += 1

    csv_path = REPO_ROOT / args.csv
    md_path = REPO_ROOT / args.md
    write_csv(rows, csv_path)
    write_markdown(rows, counts, md_path,
                   show_per_car=not args.rollup_only)

    total = len(rows)
    print(f"[report] {total} cars scanned")
    for s in ("complete_watertight", "complete_open",
              "partial", "not_started"):
        n = counts[s]
        pct = 100.0 * n / total if total else 0.0
        print(f"  {s:<22s} {n:>4d}  ({pct:5.1f}%)")
    print(f"[report] wrote {csv_path}")
    print(f"[report] wrote {md_path}")


if __name__ == "__main__":
    main()
