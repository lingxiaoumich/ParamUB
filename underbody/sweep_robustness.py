"""Robustness sweep: build 10 variations of (ride_height, diffuser_start_x,
diffuser_radius) and emit STL + iso PNG + log per case.

Each case is isolated: any failure is captured to that case's log and the
sweep continues. Skips trimesh repair / 5-view renders to keep wall-time
reasonable.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cadquery as cq
from cadquery import exporters

from underbody.build import UnderbodySpec, build_underbody


SWEEP_DIR = REPO_ROOT / "outputs" / "underbody" / "sweep"
ISO_RENDERER = Path(__file__).resolve().parent / "_render_iso.py"


# 10 variations covering ride height, diffuser start X, and diffuser radius.
# Defaults: ride_height_mm=100, diffuser_start_x_mm=None (=rear axle=-1350),
# diffuser_radius_mm=500.
CASES = [
    # baseline
    {"name": "01_default",                "spec": {}},
    # ride height extremes
    {"name": "02_ride_low_50",            "spec": {"ride_height_mm": 50}},
    {"name": "03_ride_high_200",          "spec": {"ride_height_mm": 200}},
    {"name": "04_ride_very_low_25",       "spec": {"ride_height_mm": 25}},
    # diffuser radius extremes
    {"name": "05_diff_radius_100",        "spec": {"diffuser_radius_mm": 100}},
    {"name": "06_diff_radius_1500",       "spec": {"diffuser_radius_mm": 1500}},
    # diffuser start position
    {"name": "07_diff_start_x_-800",      "spec": {"diffuser_start_x_mm": -800}},
    {"name": "08_diff_start_x_-1800",     "spec": {"diffuser_start_x_mm": -1800}},
    # combos
    {"name": "09_low_ride_huge_radius",   "spec": {"ride_height_mm": 60,
                                                    "diffuser_radius_mm": 1200}},
    {"name": "10_high_ride_diff_forward", "spec": {"ride_height_mm": 180,
                                                    "diffuser_start_x_mm": -1000,
                                                    "diffuser_radius_mm": 800}},
]


class Tee(io.TextIOBase):
    """Mirror writes to multiple streams (one of them is the per-case log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def run_case(case: dict) -> dict:
    name = case["name"]
    overrides = case["spec"]
    stl_path = SWEEP_DIR / f"{name}.stl"
    iso_path = SWEEP_DIR / f"{name}_iso.png"
    log_path = SWEEP_DIR / f"{name}.log"

    record = {
        "name": name,
        "spec": overrides,
        "stl_path": str(stl_path),
        "iso_path": str(iso_path),
        "log_path": str(log_path),
        "status": "failed",
    }

    with log_path.open("w") as log:
        tee = Tee(sys.__stdout__, log)
        t0 = time.time()
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"=== {name} ===")
                print(f"overrides: {overrides}")
                spec = UnderbodySpec(**overrides)
                print(f"resolved spec: stl_tolerance={spec.stl_tolerance}, "
                      f"ride_height_mm={spec.ride_height_mm}, "
                      f"diffuser_start_x_mm={spec.diffuser_start_x_mm}, "
                      f"diffuser_radius_mm={spec.diffuser_radius_mm}")

                t_build0 = time.time()
                asy, layout = build_underbody(spec)
                t_build = time.time() - t_build0
                print(f"build  : {t_build:.1f}s")
                print(f"layout : {layout}")

                t_export0 = time.time()
                compound = asy.toCompound()
                exporters.export(
                    compound,
                    str(stl_path),
                    exporters.ExportTypes.STL,
                    tolerance=spec.stl_tolerance,
                    angularTolerance=spec.stl_tolerance,
                )
                t_export = time.time() - t_export0
                size_mb = stl_path.stat().st_size / (1024 * 1024)
                print(f"export : {t_export:.1f}s  ->  {stl_path.name}  "
                      f"({size_mb:.1f} MB)")
                record.update({
                    "build_s": round(t_build, 1),
                    "export_s": round(t_export, 1),
                    "stl_mb": round(size_mb, 1),
                })

                # Iso render in a fresh subprocess (PyVista vs OCP conflict).
                payload = json.dumps({
                    "stl_path": str(stl_path.resolve()),
                    "output_path": str(iso_path.resolve()),
                })
                t_render0 = time.time()
                result = subprocess.run(
                    [sys.executable, str(ISO_RENDERER), payload],
                    check=False, capture_output=True, text=True,
                )
                t_render = time.time() - t_render0
                if result.stdout:
                    print(f"render stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"render stderr:\n{result.stderr}")
                if result.returncode != 0:
                    print(f"[warn] iso render returned {result.returncode}")
                    record.update({
                        "status": "render_failed",
                        "render_s": round(t_render, 1),
                    })
                else:
                    print(f"render : {t_render:.1f}s  ->  {iso_path.name}")
                    record.update({
                        "status": "ok",
                        "render_s": round(t_render, 1),
                    })
        except Exception as exc:
            log.write(f"\n=== EXCEPTION ===\n{type(exc).__name__}: {exc}\n")
            log.write(traceback.format_exc())
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["total_s"] = round(time.time() - t0, 1)
            log.write(f"\n=== status: {record['status']}  "
                      f"(total {record['total_s']}s) ===\n")
    return record


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"sweep output dir: {SWEEP_DIR}\n")

    rows = []
    for case in CASES:
        print(f">>> {case['name']} (spec={case['spec']})")
        rows.append(run_case(case))
        print()

    summary_path = SWEEP_DIR / "_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nsummary -> {summary_path}\n")

    print("=== SWEEP SUMMARY ===")
    print(f"{'case':<32}  {'status':>14}  {'STL MB':>8}  {'total s':>8}")
    print(f"{'-'*32}  {'-'*14}  {'-'*8}  {'-'*8}")
    for r in rows:
        size = f"{r.get('stl_mb', '-')}" if r.get("stl_mb") else "  -"
        print(f"{r['name']:<32}  {r['status']:>14}  "
              f"{size:>8}  {r.get('total_s', '-'):>8}")


if __name__ == "__main__":
    main()
