"""End-to-end reconstructed-car pipeline driver.

Runs all four stages on one raw STL input, tees every stage's stdout
into a single master log, and writes a consolidated summary JSON at
the end. The summary bundles every geometry knob the parametric UB
used (wheel centres, tire spec, wheelhouse heights, splitter + diffuser
sections, floor z) together with the geometric outputs of the final
watertight result (per-part dimensions, triangle counts, watertight
flags, volumes).

Pipeline (each stage runs as a subprocess so stdout can be tee'd into
the master log without leaking Python state between stages):

    1. run_shell.py             outputs/shell/<base>_*
    2. integrate_underbody.py   outputs/integrate/<base>_*
    3. fill_gap.py              outputs/integrate/<base>_{gap, combined}.stl
    4. make_watertight.py       outputs/integrate/<base>_{clean, wheel_*_clean}.{stl,json}

After stage 4:
    paramub.pipeline_summary.write_pipeline_summary
        → outputs/integrate/<base>_summary.json

Logs:
    logs/<base>_pipeline.log         master log (all 4 stages, tee'd)
    logs/<base>_stage_<n>_*.log      per-stage log (also written)

Usage::

    # Default: run all 4 stages on the alfa example
    python make_pipeline.py

    # Override input
    python make_pipeline.py --input data/<other>.stl

    # Skip stage 4 (e.g. for a quick stage 1-3 iteration on the
    # login node, leaving the heavy Blender remesh for Slurm)
    python make_pipeline.py --skip-watertight

    # Slurm wrapper that requests 128 GB / 8 CPU / 3 h for stage 4
    sbatch make_pipeline.sbatch data/<base>.stl
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _stage_log_path(logs_dir: Path, base: str, stage_n: int,
                     stage_name: str) -> Path:
    return logs_dir / f"{base}_stage_{stage_n}_{stage_name}.log"


def run_stage(label: str, stage_n: int, stage_name: str,
                cmd: list[str], base: str, logs_dir: Path,
                master_log) -> bool:
    """Run a pipeline stage as a subprocess; tee its stdout/stderr into
    both a per-stage log and the master log; mirror to this process's
    stdout. Returns True on exit 0.
    """
    stage_log_path = _stage_log_path(logs_dir, base, stage_n, stage_name)
    banner = (f"\n{'=' * 78}\n"
              f"[STAGE {stage_n}] {label}\n"
              f"  cmd: {' '.join(cmd)}\n"
              f"  log: {stage_log_path}\n"
              f"  started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"{'=' * 78}\n")
    print(banner, end="", flush=True)
    master_log.write(banner)
    master_log.flush()
    t0 = time.time()
    rc = 0
    with open(stage_log_path, "w") as stage_log:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stage_log.write(line)
            master_log.write(line)
            master_log.flush()
        rc = proc.wait()
    elapsed = time.time() - t0
    footer = (f"\n[STAGE {stage_n}] {label} exit={rc}  "
              f"elapsed={elapsed:.1f}s\n")
    print(footer, end="", flush=True)
    master_log.write(footer)
    master_log.flush()
    return rc == 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data/"
                                "alfa_romeo_giuliazhuliye_2025_image10"
                                "_71415_shadowfill.stl"),
                   help="Raw whole-car STL input. Default: alfa example.")
    p.add_argument("--shell-dir", type=Path,
                   default=Path("outputs/shell"))
    p.add_argument("--integrate-dir", type=Path,
                   default=Path("outputs/integrate"))
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--skip-shell", action="store_true",
                   help="Skip stage 1 (assumes outputs/shell already populated).")
    p.add_argument("--skip-integrate", action="store_true",
                   help="Skip stage 2 (assumes outputs/integrate has the "
                        "shell/UB/wheel STLs + integrate_meta.json).")
    p.add_argument("--skip-fillgap", action="store_true",
                   help="Skip stage 3 (assumes combined.stl already present).")
    p.add_argument("--skip-watertight", action="store_true",
                   help="Skip stage 4 (the heavy Blender remesh).")
    # Stage 4 knobs forwarded
    p.add_argument("--depth", type=int, default=11,
                   help="Stage-4 octree depth for smooth remesh. Default 11. "
                        "Only used when stage-4 mode falls back to 'full'.")
    p.add_argument("--wt-mode", choices=("auto", "lightweight", "full"),
                   default="auto",
                   help="Stage-4 strategy: auto (default) picks "
                        "lightweight when stage-3 combined.stl is "
                        "already watertight (no smooth remesh; "
                        "preserves all detail), else full. Pass "
                        "--wt-mode full to force the dual-contour "
                        "remesh.")
    args = p.parse_args()

    base = args.input.stem
    logs_dir = args.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    args.shell_dir.mkdir(parents=True, exist_ok=True)
    args.integrate_dir.mkdir(parents=True, exist_ok=True)
    master_log_path = logs_dir / f"{base}_pipeline.log"

    print(f"[pipeline] base={base}")
    print(f"[pipeline] master log -> {master_log_path}")
    t0_total = time.time()

    with open(master_log_path, "w") as master_log:
        master_log.write(
            f"=== ParamUB pipeline run ===\n"
            f"base: {base}\n"
            f"input: {args.input}\n"
            f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        py = sys.executable

        if not args.skip_shell:
            ok = run_stage(
                "shell extraction (run_shell.py)", 1, "run_shell",
                [py, "run_shell.py",
                 "--input", str(args.input),
                 "--output-dir", str(args.shell_dir)],
                base, logs_dir, master_log)
            if not ok:
                sys.exit("[pipeline] stage 1 failed")

        if not args.skip_integrate:
            ok = run_stage(
                "integrate parametric UB (integrate_underbody.py)",
                2, "integrate",
                [py, "integrate_underbody.py",
                 "--shell-meta",
                 str(args.shell_dir / f"{base}_meta.json"),
                 "--output-dir", str(args.integrate_dir)],
                base, logs_dir, master_log)
            if not ok:
                sys.exit("[pipeline] stage 2 failed")

        if not args.skip_fillgap:
            ok = run_stage(
                "fill shell ↔ UB rim gap (fill_gap.py)", 3, "fill_gap",
                [py, "fill_gap.py",
                 "--shell",
                 str(args.integrate_dir / f"{base}_shell.stl"),
                 "--ub",
                 str(args.integrate_dir / f"{base}_underbody_trimmed.stl"),
                 "--out",
                 str(args.integrate_dir / f"{base}_gap.stl"),
                 "--combined-out",
                 str(args.integrate_dir / f"{base}_combined.stl"),
                 "--repair", "--repair-close-holes-max", "10000",
                 "--repair-remesh-mm", "6",
                 "--repair-remesh-iters", "3"],
                base, logs_dir, master_log)
            if not ok:
                sys.exit("[pipeline] stage 3 failed")

        if not args.skip_watertight:
            ok = run_stage(
                "watertight finisher (make_watertight.py)",
                4, "watertight",
                [py, "make_watertight.py",
                 "--input",
                 str(args.integrate_dir / f"{base}_combined.stl"),
                 "--output",
                 str(args.integrate_dir / f"{base}_clean.stl"),
                 "--fixed-out",
                 str(args.integrate_dir / f"{base}_fixed.stl"),
                 "--json",
                 str(args.integrate_dir / f"{base}_clean.json"),
                 "--reference",
                 str(args.integrate_dir / f"{base}_combined.stl"),
                 "--mode", args.wt_mode,
                 "--depth", str(args.depth)],
                base, logs_dir, master_log)
            if not ok:
                sys.exit("[pipeline] stage 4 failed")

        # Consolidated summary across all stages.
        from paramub.pipeline_summary import write_pipeline_summary
        summary_path = write_pipeline_summary(
            base,
            shell_dir=args.shell_dir,
            integrate_dir=args.integrate_dir,
        )
        elapsed_total = time.time() - t0_total
        footer = (
            f"\n=== pipeline complete in {elapsed_total:.1f}s "
            f"({elapsed_total / 60:.1f} min) ===\n"
            f"summary: {summary_path}\n"
            f"master log: {master_log_path}\n")
        print(footer, end="", flush=True)
        master_log.write(footer)


if __name__ == "__main__":
    main()
