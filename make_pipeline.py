"""End-to-end reconstructed-car pipeline driver.

Runs all four stages on one raw STL input, tees every stage's stdout
into a single master log, and writes a consolidated summary JSON at
the end. The summary bundles every geometry knob the parametric UB
used (wheel centres, tire spec, wheelhouse heights, splitter + diffuser
sections, floor z) together with the geometric outputs of the final
watertight result (per-part dimensions, triangle counts, watertight
flags, volumes).

Per-car output layout (default):

    outputs/<base>/
    ├── shell/                  stage 1 outputs
    ├── integrate/              stages 2-4 outputs (shell, UB, wheels,
    │                           gap, combined, fixed, clean,
    │                           wheel_<corner>_clean STLs + JSONs)
    │   └── <base>_summary.json consolidated geometry + verify report
    └── logs/                   one master log + one per-stage log

Pipeline (each stage runs as a subprocess so stdout can be tee'd into
the master log without leaking Python state between stages):

    1. run_shell.py             → outputs/<base>/shell/<base>_*
    2. integrate_underbody.py   → outputs/<base>/integrate/<base>_*
    3. fill_gap.py              → outputs/<base>/integrate/<base>_{gap, combined}.stl
    4. make_watertight.py       → outputs/<base>/integrate/<base>_{clean, wheel_*_clean}.{stl,json}

After stage 4:
    paramub.pipeline.summary.write_pipeline_summary
        → outputs/<base>/integrate/<base>_summary.json

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
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _stage_log_path(logs_dir: Path, base: str, stage_n: int,
                     stage_name: str) -> Path:
    return logs_dir / f"{base}_stage_{stage_n}_{stage_name}.log"


def run_stage(label: str, stage_n: int, stage_name: str,
                cmd: list[str], base: str, logs_dir: Path,
                master_log, timeout_s: float | None = None) -> bool:
    """Run a pipeline stage as a subprocess; tee its stdout/stderr into
    both a per-stage log and the master log; mirror to this process's
    stdout. Returns True on exit 0.

    ``timeout_s`` arms a watchdog that kills the stage subprocess (and
    its children) if it runs longer than the limit. Pathological cars
    hang for hours inside a single native call (OCC ``exportStl``,
    MeshLab ``close_holes`` Liepa DP, Blender NM-fix ops) and would
    otherwise block a Slurm concurrency slot for the whole job
    walltime. On timeout the stage returns False -> the pipeline drops
    the car and the slot frees immediately.
    """
    stage_log_path = _stage_log_path(logs_dir, base, stage_n, stage_name)
    tmo_txt = f"  timeout: {timeout_s:.0f}s\n" if timeout_s else ""
    banner = (f"\n{'=' * 78}\n"
              f"[STAGE {stage_n}] {label}\n"
              f"  cmd: {' '.join(cmd)}\n"
              f"  log: {stage_log_path}\n"
              f"{tmo_txt}"
              f"  started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"{'=' * 78}\n")
    print(banner, end="", flush=True)
    master_log.write(banner)
    master_log.flush()
    t0 = time.time()
    rc = 0
    timed_out = {"hit": False}
    with open(stage_log_path, "w") as stage_log:
        # start_new_session so the watchdog can kill the whole process
        # group (the stage spawns its own children, e.g. Blender).
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        assert proc.stdout is not None

        watchdog = None
        if timeout_s:
            import os
            import signal

            def _kill_on_timeout():
                timed_out["hit"] = True
                msg = (f"\n[STAGE {stage_n}] {label} TIMEOUT after "
                       f"{timeout_s:.0f}s -- killing process group\n")
                sys.stdout.write(msg)
                sys.stdout.flush()
                master_log.write(msg)
                master_log.flush()
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()

            watchdog = threading.Timer(timeout_s, _kill_on_timeout)
            watchdog.start()

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stage_log.write(line)
            stage_log.flush()
            master_log.write(line)
            master_log.flush()
        rc = proc.wait()
        if watchdog is not None:
            watchdog.cancel()
    elapsed = time.time() - t0
    status = "TIMEOUT" if timed_out["hit"] else f"exit={rc}"
    footer = (f"\n[STAGE {stage_n}] {label} {status}  "
              f"elapsed={elapsed:.1f}s\n")
    print(footer, end="", flush=True)
    master_log.write(footer)
    master_log.flush()
    return rc == 0 and not timed_out["hit"]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data/"
                                "alfa_romeo_giuliazhuliye_2025_image10"
                                "_71415_shadowfill.stl"),
                   help="Raw whole-car STL input. Default: alfa example.")
    p.add_argument("--outputs-root", type=Path,
                   default=Path("outputs"),
                   help="Root directory under which a per-car bundle "
                        "<outputs-root>/<base>/{shell, integrate, logs}/ "
                        "is created. Default: outputs.")
    p.add_argument("--shell-dir", type=Path, default=None,
                   help="Override stage-1 output dir. Defaults to "
                        "<outputs-root>/<base>/shell.")
    p.add_argument("--integrate-dir", type=Path, default=None,
                   help="Override stages 2-4 output dir. Defaults to "
                        "<outputs-root>/<base>/integrate.")
    p.add_argument("--logs-dir", type=Path, default=None,
                   help="Override logs dir. Defaults to "
                        "<outputs-root>/<base>/logs.")
    p.add_argument("--skip-shell", action="store_true",
                   help="Skip stage 1 (assumes outputs/shell already populated).")
    p.add_argument("--skip-integrate", action="store_true",
                   help="Skip stage 2 (assumes outputs/integrate has the "
                        "shell/UB/wheel STLs + integrate_meta.json).")
    p.add_argument("--skip-fillgap", action="store_true",
                   help="Skip stage 3 (assumes combined.stl already present).")
    p.add_argument("--skip-watertight", action="store_true",
                   help="Skip stage 4 (the heavy Blender remesh).")
    # Per-stage watchdog timeouts (minutes). 0 disables a stage's
    # watchdog. Defaults are ~10x the typical stage time so only
    # pathological cars (which hang for hours in a single native call)
    # get dropped, freeing the Slurm slot instead of blocking it for
    # the whole job walltime. See the py-spy diagnosis: stage-2 hangs
    # in OCC exportStl, stage-3 in MeshLab close_holes, stage-4 in the
    # Blender NM-fix loop.
    p.add_argument("--timeout-shell-min", type=float, default=30.0,
                   help="Stage-1 (run_shell) watchdog, minutes. Default 30.")
    p.add_argument("--timeout-integrate-min", type=float, default=15.0,
                   help="Stage-2 (integrate_underbody) watchdog, minutes. "
                        "Default 15 (typical ~1.5 min).")
    p.add_argument("--timeout-fillgap-min", type=float, default=15.0,
                   help="Stage-3 (fill_gap) watchdog, minutes. Default 15 "
                        "(typical ~1.5 min).")
    p.add_argument("--timeout-watertight-min", type=float, default=45.0,
                   help="Stage-4 (make_watertight) watchdog, minutes. "
                        "Default 45 (legit big cars take up to ~35 min).")
    # Stage 1 orientation override
    p.add_argument("--force-flip", action="store_true",
                   help="Force the 180° front/rear flip in stage 1 (for "
                        "human-confirmed reversals the greenhouse-x cue can't "
                        "detect, e.g. cab-at-+x pickups).")
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
    p.add_argument("--no-verify-wheels", action="store_true",
                   help="Forward to stage 4: skip the per-wheel "
                        "surface-distance verification (wheels are still made "
                        "watertight). Saves the 4x2 distance samplings.")
    args = p.parse_args()

    base = args.input.stem
    # Per-car bundle: outputs/<base>/{shell, integrate, logs}/ by
    # default. Each can be overridden individually.
    bundle = args.outputs_root / base
    shell_dir = args.shell_dir if args.shell_dir is not None else bundle / "shell"
    integrate_dir = args.integrate_dir if args.integrate_dir is not None else bundle / "integrate"
    logs_dir = args.logs_dir if args.logs_dir is not None else bundle / "logs"
    args.shell_dir = shell_dir
    args.integrate_dir = integrate_dir
    args.logs_dir = logs_dir
    shell_dir.mkdir(parents=True, exist_ok=True)
    integrate_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
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
        # -u forces unbuffered stdout/stderr so the per-stage log
        # streams line-by-line into both the per-stage file AND the
        # tee'd master log. Without it Python block-buffers stdout
        # when it isn't a tty, and the log can sit empty for tens of
        # minutes while the subprocess silently chews on a big
        # tessellation pass.
        py_unbuf = [py, "-u"]

        def _tmo(minutes: float) -> float | None:
            return minutes * 60.0 if minutes and minutes > 0 else None

        if not args.skip_shell:
            ok = run_stage(
                "shell extraction (run_shell.py)", 1, "run_shell",
                py_unbuf + ["run_shell.py",
                 "--input", str(args.input),
                 "--output-dir", str(args.shell_dir)]
                + (["--force-flip"] if args.force_flip else []),
                base, logs_dir, master_log,
                timeout_s=_tmo(args.timeout_shell_min))
            if not ok:
                sys.exit("[pipeline] stage 1 failed")

        if not args.skip_integrate:
            ok = run_stage(
                "integrate parametric UB (integrate_underbody.py)",
                2, "integrate",
                py_unbuf + ["integrate_underbody.py",
                 "--shell-meta",
                 str(args.shell_dir / f"{base}_meta.json"),
                 "--output-dir", str(args.integrate_dir)],
                base, logs_dir, master_log,
                timeout_s=_tmo(args.timeout_integrate_min))
            if not ok:
                sys.exit("[pipeline] stage 2 failed")

        if not args.skip_fillgap:
            ok = run_stage(
                "fill shell ↔ UB rim gap (fill_gap.py)", 3, "fill_gap",
                py_unbuf + ["fill_gap.py",
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
                base, logs_dir, master_log,
                timeout_s=_tmo(args.timeout_fillgap_min))
            if not ok:
                sys.exit("[pipeline] stage 3 failed")

        if not args.skip_watertight:
            ok = run_stage(
                "watertight finisher (make_watertight.py)",
                4, "watertight",
                py_unbuf + ["make_watertight.py",
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
                 "--depth", str(args.depth)]
                + (["--no-verify-wheels"] if args.no_verify_wheels else []),
                base, logs_dir, master_log,
                timeout_s=_tmo(args.timeout_watertight_min))
            if not ok:
                sys.exit("[pipeline] stage 4 failed")

        # Consolidated summary across all stages.
        from paramub.pipeline.summary import write_pipeline_summary
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
