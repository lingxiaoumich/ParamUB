"""Unified entry point for ParamUB.

One function — :func:`generate` — takes an :class:`~paramub.ub_assem.UnderbodySpec`
and writes outputs to ``output_dir``. The ``output_mode`` argument picks the
output bundle:

  ``"stl"``: writes only ``<stem>.stl``.
  ``"all"``: writes ``<stem>.stl`` + one 8-panel composite preview PNG +
             ``<stem>.log`` (mirroring stdout/stderr) + ``<stem>.json``
             (parameter dump).

Independently, ``output_parts=True`` also writes the floor and four
wheels as separate STLs with FIXED filenames:

  ``FLOOR.STL``, ``WHEEL_FL.STL``, ``WHEEL_FR.STL``,
  ``WHEEL_RL.STL``, ``WHEEL_RR.STL``

(always uppercase, regardless of ``stem``).

The renderer runs in a subprocess (PyVista / OCP llvmpipe conflict; see
:mod:`paramub._render_views`).
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
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Literal, Optional

import cadquery as cq
from cadquery import exporters

from .ub_assem import UnderbodySpec, build_underbody


OutputMode = Literal["stl", "all"]

PACKAGE_DIR = Path(__file__).resolve().parent
RENDER_SCRIPT = PACKAGE_DIR / "_render_views.py"

# STL tessellation tolerances.
#   linear  (mm)   controls flat-surface chord error.
#   angular (rad)  controls curved-surface chord step.
#
# Global default is 0.1 rad (≈5.7°) — matches legacy behavior and keeps
# wheel/tire export fast (their many small curves dominate cost).
#
# The FLOOR per-part export uses a separate, finer angular tolerance
# (DEFAULT_FLOOR_ANGULAR_TOLERANCE_RAD) so the multisection-diffuser
# Bezier loft tessellates smoothly without paying the cost on the wheels.
DEFAULT_STL_TOLERANCE_MM = 0.1
DEFAULT_STL_ANGULAR_TOLERANCE_RAD = 0.1
DEFAULT_FLOOR_ANGULAR_TOLERANCE_RAD = 0.02


class _Tee(io.TextIOBase):
    """Mirror writes to multiple streams (one is the log file)."""
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


def _spec_to_dict(spec) -> dict:
    """Recursively convert a dataclass spec tree to plain dicts."""
    if is_dataclass(spec):
        return {k: _spec_to_dict(v) for k, v in asdict(spec).items()}
    if isinstance(spec, dict):
        return {k: _spec_to_dict(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)):
        return [_spec_to_dict(v) for v in spec]
    return spec


def _export_stl(asy: cq.Assembly, stl_path: Path, tolerance: float,
                 angular_tolerance: float) -> None:
    compound = asy.toCompound()
    exporters.export(
        compound,
        str(stl_path),
        exporters.ExportTypes.STL,
        tolerance=tolerance,
        angularTolerance=angular_tolerance,
    )


def _premesh_body(asy: cq.Assembly, linear_tol: float,
                  body_angular_tol: float) -> None:
    """Pre-mesh the 'body' child at a finer angular tolerance.

    BRepMesh refuses to refine an existing mesh when a coarser tolerance is
    later requested, so the merged STL exporter (run at the global, coarser
    angular tolerance) would otherwise lock in a coarse mesh for the body
    too. Running the fine mesh first locks the finer one in before that
    happens; the coarse export sees an already-meshed body and only meshes
    the wheels.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    for child in asy.children:
        if child.name == "body":
            BRepMesh_IncrementalMesh(
                child.obj.wrapped, linear_tol, False,
                body_angular_tol, True)


def _export_step_assembly(asy: cq.Assembly, step_path: Path) -> None:
    """Write the whole assembly as a single STEP file with named parts."""
    # Assembly.save() preserves child names + colors in the STEP file.
    asy.save(str(step_path), exportType="STEP")


def _wheel_meta_lines(label: str, wheel) -> list[str]:
    """Tire + spoke parameter summary for one wheel."""
    t = wheel.tire
    s = wheel.spoke
    return [
        f"{label}:",
        f"  tire size           = {t.section_width_mm:.0f}/"
        f"{t.aspect_ratio:.0f} R{t.rim_diameter_in:.0f}  "
        f"(OD = {t.tire_od_mm:.1f} mm)",
        f"  sidewall bulge/pos  = {t.sidewall_bulge_mm:.1f} mm @ "
        f"pos {t.sidewall_bulge_pos:.2f}",
        f"  shoulder_radius_mm  = {t.shoulder_radius_mm:.1f}",
        f"  tread_width/crown_R = {t.tread_width_mm:.0f} / "
        f"{t.crown_radius_mm:.0f}",
        f"  rim_flange_mm       = {t.rim_flange_mm:.1f}",
        f"  wheel_width / ET    = {s.wheel_width_mm:.0f} / "
        f"{s.wheel_offset_mm:.0f}",
        f"  hub_bore_mm         = {s.hub_bore_mm:.0f}",
        f"  dish depth/profile  = {s.dish_depth_mm:.1f} / {s.dish_profile}",
        f"  num_spokes          = {s.num_spokes}  "
        f"(hub/rim width {s.spoke_width_hub_mm:.0f}/"
        f"{s.spoke_width_rim_mm:.0f})",
        f"  window gaps in/out  = {s.window_inner_gap_mm:.0f} / "
        f"{s.window_outer_gap_mm:.0f}  "
        f"(corner R={s.window_corner_radius_mm:.0f})",
        f"  rim_band/flange_t   = {s.rim_band_width_mm:.0f} / "
        f"{s.flange_axial_thickness_mm:.0f}",
        f"  spoke thick/crown   = {s.spoke_thickness_mm:.0f} / "
        f"{s.spoke_outer_crown_mm:.0f}",
    ]


def _meta_lines(spec: UnderbodySpec, layout: dict, stl_path: Path) -> list[str]:
    size_mb = stl_path.stat().st_size / (1024 * 1024)
    lines = [
        "[underbody]",
        f"  wheelbase_mm        = {spec.wheelbase_mm}",
        f"  track F/R (mm)      = {spec.track_front_mm} / "
        f"{spec.track_rear_mm}",
        f"  overhang F/R (mm)   = {spec.front_overhang_mm} / "
        f"{spec.rear_overhang_mm}",
        f"  overall length (mm) = {layout['overall_length_mm']:.0f}",
        f"  floor_width_mm      = {spec.floor_width_mm}",
        f"  ride_height_mm      = {spec.ride_height_mm}",
        f"  floor_angle_deg     = {spec.floor_angle_deg}",
        f"  diffuser_start_x_mm = {layout['diff_start_x']:.0f}",
        f"  diffuser angle/R    = {spec.diffuser_angle_deg}° / "
        f"R{spec.diffuser_radius_mm}",
        f"  camber F/R (deg)    = {spec.camber_front_deg} / "
        f"{spec.camber_rear_deg}",
        f"  toe F/R (deg)       = {spec.toe_front_deg} / {spec.toe_rear_deg}",
        f"  wheelhouse fillet F/R = "
        f"{spec.front_wheel_house_fillet_mm:.0f} / "
        f"{spec.rear_wheel_house_fillet_mm:.0f}",
        "",
    ]
    if spec.rear_wheel is None:
        lines.extend(_wheel_meta_lines("[wheel — front & rear]", spec.wheel))
    else:
        lines.extend(_wheel_meta_lines("[front wheel]", spec.wheel))
        lines.append("")
        lines.extend(_wheel_meta_lines("[rear wheel]", spec.rear_wheel))
    lines.append("")
    lines.append(f"STL: {stl_path.name}  ({size_mb:.1f} MB)")
    return lines


def _axle_payload(spec: UnderbodySpec, layout: dict) -> dict:
    tire_r_f = spec.wheel.tire_radius_mm
    tire_r_r = (spec.rear_wheel.tire_radius_mm
                if spec.rear_wheel is not None else tire_r_f)
    return {
        "front_axle_x": layout["front_axle_x"],
        "rear_axle_x": layout["rear_axle_x"],
        "track_front_mm": spec.track_front_mm,
        "track_rear_mm": spec.track_rear_mm,
        "tire_radius_f_mm": tire_r_f,
        "tire_radius_r_mm": tire_r_r,
    }


def _spawn_renderer(stl_path: Path, output_dir: Path,
                    meta_lines: list[str], preview_name: str,
                    axles: dict) -> None:
    payload = json.dumps({
        "stl_path": str(stl_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "meta_lines": meta_lines,
        "preview_name": preview_name,
        "axles": axles,
    })
    print(f"[render] spawning subprocess renderer ...")
    result = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), payload],
        check=False, cwd=str(output_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(f"render subprocess failed: {result.returncode}")


# Fixed mapping from assembly child name -> base filename (without extension)
# for per-part exports. Extension is appended per format (.STL / .STEP).
_PART_BASENAME = {
    "body":     "FLOOR",
    "wheel_fl": "WHEEL_FL",
    "wheel_fr": "WHEEL_FR",
    "wheel_rl": "WHEEL_RL",
    "wheel_rr": "WHEEL_RR",
}


def _export_parts_stl(asy: cq.Assembly, output_dir: Path,
                       tolerance: float, angular_tolerance: float,
                       floor_angular_tolerance: float) -> dict[str, str]:
    """Write one STL per named child of ``asy`` with fixed UPPERCASE names.

    The FLOOR (body) part uses ``floor_angular_tolerance`` so the
    multisection-diffuser Bezier loft tessellates finely; all other
    parts (wheels) use the global ``angular_tolerance``.
    """
    written: dict[str, str] = {}
    for child in asy.children:
        base = _PART_BASENAME.get(child.name)
        if base is None:
            print(f"  [warn] no filename mapping for child {child.name!r}, "
                  "skipping STL part")
            continue
        sub = cq.Assembly()
        sub.add(child.obj, name=child.name)
        out_path = output_dir / f"{base}.STL"
        part_ang = (floor_angular_tolerance if child.name == "body"
                    else angular_tolerance)
        exporters.export(
            sub.toCompound(),
            str(out_path),
            exporters.ExportTypes.STL,
            tolerance=tolerance,
            angularTolerance=part_ang,
        )
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  part   : {out_path.name}  ({size_mb:.1f} MB)")
        written[child.name] = str(out_path)
    return written


def _export_parts_step(asy: cq.Assembly, output_dir: Path) -> dict[str, str]:
    """Write one STEP per named child of ``asy`` with fixed UPPERCASE names."""
    written: dict[str, str] = {}
    for child in asy.children:
        base = _PART_BASENAME.get(child.name)
        if base is None:
            print(f"  [warn] no filename mapping for child {child.name!r}, "
                  "skipping STEP part")
            continue
        sub = cq.Assembly()
        sub.add(child.obj, name=child.name)
        out_path = output_dir / f"{base}.STEP"
        sub.save(str(out_path), exportType="STEP")
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  part   : {out_path.name}  ({size_mb:.1f} MB)")
        written[child.name] = str(out_path)
    return written


def generate(
    spec: Optional[UnderbodySpec] = None,
    output_dir: str | os.PathLike = "outputs",
    stem: str = "underbody",
    output_mode: OutputMode = "stl",
    output_parts: bool = False,
    export_step: bool = False,
    stl_tolerance_mm: float = DEFAULT_STL_TOLERANCE_MM,
    stl_angular_tolerance_rad: float = DEFAULT_STL_ANGULAR_TOLERANCE_RAD,
    floor_angular_tolerance_rad: float = DEFAULT_FLOOR_ANGULAR_TOLERANCE_RAD,
    half_only: bool = False,
) -> dict:
    """Build the underbody and write outputs.

    Parameters
    ----------
    spec:
        UnderbodySpec. If None, uses defaults.
    output_dir:
        Directory to write to (created if missing).
    stem:
        File stem for stl / log / json / preview.
    output_mode:
        ``"stl"`` -> just the merged STL file.
        ``"all"`` -> merged STL + 8-panel composite preview + log + json.
    output_parts:
        If True, ALSO write per-part STLs with fixed UPPERCASE names:
        ``FLOOR.STL``, ``WHEEL_FL.STL``, ``WHEEL_FR.STL``, ``WHEEL_RL.STL``,
        ``WHEEL_RR.STL``. Independent of ``output_mode``.
    export_step:
        If True, ALSO write a STEP file ``<stem>.step`` (BREP, no
        tessellation; preserves named subparts). When combined with
        ``output_parts=True``, also writes per-part ``*.STEP`` files.
    stl_tolerance_mm:
        Linear chord tolerance (mm) for the STL exporter — controls flat
        surface accuracy.
    stl_angular_tolerance_rad:
        Angular chord tolerance (rad) applied to the merged STL and to
        per-part wheel STLs. Coarser = faster + smaller files.
    floor_angular_tolerance_rad:
        Angular chord tolerance (rad) applied to the FLOOR per-part STL
        only. Finer than ``stl_angular_tolerance_rad`` so the
        multisection-diffuser Bezier loft tessellates smoothly without
        making the wheel parts pay the cost.
    half_only:
        If True, build and export only the left (y<0) half of the car —
        2 wheels + half floor + half-arch geometry, sliced at y=0.

    Returns a record dict with paths + status.
    """
    if output_mode not in ("stl", "all"):
        raise ValueError(f"output_mode must be 'stl' or 'all', got {output_mode!r}")

    spec = spec or UnderbodySpec()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / f"{stem}.stl"
    step_path = output_dir / f"{stem}.step" if export_step else None

    record = {
        "status": "failed",
        "output_mode": output_mode,
        "output_parts": output_parts,
        "export_step": export_step,
        "stl_path": str(stl_path),
    }
    if step_path is not None:
        record["step_path"] = str(step_path)

    if output_mode == "stl":
        # Direct path; no log capture, no renderer.
        t0 = time.time()
        asy, layout = build_underbody(spec, half_only=half_only)
        _premesh_body(asy, stl_tolerance_mm, floor_angular_tolerance_rad)
        _export_stl(asy, stl_path, stl_tolerance_mm,
                    stl_angular_tolerance_rad)
        if output_parts:
            record["parts"] = _export_parts_stl(
                asy, output_dir, stl_tolerance_mm,
                stl_angular_tolerance_rad,
                floor_angular_tolerance_rad)
        if export_step:
            _export_step_assembly(asy, step_path)
            print(f"  step   : {step_path.name}  "
                  f"({step_path.stat().st_size / (1024 * 1024):.1f} MB)")
            if output_parts:
                record["step_parts"] = _export_parts_step(asy, output_dir)
        record.update({
            "status": "ok",
            "layout": layout,
            "total_s": round(time.time() - t0, 1),
            "stl_mb": round(stl_path.stat().st_size / (1024 * 1024), 1),
        })
        print(f"[ok] {stl_path}  ({record['stl_mb']} MB)  "
              f"in {record['total_s']}s")
        return record

    # output_mode == "all": tee stdout/stderr into a log file, generate
    # composite preview + parameter dump alongside the STL.
    log_path = output_dir / f"{stem}.log"
    json_path = output_dir / f"{stem}.json"
    preview_name = f"{stem}_preview.png"
    record.update({
        "log_path": str(log_path),
        "json_path": str(json_path),
        "preview_path": str(output_dir / preview_name),
    })

    with log_path.open("w") as log:
        tee = _Tee(sys.__stdout__, log)
        t0 = time.time()
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"=== {stem} ===")
                json_path.write_text(json.dumps(_spec_to_dict(spec), indent=2))
                print(f"params dumped -> {json_path.name}")

                t_build0 = time.time()
                asy, layout = build_underbody(spec, half_only=half_only)
                t_build = time.time() - t_build0
                print(f"build  : {t_build:.1f}s")

                t_export0 = time.time()
                _premesh_body(asy, stl_tolerance_mm,
                              floor_angular_tolerance_rad)
                _export_stl(asy, stl_path, stl_tolerance_mm,
                            stl_angular_tolerance_rad)
                t_export = time.time() - t_export0
                size_mb = stl_path.stat().st_size / (1024 * 1024)
                print(f"export : {t_export:.1f}s  ->  {stl_path.name}  ({size_mb:.1f} MB)")

                if output_parts:
                    record["parts"] = _export_parts_stl(
                        asy, output_dir, stl_tolerance_mm,
                        stl_angular_tolerance_rad,
                        floor_angular_tolerance_rad)

                if export_step:
                    t_step0 = time.time()
                    _export_step_assembly(asy, step_path)
                    t_step = time.time() - t_step0
                    step_mb = step_path.stat().st_size / (1024 * 1024)
                    print(f"step   : {t_step:.1f}s  ->  "
                          f"{step_path.name}  ({step_mb:.1f} MB)")
                    if output_parts:
                        record["step_parts"] = _export_parts_step(
                            asy, output_dir)

                meta = _meta_lines(spec, layout, stl_path)
                axles = _axle_payload(spec, layout)
                t_render0 = time.time()
                _spawn_renderer(stl_path, output_dir, meta,
                                preview_name, axles)
                t_render = time.time() - t_render0
                print(f"render : {t_render:.1f}s")

                record.update({
                    "status": "ok",
                    "layout": layout,
                    "build_s": round(t_build, 1),
                    "export_s": round(t_export, 1),
                    "render_s": round(t_render, 1),
                    "stl_mb": round(size_mb, 1),
                })
        except Exception as exc:
            log.write(f"\n=== EXCEPTION ===\n{type(exc).__name__}: {exc}\n")
            log.write(traceback.format_exc())
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["total_s"] = round(time.time() - t0, 1)
            log.write(f"\n=== status: {record['status']}  "
                      f"(total {record['total_s']}s) ===\n")

    return record


if __name__ == "__main__":
    # Quick smoke test: build the default underbody and emit just the STL.
    generate(output_mode="stl", output_dir="outputs", stem="underbody_default")
