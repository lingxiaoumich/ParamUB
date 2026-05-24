"""Consolidated summary writer for the full reconstructed-car
pipeline.

After all four stages run (run_shell → integrate_underbody → fill_gap
→ make_watertight), the artefacts on disk look like::

    outputs/shell/<base>_meta.json                 # stage 1 metadata
    outputs/integrate/<base>_integrate_meta.json   # stage 2 spec + hints
    outputs/integrate/<base>_clean.json            # stage 4 body verify
    outputs/integrate/<base>_wheel_<corner>_clean.json   # stage 4 wheels

Each file is useful in isolation, but a human chasing the question
"what does this car look like geometrically?" wants a single flat
JSON with:

  * Geometry inputs used to generate the parametric UB
    - wheel center coordinates (xyz) per corner
    - tire spec (section width, aspect, rim diameter, ...)
    - wheelhouse heights + clearances
    - splitter / diffuser Bezier sections
    - floor z + ride height + width + angle
  * Geometric outputs of the final watertight result
    - body: dimensions, triangle count, watertight flags, volume, area
    - each wheel: dimensions, triangle count, watertight flag, volume
    - aggregate: total volume, total triangles

:func:`write_pipeline_summary` is the public entrypoint; it reads
the four metadata files and emits one ``<base>_summary.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


WHEEL_CORNERS = ("front_left", "front_right", "rear_left", "rear_right")


def _safe_load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _wheel_center_from_stl(stl_path: Path) -> list[float] | None:
    """bbox-center of a wheel STL = its 3-D wheel-hub centre. Returns
    None if the STL can't be loaded."""
    if not stl_path.exists():
        return None
    import trimesh
    m = trimesh.load(str(stl_path), force="mesh", process=True)
    centre = 0.5 * (m.bounds[0] + m.bounds[1])
    return [float(centre[0]), float(centre[1]), float(centre[2])]


def _extract_geometry(integrate_meta: dict, wheels_dir: Path,
                       base: str) -> dict[str, Any]:
    """Flatten the geometry knobs that ``integrate_underbody.py``
    feeds into the parametric UB."""
    hints = integrate_meta.get("hints", {})
    spec = integrate_meta.get("spec", {})
    wheel_spec = spec.get("wheel", {})
    tire = wheel_spec.get("tire", {})
    spoke = wheel_spec.get("spoke", {})

    geometry: dict[str, Any] = {
        # Vehicle plan
        "wheelbase_mm": spec.get("wheelbase_mm"),
        "front_overhang_mm": spec.get("front_overhang_mm"),
        "rear_overhang_mm": spec.get("rear_overhang_mm"),
        "track_front_mm": spec.get("track_front_mm"),
        "track_rear_mm": spec.get("track_rear_mm"),
        "scale_factor_shell_to_mm": hints.get("scale"),

        # Floor / ride
        "floor": {
            "ride_height_mm": spec.get("ride_height_mm"),
            "z_mm": spec.get("ride_height_mm"),
            "width_mm": spec.get("floor_width_mm"),
            "angle_deg": spec.get("floor_angle_deg"),
        },

        # Wheelhouses
        "wheelhouses": {
            "front_top_z_mm": (hints.get("wheelhouse_top_mm") or {}).get("front"),
            "rear_top_z_mm": (hints.get("wheelhouse_top_mm") or {}).get("rear"),
            "axial_clearance_mm": spec.get("wheel_house_axial_clearance_mm"),
            "lateral_clearance_mm": spec.get("wheel_house_lateral_clearance_mm"),
            "thickness_mm": spec.get("wheel_house_thickness_mm"),
            "front_steering_clearance_mm": spec.get("front_steering_clearance_mm"),
            "rear_steering_clearance_mm": spec.get("rear_steering_clearance_mm"),
            "front_fillet_mm": spec.get("front_wheel_house_fillet_mm"),
            "rear_fillet_mm": spec.get("rear_wheel_house_fillet_mm"),
            "lateral_clearance_overrides_mm": spec.get(
                "lateral_clearance_overrides_mm"),
        },

        # Alignment
        "alignment": {
            "camber_front_deg": spec.get("camber_front_deg"),
            "camber_rear_deg": spec.get("camber_rear_deg"),
            "toe_front_deg": spec.get("toe_front_deg"),
            "toe_rear_deg": spec.get("toe_rear_deg"),
        },

        # Tire + spoke
        "tire": tire,
        "spoke": spoke,

        # Splitter / diffuser Bezier sections (multisection mode)
        "splitter_sections": spec.get("splitter_sections") or [],
        "diffuser_sections": spec.get("diffuser_sections") or [],
        # Legacy single-section diffuser (None when multisection is used)
        "diffuser_legacy": {
            "start_x_mm": spec.get("diffuser_start_x_mm"),
            "angle_deg": spec.get("diffuser_angle_deg"),
            "radius_mm": spec.get("diffuser_radius_mm"),
        },
    }

    # Wheel centres: bbox-center of each raw stage-2 wheel STL.
    wheel_centers: dict[str, list[float] | None] = {}
    for corner in WHEEL_CORNERS:
        wheel_centers[corner] = _wheel_center_from_stl(
            wheels_dir / f"{base}_wheel_{corner}.stl")
    geometry["wheel_centers_mm"] = wheel_centers

    return geometry


def _body_output_summary(body_report: dict) -> dict[str, Any]:
    """Project a watertight verify report to the summary schema."""
    if not body_report:
        return {}
    return {
        "file": body_report.get("file"),
        "faces": body_report.get("faces"),
        "vertices": body_report.get("vertices"),
        "is_watertight": body_report.get("is_watertight"),
        "is_winding_consistent": body_report.get("is_winding_consistent"),
        "euler_number": body_report.get("euler_number"),
        "bbox_min_mm": body_report.get("bbox_min"),
        "bbox_max_mm": body_report.get("bbox_max"),
        "extents_mm": body_report.get("extents"),
        "surface_area_m2": body_report.get("surface_area_m2"),
        "volume_m3": body_report.get("volume_m3"),
        "dist_remesh_to_orig_mm": body_report.get("dist_remesh_to_orig_mm"),
        "dist_orig_to_remesh_mm": body_report.get("dist_orig_to_remesh_mm"),
    }


def _wheel_outputs_summary(integrate_dir: Path, base: str
                             ) -> tuple[dict[str, Any], float]:
    wheels: dict[str, Any] = {}
    total_vol = 0.0
    for corner in WHEEL_CORNERS:
        rep = _safe_load_json(
            integrate_dir / f"{base}_wheel_{corner}_clean.json")
        if not rep:
            wheels[corner] = None
            continue
        wheels[corner] = _body_output_summary(rep)
        total_vol += float(rep.get("volume_m3") or 0.0)
    return wheels, total_vol


def write_pipeline_summary(base: str,
                             shell_dir: str | Path = "outputs/shell",
                             integrate_dir: str | Path = "outputs/integrate",
                             out_path: str | Path | None = None,
                             ) -> Path:
    """Read all per-stage metadata for ``base`` and write one
    consolidated ``<base>_summary.json``.

    Parameters
    ----------
    base:
        Stem of the dataset (e.g. ``"alfa_..._shadowfill"``). The
        per-stage files are looked up by this name in ``shell_dir`` and
        ``integrate_dir``.
    out_path:
        Override the output path; defaults to
        ``<integrate_dir>/<base>_summary.json``.
    """
    shell_dir = Path(shell_dir)
    integrate_dir = Path(integrate_dir)
    if out_path is None:
        out_path = integrate_dir / f"{base}_summary.json"
    out_path = Path(out_path)

    shell_meta = _safe_load_json(shell_dir / f"{base}_meta.json")
    integrate_meta = _safe_load_json(
        integrate_dir / f"{base}_integrate_meta.json")
    body_report = _safe_load_json(integrate_dir / f"{base}_clean.json")

    geometry = _extract_geometry(integrate_meta, integrate_dir, base)
    body_out = _body_output_summary(body_report)
    wheels_out, wheels_vol = _wheel_outputs_summary(integrate_dir, base)

    total_faces = int(body_out.get("faces") or 0) + sum(
        int((w or {}).get("faces") or 0) for w in wheels_out.values())
    total_vol = float(body_out.get("volume_m3") or 0.0) + wheels_vol

    summary: dict[str, Any] = {
        "name": base,
        "schema_version": "1",
        "inputs": {
            "raw_mesh": shell_meta.get("input"),
            "shell_meta_json": str(shell_dir / f"{base}_meta.json"),
            "integrate_meta_json": str(
                integrate_dir / f"{base}_integrate_meta.json"),
        },
        "geometry": geometry,
        "outputs": {
            "body": body_out,
            "wheels": wheels_out,
            "totals": {
                "triangles": total_faces,
                "volume_m3": total_vol,
                "all_watertight": (
                    bool(body_out.get("is_watertight"))
                    and all(bool((w or {}).get("is_watertight"))
                            for w in wheels_out.values()
                            if w is not None)
                ),
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {out_path}  "
          f"watertight_body={summary['outputs']['body'].get('is_watertight')}  "
          f"wheels_ok={summary['outputs']['totals']['all_watertight']}",
          flush=True)
    return out_path
