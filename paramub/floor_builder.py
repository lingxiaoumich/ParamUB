"""Floor + rear diffuser builder.

Two output modes, picked by ``floor_thickness_mm``:

  - ``floor_thickness_mm == 0`` -> a **Face** (zero-thickness, the
    "surface" mode used for CFD meshing).
  - ``floor_thickness_mm  > 0`` -> a **Solid** slab of that thickness.

Both modes draw the same underside contour in XZ at Y = 0 and sweep / extrude
it across the floor width:

    flat floor (z = ride_height)  ->  R fillet  ->  diffuser ramp (angle)

Diffuser geometry:

  - starts kicking up at ``diffuser_start_x_mm`` (defaults to the rear
    axle, which the underbody assembler passes in via ``rear_axle_x``).
  - tangent fillet of radius ``diffuser_radius_mm`` connects the flat
    floor to the inclined diffuser ramp.
  - ramp angle is ``diffuser_angle_deg``.

The companion ``below_cropper`` Solid (see :func:`build_below_cropper`)
is a half-space below the floor contour, used elsewhere to clip
wheelhouse faces so they follow the diffuser slope instead of poking out.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from typing import Optional

import cadquery as cq


@dataclass
class FloorSpec:
    """Plan + diffuser + ride-height parameters for the underbody surface.

    Plan
        floor_x_min, floor_x_max:  longitudinal extent of the floor.
                                   (Computed by the underbody assembler
                                   from wheelbase + overhangs.)
        floor_width_mm:            full lateral width (±half each side).

    Ride
        ride_height_mm:            ground -> floor underside.
        floor_thickness_mm:        0 = surface mode (Face); >0 = solid slab.

    Diffuser
        diffuser_start_x_mm:       X where the floor stops being flat and
                                   the fillet starts. None means "use the
                                   rear axle X" — set by the assembler.
        diffuser_angle_deg:        ramp angle (positive = kicks upward at
                                   the rear).
        diffuser_radius_mm:        tangent fillet between flat and ramp.
    """
    floor_x_min: float
    floor_x_max: float
    floor_width_mm: float = 1800.0

    ride_height_mm: float = 100.0
    floor_thickness_mm: float = 0.0

    diffuser_start_x_mm: Optional[float] = None
    diffuser_angle_deg: float = 7.0
    diffuser_radius_mm: float = 500.0

    def validate(self) -> None:
        if self.floor_x_max <= self.floor_x_min:
            raise ValueError("floor_x_max must be > floor_x_min")
        if self.diffuser_start_x_mm is not None:
            if self.diffuser_start_x_mm <= self.floor_x_min:
                raise ValueError(
                    "diffuser_start_x_mm must be > floor_x_min (otherwise the "
                    "diffuser has zero length)")


def _diff_geometry(spec: FloorSpec):
    """Return the key contour points + segments along the underside profile."""
    spec.validate()
    diff_start_x = (spec.diffuser_start_x_mm
                    if spec.diffuser_start_x_mm is not None
                    else 0.0)  # fallback; assembler always passes a value.
    ride_h = spec.ride_height_mm
    angle_rad = radians(spec.diffuser_angle_deg)

    diff_length = diff_start_x - spec.floor_x_min
    diff_end_z = ride_h + diff_length * tan(angle_rad)

    d_consume = spec.diffuser_radius_mm * tan(angle_rad / 2.0)
    floor_tangent_x = diff_start_x + d_consume
    diff_tangent_x = diff_start_x - d_consume * cos(angle_rad)
    diff_tangent_z = ride_h + d_consume * sin(angle_rad)

    arc_mid_x = diff_start_x + spec.diffuser_radius_mm * (
        tan(angle_rad / 2.0) - sin(angle_rad / 2.0))
    arc_mid_z = ride_h + spec.diffuser_radius_mm * (1.0 - cos(angle_rad / 2.0))

    return dict(
        ride_h=ride_h,
        diff_end_z=diff_end_z,
        floor_tangent_x=floor_tangent_x,
        diff_tangent_x=diff_tangent_x,
        diff_tangent_z=diff_tangent_z,
        arc_mid_x=arc_mid_x,
        arc_mid_z=arc_mid_z,
    )


def _contour_wire(spec: FloorSpec):
    """Open wire (in XZ plane at Y = 0) tracing the underbody underside."""
    g = _diff_geometry(spec)
    e_floor = cq.Edge.makeLine(
        cq.Vector(spec.floor_x_max, 0, g["ride_h"]),
        cq.Vector(g["floor_tangent_x"], 0, g["ride_h"]),
    )
    e_fillet = cq.Edge.makeThreePointArc(
        cq.Vector(g["floor_tangent_x"], 0, g["ride_h"]),
        cq.Vector(g["arc_mid_x"], 0, g["arc_mid_z"]),
        cq.Vector(g["diff_tangent_x"], 0, g["diff_tangent_z"]),
    )
    e_diffuser = cq.Edge.makeLine(
        cq.Vector(g["diff_tangent_x"], 0, g["diff_tangent_z"]),
        cq.Vector(spec.floor_x_min, 0, g["diff_end_z"]),
    )
    return cq.Wire.assembleEdges([e_floor, e_fillet, e_diffuser])


def build_floor_surface(spec: FloorSpec):
    """Zero-thickness Face: floor + diffuser underside, full floor width."""
    contour = _contour_wire(spec)
    y_half = spec.floor_width_mm / 2.0
    contour_neg = contour.translate(cq.Vector(0, -y_half, 0))
    contour_pos = contour.translate(cq.Vector(0, +y_half, 0))
    return cq.Face.makeRuledSurface(contour_neg, contour_pos)


def build_floor_solid(spec: FloorSpec):
    """Slab of thickness ``floor_thickness_mm`` following the same contour."""
    g = _diff_geometry(spec)
    t = spec.floor_thickness_mm
    profile = (
        cq.Workplane("XZ")
        .moveTo(spec.floor_x_max, g["ride_h"])
        .lineTo(g["floor_tangent_x"], g["ride_h"])
        .threePointArc((g["arc_mid_x"], g["arc_mid_z"]),
                       (g["diff_tangent_x"], g["diff_tangent_z"]))
        .lineTo(spec.floor_x_min, g["diff_end_z"])
        .lineTo(spec.floor_x_min, g["diff_end_z"] + t)
        .lineTo(g["diff_tangent_x"], g["diff_tangent_z"] + t)
        .threePointArc((g["arc_mid_x"], g["arc_mid_z"] + t),
                       (g["floor_tangent_x"], g["ride_h"] + t))
        .lineTo(spec.floor_x_max, g["ride_h"] + t)
        .close()
    )
    return profile.extrude(spec.floor_width_mm / 2.0, both=True)


def build_floor(spec: FloorSpec):
    """Return Face (surface mode) or Solid (slab mode) per ``floor_thickness_mm``."""
    if spec.floor_thickness_mm <= 0:
        return build_floor_surface(spec)
    return build_floor_solid(spec)


def build_below_cropper(spec: FloorSpec):
    """Solid representing the half-space BELOW the underbody underside.

    Used to crop wheelhouse faces so the rear arches follow the diffuser
    slope. Extends past floor edges in X and ±1.5×floor_width in Y so every
    wheel is enclosed.
    """
    g = _diff_geometry(spec)
    z_below = -2000.0
    pad_x = 200.0
    cropper_x_min = spec.floor_x_min - pad_x
    cropper_x_max = spec.floor_x_max + pad_x

    profile = (
        cq.Workplane("XZ")
        .moveTo(cropper_x_min, z_below)
        .lineTo(cropper_x_min, g["diff_end_z"])
        .lineTo(spec.floor_x_min, g["diff_end_z"])
        .lineTo(g["diff_tangent_x"], g["diff_tangent_z"])
        .threePointArc((g["arc_mid_x"], g["arc_mid_z"]),
                       (g["floor_tangent_x"], g["ride_h"]))
        .lineTo(spec.floor_x_max, g["ride_h"])
        .lineTo(cropper_x_max, g["ride_h"])
        .lineTo(cropper_x_max, z_below)
        .close()
    )
    return profile.extrude(spec.floor_width_mm * 1.5, both=True)
