"""Parametric car underbody — floor + rear diffuser + 4 wheel houses + 4 wheels.

Coordinate system (right-handed, mm):
  +X = vehicle forward
  +Y = vehicle right
  +Z = up (ground plane at Z = 0)

Origin is at ground level under the wheelbase midpoint. Front axle is at
+wheelbase/2, rear axle at -wheelbase/2; right wheels at +track/2.

The wheels come from tire_assembly.build_* and are oriented in the car
frame by:
  1. rotating the ParamUB local +Z (outboard) onto global +Y (right) or
     -Y (left) via a -90° / +90° rotation around the X axis,
  2. applying camber as a rotation around the longitudinal X axis,
  3. applying toe as a rotation around the vertical Z axis,
  4. translating to (axle_x, +/-track/2, tire_radius).

Sign conventions used throughout:
  camber_deg: negative camber = top of wheel tilted INWARD (toward
              centerline). Symmetric across the car.
  toe_deg:    positive toe = TOE-IN (front of wheel rotated inward).
              Symmetric across the car.

Per-axle wheel overrides are accepted as plain dicts that are merged with
the ParamUB defaults via tire_assembly.normalize_parameters().
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from math import cos, radians, sin, tan
from pathlib import Path
from typing import Optional

import cadquery as cq
from cadquery import exporters

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import tire_assembly as ta


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class UnderbodySpec:
    # Vehicle geometry
    wheelbase_mm: float = 2700.0
    front_overhang_mm: float = 850.0
    rear_overhang_mm: float = 800.0
    track_front_mm: float = 1550.0
    track_rear_mm: float = 1540.0

    # Ride / floor
    ride_height_mm: float = 100.0           # ground -> bottom of flat floor
    floor_width_mm: float = 1800.0          # rectangular plan width (overall)
    floor_thickness_mm: float = 0.0         # 0 = TRUE zero-thickness surface mode
                                            # (Compound of Faces, no inside/outside).
                                            # >0 = thin manifold slab of that thickness.
    floor_angle_deg: float = 0.0            # rake — small angle around lateral axis
                                            # (positive = nose-down / rear ride
                                            # height raised about the front edge)

    # Diffuser
    diffuser_start_x_mm: Optional[float] = None   # absolute X (mm) where the
                                                  # diffuser "kicks up"; None means
                                                  # "use the rear axle position"
    diffuser_angle_deg: float = 7.0
    diffuser_radius_mm: float = 500.0       # tangent fillet between floor & diffuser
                                            # (bumped from 200 → generous radius)

    # Wheel houses — half-cylindrical arches that cover each wheel from above.
    # In car-body terminology the arch is a "wheel house" that the tire fits
    # inside; the body shells around it like a tunnel along the spin axis.
    wheel_house_axial_clearance_mm: float = 30.0  # X/Z clearance around the tire
                                                  # (= cavity_radius - tire_radius)
    wheel_house_thickness_mm: float = 6.0         # arch shell thickness
                                                  # (solid mode only)
    wheel_house_lateral_clearance_mm: float = 35.0  # base Y clearance per side
    front_steering_clearance_mm: float = 150.0    # additional Y clearance per side
                                                  # on the FRONT arches so the
                                                  # front tires can swing through
                                                  # their steering range
    rear_steering_clearance_mm: float = 0.0       # rear wheels don't steer
    front_wheel_house_fillet_mm: float = 100.0    # tangent fillet on the inboard
                                                  # arc edge of the FRONT wheel
                                                  # houses (cap <-> cylinder).
                                                  # 0 disables the fillet.
    rear_wheel_house_fillet_mm: float = 30.0      # same for the REAR wheel houses.

    # Wheel alignment
    camber_front_deg: float = -1.5
    camber_rear_deg: float = -1.0
    toe_front_deg: float = 0.10
    toe_rear_deg: float = 0.0

    # STL mesh density. Used for BOTH the linear deflection (mm) and the
    # angular deflection (rad) on cq.exporters.export. Smaller -> finer.
    # 0.1 gives ~400k faces / ~19 MB on the default underbody — a good
    # balance for CFD surface meshes without dragging the export out.
    stl_tolerance: float = 0.1

    # Per-axle wheel parameter overrides (passed through to
    # tire_assembly.normalize_parameters). Empty -> use ParamUB defaults.
    front_wheel_overrides: dict = field(default_factory=dict)
    rear_wheel_overrides: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def compute_layout(spec: UnderbodySpec) -> dict:
    rear_axle_x = -spec.wheelbase_mm / 2.0
    front_axle_x = +spec.wheelbase_mm / 2.0
    floor_x_max = front_axle_x + spec.front_overhang_mm        # nose of the floor
    floor_x_min = rear_axle_x - spec.rear_overhang_mm          # tail of the floor
    if spec.diffuser_start_x_mm is None:
        diff_start_x = rear_axle_x
    else:
        diff_start_x = float(spec.diffuser_start_x_mm)
    overall_length_mm = floor_x_max - floor_x_min
    return dict(
        rear_axle_x=rear_axle_x,
        front_axle_x=front_axle_x,
        floor_x_max=floor_x_max,
        floor_x_min=floor_x_min,
        diff_start_x=diff_start_x,
        overall_length_mm=overall_length_mm,
    )


# ---------------------------------------------------------------------------
# Wheel: build then place
# ---------------------------------------------------------------------------

def _build_one_wheel(overrides: dict):
    """Build a single wheel solid using tire_assembly's components.

    Returns (workplane_wheel, derived_dict, tire_axial_width_mm).
    """
    with ta.temporary_parameters(overrides):
        d = ta.compute_derived()
        tire = ta.build_tire(d)
        disc = ta.build_wheel_disc(d)
        disc_cut = ta.cut_spoke_windows(disc, d)
        barrel = ta.build_rim_barrel(d)
        assembly = ta.build_assembly(tire, disc_cut, barrel)
        tire_axial_width = ta.section_width_mm
    return assembly, d, tire_axial_width


def _place_wheel(wheel_wp, *, side: str, axle_x: float, track: float,
                 camber_deg: float, toe_deg: float, tire_radius: float):
    """Rotate/translate one wheel from ParamUB local coords into the car frame."""
    sign = +1.0 if side == "right" else -1.0
    spin_axis_rotation = -90.0 if side == "right" else +90.0
    track_y = sign * track / 2.0

    # Order: spin-axis flip → camber → toe → translate.
    # Camber sign:  negative camber should tilt the wheel top INWARD on BOTH
    # sides. After the spin-axis flip, rotation around +X by +θ tilts +Z
    # toward -Y. For the right wheel (-Y = inward), this matches a positive
    # rotation when we want negative camber, so we apply -sign * camber.
    # Toe sign:    toe-in moves the wheel front inward (-Y for right, +Y for
    # left). Rotation around +Z by +θ moves +X toward +Y; we want +X toward
    # -Y for the right wheel toe-in, so we apply -sign * toe.
    return (
        wheel_wp
        .rotate((0, 0, 0), (1, 0, 0), spin_axis_rotation)
        .rotate((0, 0, 0), (1, 0, 0), -sign * camber_deg)
        .rotate((0, 0, 0), (0, 0, 1), -sign * toe_deg)
        .translate((axle_x, track_y, tire_radius))
    )


# ---------------------------------------------------------------------------
# Underbody solid
# ---------------------------------------------------------------------------

def build_floor_diffuser_thin(spec: UnderbodySpec, layout: dict):
    """Thin (floor_thickness_mm) floor + rear diffuser, one solid.

    Both the underside and the topside of the slab follow the floor +
    diffuser contour, separated by floor_thickness_mm. The diffuser end
    has a true tangent R200 fillet.

    Profile lives in workplane "XZ" (local x = global X, local y = global Z)
    and is extruded symmetrically along Y to span ±floor_width/2.
    """
    angle_rad = radians(spec.diffuser_angle_deg)
    diff_length = layout["diff_start_x"] - layout["floor_x_min"]
    if diff_length <= 0:
        raise ValueError(
            "diffuser_start_x_mm puts the diffuser start at or behind the "
            "rear of the floor; pick a larger rear_overhang_mm or a more "
            "positive diffuser_start_x_mm (further forward)."
        )

    diff_end_z = spec.ride_height_mm + diff_length * tan(angle_rad)

    d_consume = spec.diffuser_radius_mm * tan(angle_rad / 2.0)
    floor_tangent_x = layout["diff_start_x"] + d_consume
    diff_tangent_x = layout["diff_start_x"] - d_consume * cos(angle_rad)
    diff_tangent_z = spec.ride_height_mm + d_consume * sin(angle_rad)

    arc_mid_x = layout["diff_start_x"] + spec.diffuser_radius_mm * (
        tan(angle_rad / 2.0) - sin(angle_rad / 2.0)
    )
    arc_mid_z = spec.ride_height_mm + spec.diffuser_radius_mm * (
        1.0 - cos(angle_rad / 2.0)
    )

    t = spec.floor_thickness_mm  # shorthand

    # CCW closed wire: underside contour going rear -> nose, then topside
    # contour going nose -> rear, offset upward by floor thickness.
    profile = (
        cq.Workplane("XZ")
        # Underside, nose to fillet to diffuser to tail
        .moveTo(layout["floor_x_max"], spec.ride_height_mm)
        .lineTo(floor_tangent_x, spec.ride_height_mm)
        .threePointArc((arc_mid_x, arc_mid_z), (diff_tangent_x, diff_tangent_z))
        .lineTo(layout["floor_x_min"], diff_end_z)
        # Rear face (vertical jog by thickness)
        .lineTo(layout["floor_x_min"], diff_end_z + t)
        # Topside back to nose, offset upward by t
        .lineTo(diff_tangent_x, diff_tangent_z + t)
        .threePointArc((arc_mid_x, arc_mid_z + t),
                       (floor_tangent_x, spec.ride_height_mm + t))
        .lineTo(layout["floor_x_max"], spec.ride_height_mm + t)
        # Front face (vertical jog by thickness)
        .close()
    )

    return profile.extrude(spec.floor_width_mm / 2.0, both=True)


def _wheel_house_cutout_x_at(z_eval: float, tire_radius: float,
                              spec: UnderbodySpec) -> float:
    """X half-extent of the wheel-house INNER cavity at a given z.

    Inner cavity = circle of radius (tire_radius + axial_clearance) centered
    at z = tire_radius. The cutout in the floor must be at least this wide
    in X at the floor's plane, otherwise the floor blocks the wheel.
    """
    inner_r = tire_radius + spec.wheel_house_axial_clearance_mm
    dz = abs(z_eval - tire_radius)
    if dz >= inner_r:
        return 0.0
    return (inner_r ** 2 - dz ** 2) ** 0.5


def cut_wheel_house_openings(body, spec: UnderbodySpec, layout: dict,
                              tire_radius_f: float, tire_width_f: float,
                              tire_radius_r: float, tire_width_r: float):
    """Cut through-openings in the thin floor where each wheel passes through.

    The opening at each wheel is sized to clear the inner arch cavity at the
    floor's plane plus the lateral clearance (and steering clearance for
    the front).
    """
    targets = [
        (layout["front_axle_x"], +spec.track_front_mm / 2.0, tire_radius_f, tire_width_f,
         spec.wheel_house_lateral_clearance_mm + spec.front_steering_clearance_mm),
        (layout["front_axle_x"], -spec.track_front_mm / 2.0, tire_radius_f, tire_width_f,
         spec.wheel_house_lateral_clearance_mm + spec.front_steering_clearance_mm),
        (layout["rear_axle_x"],  +spec.track_rear_mm  / 2.0, tire_radius_r, tire_width_r,
         spec.wheel_house_lateral_clearance_mm + spec.rear_steering_clearance_mm),
        (layout["rear_axle_x"],  -spec.track_rear_mm  / 2.0, tire_radius_r, tire_width_r,
         spec.wheel_house_lateral_clearance_mm + spec.rear_steering_clearance_mm),
    ]

    for axle_x, y_track, tire_r, tire_w, lat_clear in targets:
        # Cut for the arch's FULL outer-cylinder footprint, not just the
        # inner cavity at the floor plane. The earlier inner-cavity sizing
        # left a strip of floor inside the arch's outer envelope, visible
        # on the rear half of the rear arch where the diffuser slope
        # pushes the floor into the wheel-house volume.
        outer_r_full = (
            tire_r
            + spec.wheel_house_axial_clearance_mm
            + max(spec.wheel_house_thickness_mm, 0.0)
        )
        cutout_length_x = 2.0 * outer_r_full + 4.0
        cutout_width_y = tire_w + 2.0 * lat_clear
        # Cut through the slab; make it deliberately tall in Z so it punches
        # both surfaces cleanly even if the diffuser slope is in play.
        cutout = (
            cq.Workplane("XY")
            .box(cutout_length_x, cutout_width_y,
                 spec.floor_thickness_mm * 6.0 + 200.0)
            .translate((axle_x, y_track,
                        spec.ride_height_mm + spec.floor_thickness_mm / 2.0))
        )
        body = body.cut(cutout)
    return body


def build_below_underbody_cropper(spec: UnderbodySpec, layout: dict):
    """Solid representing the volume BELOW the underbody's actual underside.

    Use this (rather than a flat z = ride_height plane) when cropping the
    wheel-arch shells, so the arches follow the diffuser slope at the rear
    instead of leaving material hanging below the diffuser surface.

    The cropper extends slightly past floor_x_min / floor_x_max in X (so
    arches near the floor edges are still covered) and ±1.5 × floor_width
    in Y (so every wheel is enclosed).
    """
    ride_h = spec.ride_height_mm
    angle_rad = radians(spec.diffuser_angle_deg)
    diff_length = layout["diff_start_x"] - layout["floor_x_min"]
    diff_end_z = ride_h + max(diff_length, 0.0) * tan(angle_rad)

    d_consume = spec.diffuser_radius_mm * tan(angle_rad / 2.0)
    floor_tangent_x = layout["diff_start_x"] + d_consume
    diff_tangent_x = layout["diff_start_x"] - d_consume * cos(angle_rad)
    diff_tangent_z = ride_h + d_consume * sin(angle_rad)
    arc_mid_x = layout["diff_start_x"] + spec.diffuser_radius_mm * (
        tan(angle_rad / 2.0) - sin(angle_rad / 2.0)
    )
    arc_mid_z = ride_h + spec.diffuser_radius_mm * (
        1.0 - cos(angle_rad / 2.0)
    )

    z_below = -2000.0          # well below ground
    pad_x = 200.0              # X extension past the floor edges

    cropper_x_min = layout["floor_x_min"] - pad_x
    cropper_x_max = layout["floor_x_max"] + pad_x

    # CCW closed profile in workplane "XZ" (local x = global X, local y = global Z).
    # Top of the profile traces the underbody's underside, bottom is at z_below.
    profile = (
        cq.Workplane("XZ")
        .moveTo(cropper_x_min, z_below)
        .lineTo(cropper_x_min, diff_end_z)                                # rear, extended
        .lineTo(layout["floor_x_min"], diff_end_z)                        # over to actual rear edge
        .lineTo(diff_tangent_x, diff_tangent_z)                           # down the diffuser
        .threePointArc((arc_mid_x, arc_mid_z),
                       (floor_tangent_x, ride_h))                        # R fillet
        .lineTo(layout["floor_x_max"], ride_h)                            # along flat floor
        .lineTo(cropper_x_max, ride_h)                                    # forward, extended
        .lineTo(cropper_x_max, z_below)
        .close()
    )

    return profile.extrude(spec.floor_width_mm * 1.5, both=True)


def build_wheel_arch_shell(axle_x: float, y_track: float,
                            tire_radius: float, tire_width: float,
                            lateral_clear: float, spec: UnderbodySpec,
                            side: str, below_cropper=None):
    """Half-cylindrical wheel-house shell with inboard cap, outboard trim.

    The arch is a half-cylinder (axis along Y, only the portion above
    z = ride_height) with:
      - INBOARD end CLOSED by a flat half-disk cap of thickness
        wheel_house_thickness_mm, extending one shell-thickness further
        toward the centerline so the cap is *outside* the cavity.
      - OUTBOARD end TRIMMED at Y = +/- floor_width/2 so the arch never
        sticks past the rectangular floor's lateral perimeter. When the
        arch already fits inside that perimeter the trim is a no-op.

    side: "left" or "right" — controls which Y direction counts as inboard.
    """
    sign = +1.0 if side == "right" else -1.0
    inner_r = tire_radius + spec.wheel_house_axial_clearance_mm
    outer_r = inner_r + spec.wheel_house_thickness_mm
    arch_length = tire_width + 2.0 * lateral_clear

    y_inboard = y_track - sign * arch_length / 2.0
    y_outboard = y_track + sign * arch_length / 2.0

    # --- Cylindrical shell over the wheel ----------------------------------
    outer_cyl = (
        cq.Workplane("XZ")
        .circle(outer_r)
        .extrude(arch_length / 2.0, both=True)
        .translate((axle_x, y_track, tire_radius))
    )
    inner_cyl = (
        cq.Workplane("XZ")
        .circle(inner_r)
        .extrude(arch_length / 2.0 + 5.0, both=True)
        .translate((axle_x, y_track, tire_radius))
    )
    shell = outer_cyl.cut(inner_cyl)

    if below_cropper is not None:
        # Crop against the actual underbody surface so the arch's underside
        # follows the diffuser slope where applicable.
        arch = shell.cut(below_cropper)
    else:
        floor_cropper = (
            cq.Workplane("XY")
            .box(outer_r * 4.0, arch_length + 200.0, 2000.0)
            .translate((axle_x, y_track, spec.ride_height_mm - 1000.0))
        )
        arch = shell.cut(floor_cropper)

    # --- Inboard cap (half-disk just inboard of the cylinder's end) --------
    cap_thickness = spec.wheel_house_thickness_mm
    cap_center_y = y_inboard - sign * cap_thickness / 2.0
    cap = (
        cq.Workplane("XZ")
        .circle(outer_r)
        .extrude(cap_thickness / 2.0, both=True)
        .translate((axle_x, cap_center_y, tire_radius))
    )
    if below_cropper is not None:
        cap = cap.cut(below_cropper)
    else:
        cap_cropper = (
            cq.Workplane("XY")
            .box(outer_r * 4.0, cap_thickness * 4.0, 2000.0)
            .translate((axle_x, cap_center_y, spec.ride_height_mm - 1000.0))
        )
        cap = cap.cut(cap_cropper)
    arch = arch.union(cap)

    # --- Trim outboard end at the floor's lateral perimeter ----------------
    floor_edge_signed = sign * spec.floor_width_mm / 2.0
    if (sign > 0 and y_outboard > floor_edge_signed) or \
       (sign < 0 and y_outboard < floor_edge_signed):
        huge = 10000.0
        if sign > 0:
            trim_box = (
                cq.Workplane("XY").box(huge, huge, huge)
                .translate((0, floor_edge_signed + huge / 2.0, 0))
            )
        else:
            trim_box = (
                cq.Workplane("XY").box(huge, huge, huge)
                .translate((0, floor_edge_signed - huge / 2.0, 0))
            )
        arch = arch.cut(trim_box)

    return arch


def _wheel_targets(spec: UnderbodySpec, layout: dict,
                    tire_radius_f, tire_width_f,
                    tire_radius_r, tire_width_r):
    front_lat = spec.wheel_house_lateral_clearance_mm + spec.front_steering_clearance_mm
    rear_lat  = spec.wheel_house_lateral_clearance_mm + spec.rear_steering_clearance_mm
    return [
        # (axle_x, y_track, tire_radius, tire_width, lateral_clear, side)
        (layout["front_axle_x"], +spec.track_front_mm / 2.0,
         tire_radius_f, tire_width_f, front_lat, "right"),
        (layout["front_axle_x"], -spec.track_front_mm / 2.0,
         tire_radius_f, tire_width_f, front_lat, "left"),
        (layout["rear_axle_x"],  +spec.track_rear_mm  / 2.0,
         tire_radius_r, tire_width_r, rear_lat,  "right"),
        (layout["rear_axle_x"],  -spec.track_rear_mm  / 2.0,
         tire_radius_r, tire_width_r, rear_lat,  "left"),
    ]


def build_body_solid(spec: UnderbodySpec, layout: dict,
                      tire_radius_f, tire_width_f,
                      tire_radius_r, tire_width_r):
    """Thin-shell mode: floor+diffuser as a solid slab + 4 half-cylinder arch shells."""
    body = build_floor_diffuser_thin(spec, layout)
    body = cut_wheel_house_openings(body, spec, layout,
                                    tire_radius_f, tire_width_f,
                                    tire_radius_r, tire_width_r)

    below_cropper = build_below_underbody_cropper(spec, layout)

    for axle_x, y_track, tire_r, tire_w, lat_clear, side in _wheel_targets(
            spec, layout,
            tire_radius_f, tire_width_f, tire_radius_r, tire_width_r):
        arch = build_wheel_arch_shell(axle_x, y_track, tire_r, tire_w,
                                      lat_clear, spec, side,
                                      below_cropper=below_cropper)
        body = body.union(arch)
    return body


def _cut_face_with_solid(face_shape, solid_shape):
    """Boolean-cut a Face (or Compound of Faces) with a Solid via OCP."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

    cut = BRepAlgoAPI_Cut(face_shape.wrapped, solid_shape.wrapped)
    cut.Build()
    return cq.Shape.cast(cut.Shape())


def _contour_wire(spec: UnderbodySpec, layout: dict):
    """Open wire (in XZ plane, at Y = 0) tracing the underbody underside."""
    ride_h = spec.ride_height_mm
    angle_rad = radians(spec.diffuser_angle_deg)
    diff_length = layout["diff_start_x"] - layout["floor_x_min"]
    if diff_length <= 0:
        raise ValueError(
            "diffuser_start_x_mm puts the diffuser start at or behind the rear "
            "of the floor; pick a larger rear_overhang_mm or a more positive "
            "diffuser_start_x_mm."
        )
    diff_end_z = ride_h + diff_length * tan(angle_rad)

    d_consume = spec.diffuser_radius_mm * tan(angle_rad / 2.0)
    floor_tangent_x = layout["diff_start_x"] + d_consume
    diff_tangent_x = layout["diff_start_x"] - d_consume * cos(angle_rad)
    diff_tangent_z = ride_h + d_consume * sin(angle_rad)
    arc_mid_x = layout["diff_start_x"] + spec.diffuser_radius_mm * (
        tan(angle_rad / 2.0) - sin(angle_rad / 2.0)
    )
    arc_mid_z = ride_h + spec.diffuser_radius_mm * (
        1.0 - cos(angle_rad / 2.0)
    )

    # Build edges directly (Workplane.edges() only returns the latest edge
    # in the chain, not all of them).
    e_floor = cq.Edge.makeLine(
        cq.Vector(layout["floor_x_max"], 0, ride_h),
        cq.Vector(floor_tangent_x, 0, ride_h),
    )
    e_fillet = cq.Edge.makeThreePointArc(
        cq.Vector(floor_tangent_x, 0, ride_h),
        cq.Vector(arc_mid_x, 0, arc_mid_z),
        cq.Vector(diff_tangent_x, 0, diff_tangent_z),
    )
    e_diffuser = cq.Edge.makeLine(
        cq.Vector(diff_tangent_x, 0, diff_tangent_z),
        cq.Vector(layout["floor_x_min"], 0, diff_end_z),
    )
    return cq.Wire.assembleEdges([e_floor, e_fillet, e_diffuser])


def build_body_surface(spec: UnderbodySpec, layout: dict,
                        tire_radius_f, tire_width_f,
                        tire_radius_r, tire_width_r):
    """True zero-thickness mode: returns a Compound of Faces.

    Components:
      - One ruled-surface floor+diffuser face spanning ±floor_width/2 in Y,
        with rectangular wheel-house openings cut out.
      - 4 half-cylinder wheel-house faces (axis along Y, radius
        tire_r + axial_clearance), trimmed outboard at floor edge and
        cropped at the underbody underside profile.
      - 4 inboard cap half-disk faces closing each wheel house on the
        centerline-facing end.

    The Compound exports cleanly to STL as a single-sided triangle mesh
    (useful for CFD/aero meshers that extrude wall thickness later).
    """
    contour = _contour_wire(spec, layout)
    y_half = spec.floor_width_mm / 2.0
    contour_neg = contour.translate(cq.Vector(0, -y_half, 0))
    contour_pos = contour.translate(cq.Vector(0, +y_half, 0))
    floor_face = cq.Face.makeRuledSurface(contour_neg, contour_pos)

    # Build each wheelhouse as a FILLETED solid first (per-axle fillet radius).
    # Then use those solids both to cut the floor (so the floor's cutout outline
    # picks up the fillet) and to extract the outer surfaces. This guarantees
    # the floor edge and the wheelhouse surfaces share the same boundary curve
    # — no holes or gaps around the fillet.
    wheelhouse_solids = []
    for axle_x, y_track, tire_r, tire_w, lat_clear, side in _wheel_targets(
            spec, layout,
            tire_radius_f, tire_width_f, tire_radius_r, tire_width_r):
        is_front = axle_x > 0
        fillet_r = (spec.front_wheel_house_fillet_mm if is_front
                    else spec.rear_wheel_house_fillet_mm)
        solid = _build_wheelhouse_solid(
            axle_x, y_track, tire_r, tire_w, lat_clear,
            spec, side, fillet_r,
        )
        if solid is not None:
            wheelhouse_solids.append((axle_x, y_track, side, solid))

    # Cut the floor with each wheelhouse solid. The solid's bottom (a flat
    # chord at z = ride_height) IS the wheelhouse footprint after filleting,
    # so the floor's cutout edge ends up exactly on the wheelhouse boundary.
    for _ax, _y, _side, solid in wheelhouse_solids:
        floor_face = _cut_face_with_solid(floor_face, solid.val())

    below_cropper = build_below_underbody_cropper(spec, layout).val()

    wheelhouse_faces = []
    for _ax, _y, side, solid in wheelhouse_solids:
        wheelhouse_faces.extend(
            _extract_wheelhouse_surfaces(solid, spec, side, below_cropper)
        )

    return cq.Compound.makeCompound([floor_face, *wheelhouse_faces])


def _build_wheelhouse_solid(axle_x: float, y_track: float,
                             tire_r: float, tire_w: float,
                             lateral_clear: float,
                             spec: UnderbodySpec, side: str,
                             fillet_r: float):
    """Closed half-cylinder solid for one wheelhouse, with optional 3D fillet
    on the inboard arc edge and outboard trim at the floor's lateral edge.

    Bottom is FLAT at z = ride_height (the chord). The diffuser-following
    crop happens later, on the extracted faces, not on the solid (so the
    solid can be used as a clean cutter for the floor).

    Returns a cq.Workplane wrapping the resulting Solid, or None if the
    geometry collapses.
    """
    sign = +1.0 if side == "right" else -1.0
    arch_r = tire_r + spec.wheel_house_axial_clearance_mm
    arch_length = tire_w + 2.0 * lateral_clear
    y_outboard = y_track + sign * arch_length / 2.0

    # 1) Closed "arch" solid: vertical sidewalls from the floor up to the
    #    cylinder's widest point (z = tire_radius), then a semicircular dome
    #    over the top. The sidewalls are tangent to the dome at z = tire_r,
    #    so the cross-section reads as a single smooth curve there.
    #
    #    Old shape was the full circular wedge above the floor (from the
    #    floor on one side, over the apex, back to the floor on the other);
    #    this one extrudes the widest point down with straight walls instead.
    cross_section = (
        cq.Workplane("XZ")
        .moveTo(axle_x - arch_r, spec.ride_height_mm)
        .lineTo(axle_x - arch_r, tire_r)                              # L sidewall
        .threePointArc((axle_x, tire_r + arch_r),
                       (axle_x + arch_r, tire_r))                     # dome
        .lineTo(axle_x + arch_r, spec.ride_height_mm)                 # R sidewall
        .close()                                                      # floor chord
    )
    solid_wp = (cross_section
                .extrude(arch_length / 2.0, both=True)
                .translate((0, y_track, 0)))

    # 2) Inboard fillet — propagate around the WHOLE upper outline of the
    #    inboard cap (dome arc + both vertical sidewalls). The fillet runs
    #    continuously from the floor on one side, up the sidewall, around
    #    the dome, and back down to the floor on the other side. The
    #    sidewall portion of the fillet terminates at z = ride_height,
    #    where the floor cut later trims it to a clean edge.
    #    The chord at the floor (z = ride_height) is excluded — that's
    #    where the wheelhouse meets the floor and gets trimmed by the
    #    floor-cut step downstream.
    fillet_r_clamped = max(min(fillet_r, arch_r * 0.9), 0.0)
    if fillet_r_clamped > 0:
        inboard_sel = "<Y" if side == "right" else ">Y"
        try:
            inboard_face = solid_wp.faces(inboard_sel).val()
            upper_edges = [
                e for e in inboard_face.Edges()
                if e.Center().z > spec.ride_height_mm + 1.0
            ]
            if upper_edges:
                solid_wp = solid_wp.newObject(upper_edges).fillet(fillet_r_clamped)
        except Exception as exc:
            print(f"[warn] inboard fillet failed on {side} wheel arch: {exc}")

    # 3) Outboard trim at the floor's lateral perimeter.
    floor_edge_signed = sign * spec.floor_width_mm / 2.0
    if (sign > 0 and y_outboard > floor_edge_signed) or \
       (sign < 0 and y_outboard < floor_edge_signed):
        huge = 10000.0
        if sign > 0:
            trim_box = (cq.Workplane("XY").box(huge, huge, huge)
                        .translate((0, floor_edge_signed + huge / 2.0, 0)))
        else:
            trim_box = (cq.Workplane("XY").box(huge, huge, huge)
                        .translate((0, floor_edge_signed - huge / 2.0, 0)))
        solid_wp = solid_wp.cut(trim_box)

    return solid_wp


def _extract_wheelhouse_surfaces(solid_wp, spec: UnderbodySpec, side: str,
                                  below_cropper):
    """Extract the outer surfaces from a wheelhouse solid: keep the cylinder
    face, the inboard cap, and the fillet face. Drop the bottom chord and
    the outboard end cap. Crop each retained face against the underbody's
    underside profile so rear arches follow the diffuser slope.
    """
    sign = +1.0 if side == "right" else -1.0
    bbox = solid_wp.val().BoundingBox()

    keep = []
    for f in solid_wp.faces().vals():
        fb = f.BoundingBox()
        # Bottom chord (flat at z = ride_height).
        if abs(fb.zmax - spec.ride_height_mm) < 0.5 and (fb.zmax - fb.zmin) < 0.5:
            continue
        # Outboard cap (flat at the most-outboard Y).
        if sign > 0:
            if abs(fb.ymin - bbox.ymax) < 0.5 and (fb.ymax - fb.ymin) < 0.5:
                continue
        else:
            if abs(fb.ymax - bbox.ymin) < 0.5 and (fb.ymax - fb.ymin) < 0.5:
                continue
        keep.append(f)

    cropped = []
    for f in keep:
        cropped_shape = _cut_face_with_solid(f, below_cropper)
        if cropped_shape is not None and not cropped_shape.isNull():
            cropped.append(cropped_shape)
    return cropped


def build_body(spec: UnderbodySpec, layout: dict,
                tire_radius_f, tire_width_f,
                tire_radius_r, tire_width_r):
    """Dispatch: surface mode (floor_thickness_mm == 0) vs thin-solid mode."""
    if spec.floor_thickness_mm <= 0:
        return build_body_surface(spec, layout,
                                  tire_radius_f, tire_width_f,
                                  tire_radius_r, tire_width_r)
    return build_body_solid(spec, layout,
                            tire_radius_f, tire_width_f,
                            tire_radius_r, tire_width_r)


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

def build_underbody(spec: UnderbodySpec | None = None):
    spec = spec or UnderbodySpec()
    layout = compute_layout(spec)
    print(f"[layout] {layout}")

    print("[wheels] building front and rear wheels via ParamUB ...")
    front_wheel, front_d, tire_width_f = _build_one_wheel(spec.front_wheel_overrides)
    if spec.rear_wheel_overrides == spec.front_wheel_overrides:
        rear_wheel, rear_d, tire_width_r = front_wheel, front_d, tire_width_f
    else:
        rear_wheel, rear_d, tire_width_r = _build_one_wheel(spec.rear_wheel_overrides)

    tire_radius_f = front_d["tire_od_mm"] / 2.0
    tire_radius_r = rear_d["tire_od_mm"] / 2.0
    print(f"  front: OD={2*tire_radius_f:.1f} mm width={tire_width_f}")
    print(f"  rear : OD={2*tire_radius_r:.1f} mm width={tire_width_r}")

    print("[body] building thin floor + diffuser + wheel-house arches ...")
    body = build_body(
        spec, layout,
        tire_radius_f=tire_radius_f, tire_width_f=tire_width_f,
        tire_radius_r=tire_radius_r, tire_width_r=tire_width_r,
    )

    if abs(spec.floor_angle_deg) > 1e-6:
        # Rake about the front-bottom edge so the front of the flat floor stays
        # at ride_height; positive rake raises the tail.
        pivot_x = layout["floor_x_max"]
        body = body.rotate(
            (pivot_x, 0, spec.ride_height_mm),
            (pivot_x, 1, spec.ride_height_mm),
            -spec.floor_angle_deg,
        )

    print("[wheels] placing 4 corners ...")
    wheels = [
        ("wheel_fl",
         _place_wheel(front_wheel, side="left",
                      axle_x=layout["front_axle_x"], track=spec.track_front_mm,
                      camber_deg=spec.camber_front_deg, toe_deg=spec.toe_front_deg,
                      tire_radius=tire_radius_f)),
        ("wheel_fr",
         _place_wheel(front_wheel, side="right",
                      axle_x=layout["front_axle_x"], track=spec.track_front_mm,
                      camber_deg=spec.camber_front_deg, toe_deg=spec.toe_front_deg,
                      tire_radius=tire_radius_f)),
        ("wheel_rl",
         _place_wheel(rear_wheel, side="left",
                      axle_x=layout["rear_axle_x"], track=spec.track_rear_mm,
                      camber_deg=spec.camber_rear_deg, toe_deg=spec.toe_rear_deg,
                      tire_radius=tire_radius_r)),
        ("wheel_rr",
         _place_wheel(rear_wheel, side="right",
                      axle_x=layout["rear_axle_x"], track=spec.track_rear_mm,
                      camber_deg=spec.camber_rear_deg, toe_deg=spec.toe_rear_deg,
                      tire_radius=tire_radius_r)),
    ]

    print("[assemble] combining body + 4 wheels ...")
    asy = cq.Assembly()
    asy.add(body, name="body", color=cq.Color(0.78, 0.80, 0.84))
    for name, w in wheels:
        asy.add(w, name=name, color=cq.Color(0.18, 0.18, 0.20))

    return asy, layout


# ---------------------------------------------------------------------------
# Pipeline (export + validate + render)
# ---------------------------------------------------------------------------

def run_pipeline(
    spec: UnderbodySpec | None = None,
    output_dir: str | os.PathLike = None,
    stem: str = "underbody",
    render: bool = True,
):
    spec = spec or UnderbodySpec()
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "underbody"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asy, layout = build_underbody(spec)

    stl_path = output_dir / f"{stem}.stl"
    print(f"[export] writing {stl_path} (stl_tolerance = {spec.stl_tolerance})")
    compound = asy.toCompound()
    exporters.export(
        compound,
        str(stl_path),
        exporters.ExportTypes.STL,
        tolerance=spec.stl_tolerance,
        angularTolerance=spec.stl_tolerance,
    )

    print("[validate] trimesh ...")
    import trimesh
    mesh = trimesh.load(str(stl_path))
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume cm^3: {mesh.volume / 1000.0:.1f}")
    print(f"  Face count : {len(mesh.faces)}")
    print(f"  Bounds mm  : {mesh.bounds}")
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        mesh.export(str(stl_path))
        print(f"  Watertight (post-repair): {mesh.is_watertight}")

    if render:
        meta_lines = [
            f"wheelbase_mm        = {spec.wheelbase_mm}",
            f"track F/R (mm)      = {spec.track_front_mm} / {spec.track_rear_mm}",
            f"overhang F/R (mm)   = {spec.front_overhang_mm} / {spec.rear_overhang_mm}",
            f"overall length (mm) = {layout['overall_length_mm']:.0f}",
            f"floor_width_mm      = {spec.floor_width_mm}",
            f"ride_height_mm      = {spec.ride_height_mm}",
            f"floor_angle_deg     = {spec.floor_angle_deg}",
            f"diffuser angle/R    = {spec.diffuser_angle_deg}° / R{spec.diffuser_radius_mm}",
            f"camber F/R (deg)    = {spec.camber_front_deg} / {spec.camber_rear_deg}",
            f"toe F/R (deg)       = {spec.toe_front_deg} / {spec.toe_rear_deg}",
            "",
            f"Watertight: {bool(mesh.is_watertight)}",
            f"Volume cm^3: {mesh.volume / 1000.0:.1f}",
        ]
        _spawn_renderer(stl_path, output_dir, meta_lines)

    return {
        "stl_path": str(stl_path),
        "watertight": bool(mesh.is_watertight),
        "volume_cm3": round(mesh.volume / 1000.0, 1),
        "face_count": len(mesh.faces),
        "layout": layout,
    }


def _spawn_renderer(stl_path, output_dir, meta_lines):
    """Run the standalone underbody renderer in a fresh subprocess.

    (Same OCP / PyVista llvmpipe conflict as the wheel pipeline.)
    """
    import json
    import subprocess

    renderer = Path(__file__).resolve().parent / "_render_views.py"
    payload = json.dumps({
        "stl_path": str(Path(stl_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "meta_lines": meta_lines,
    })
    print("[render] spawning subprocess renderer ...")
    result = subprocess.run(
        [sys.executable, str(renderer), payload],
        check=False,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"underbody render subprocess failed: {result.returncode}")


if __name__ == "__main__":
    run_pipeline()
