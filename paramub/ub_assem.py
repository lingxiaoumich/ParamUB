"""Underbody assembler: composes floor + 4 wheelhouses + 4 wheels.

Coordinate system (right-handed, mm):
    +X = vehicle forward, +Y = vehicle right, +Z = up
    origin at ground under the wheelbase midpoint
    front axle at +wheelbase/2, rear axle at -wheelbase/2
    right wheels at +track/2

Sign conventions:
    camber_deg: negative camber = top of wheel tilted INWARD (toward
                centerline). Applied symmetric across the car.
    toe_deg:    positive toe   = TOE-IN (front of wheel rotated inward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians
from typing import Optional

import cadquery as cq

from .floor_builder import (
    FloorSpec,
    build_below_cropper,
    build_floor,
)
from .wheel_assem import WheelSpec, assemble_wheel
from .wheelhouse_builder import (
    WheelhouseSpec,
    build_wheelhouse_solid,
    extract_wheelhouse_surfaces,
)


@dataclass
class UnderbodySpec:
    """One spec covering wheels, floor, and body.

    Vehicle plan
        wheelbase_mm, front_overhang_mm, rear_overhang_mm:
            longitudinal layout.
        track_front_mm, track_rear_mm:
            lateral wheel centers.

    Ride / floor / diffuser
        ride_height_mm:          ground -> floor underside.
        floor_width_mm:          plan width of the rectangular floor.
        floor_thickness_mm:      0 -> surface (Face), >0 -> slab solid.
        floor_angle_deg:         rake about the front edge.
        diffuser_start_x_mm:     None -> rear axle, else absolute X.
        diffuser_angle_deg:      ramp angle.
        diffuser_radius_mm:      tangent fillet between floor and ramp.

    Wheelhouses (per-axle clearances + fillets)
        wheel_house_axial_clearance_mm:   radial gap tire OD -> arch ID.
        wheel_house_thickness_mm:         shell thickness (solid mode only;
                                          surface mode is zero-thickness).
        wheel_house_lateral_clearance_mm: base Y clearance per side.
        front_steering_clearance_mm:      extra Y on front arches.
        rear_steering_clearance_mm:       extra Y on rear arches.
        front_wheel_house_fillet_mm:      inboard fillet radius (front).
        rear_wheel_house_fillet_mm:       inboard fillet radius (rear).

    Wheel alignment
        camber_front_deg / camber_rear_deg / toe_front_deg / toe_rear_deg

    Wheel geometry
        wheel:                full WheelSpec used for ALL four corners
                              unless overridden by rear_wheel.
        rear_wheel:           optional override for the rear axle. If None,
                              uses ``wheel``.
    """
    # plan
    wheelbase_mm: float = 2700.0
    front_overhang_mm: float = 850.0
    rear_overhang_mm: float = 800.0
    track_front_mm: float = 1550.0
    track_rear_mm: float = 1540.0

    # ride / floor / diffuser
    ride_height_mm: float = 100.0
    floor_width_mm: float = 1800.0
    floor_thickness_mm: float = 0.0
    floor_angle_deg: float = 0.0
    diffuser_start_x_mm: Optional[float] = None
    diffuser_angle_deg: float = 7.0
    diffuser_radius_mm: float = 500.0

    # wheelhouses
    wheel_house_axial_clearance_mm: float = 30.0
    wheel_house_thickness_mm: float = 6.0
    wheel_house_lateral_clearance_mm: float = 35.0
    front_steering_clearance_mm: float = 150.0
    rear_steering_clearance_mm: float = 0.0
    front_wheel_house_fillet_mm: float = 100.0
    rear_wheel_house_fillet_mm: float = 30.0

    # alignment
    camber_front_deg: float = -1.5
    camber_rear_deg: float = -1.0
    toe_front_deg: float = 0.10
    toe_rear_deg: float = 0.0

    # wheels
    wheel: WheelSpec = field(default_factory=WheelSpec)
    rear_wheel: Optional[WheelSpec] = None


def _compute_layout(spec: UnderbodySpec) -> dict:
    front_axle_x = +spec.wheelbase_mm / 2.0
    rear_axle_x = -spec.wheelbase_mm / 2.0
    floor_x_max = front_axle_x + spec.front_overhang_mm
    floor_x_min = rear_axle_x - spec.rear_overhang_mm
    diff_start_x = (rear_axle_x if spec.diffuser_start_x_mm is None
                    else float(spec.diffuser_start_x_mm))
    return dict(
        front_axle_x=front_axle_x,
        rear_axle_x=rear_axle_x,
        floor_x_min=floor_x_min,
        floor_x_max=floor_x_max,
        diff_start_x=diff_start_x,
        overall_length_mm=floor_x_max - floor_x_min,
    )


def _place_wheel(wheel_wp, *, side: str, axle_x: float, track: float,
                 camber_deg: float, toe_deg: float, tire_radius: float):
    """Rotate/translate one wheel from local coords into the car frame."""
    sign = +1.0 if side == "right" else -1.0
    spin_axis_rotation = -90.0 if side == "right" else +90.0
    track_y = sign * track / 2.0
    return (
        wheel_wp
        .rotate((0, 0, 0), (1, 0, 0), spin_axis_rotation)
        .rotate((0, 0, 0), (1, 0, 0), -sign * camber_deg)
        .rotate((0, 0, 0), (0, 0, 1), -sign * toe_deg)
        .translate((axle_x, track_y, tire_radius))
    )


def _cut_face_with_solid(face_shape, solid_shape):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    cut = BRepAlgoAPI_Cut(face_shape.wrapped, solid_shape.wrapped)
    cut.Build()
    return cq.Shape.cast(cut.Shape())


def _make_wheelhouse_specs(spec: UnderbodySpec, layout: dict,
                            tire_radius_f: float, tire_width_f: float,
                            tire_radius_r: float, tire_width_r: float):
    front_lat = (spec.wheel_house_lateral_clearance_mm
                 + spec.front_steering_clearance_mm)
    rear_lat = (spec.wheel_house_lateral_clearance_mm
                + spec.rear_steering_clearance_mm)
    y_half = spec.floor_width_mm / 2.0

    targets = [
        # axle_x, y_track, tire_r, tire_w, lat_clear, side, fillet
        (layout["front_axle_x"], +spec.track_front_mm / 2.0,
         tire_radius_f, tire_width_f, front_lat, "right",
         spec.front_wheel_house_fillet_mm),
        (layout["front_axle_x"], -spec.track_front_mm / 2.0,
         tire_radius_f, tire_width_f, front_lat, "left",
         spec.front_wheel_house_fillet_mm),
        (layout["rear_axle_x"], +spec.track_rear_mm / 2.0,
         tire_radius_r, tire_width_r, rear_lat, "right",
         spec.rear_wheel_house_fillet_mm),
        (layout["rear_axle_x"], -spec.track_rear_mm / 2.0,
         tire_radius_r, tire_width_r, rear_lat, "left",
         spec.rear_wheel_house_fillet_mm),
    ]
    out = []
    for ax, yt, tr, tw, lc, side, fr in targets:
        sign = +1.0 if side == "right" else -1.0
        out.append(WheelhouseSpec(
            axle_x=ax, y_track=yt, side=side,
            tire_radius_mm=tr, tire_width_mm=tw,
            axial_clearance_mm=spec.wheel_house_axial_clearance_mm,
            lateral_clearance_mm=lc,
            thickness_mm=spec.wheel_house_thickness_mm,
            fillet_mm=fr,
            ride_height_mm=spec.ride_height_mm,
            floor_edge_y=sign * y_half,
        ))
    return out


def build_underbody(spec: Optional[UnderbodySpec] = None):
    """Top-level: build floor + 4 wheelhouses + place 4 wheels.

    Returns (Assembly, layout dict).
    """
    spec = spec or UnderbodySpec()
    layout = _compute_layout(spec)
    print(f"[layout] {layout}")

    print("[wheels] building wheels via wheel_assem ...")
    front_wheel = assemble_wheel(spec.wheel)
    tire_radius_f = spec.wheel.tire_radius_mm
    tire_width_f = spec.wheel.tire_section_width_mm
    if spec.rear_wheel is None:
        rear_wheel = front_wheel
        tire_radius_r = tire_radius_f
        tire_width_r = tire_width_f
    else:
        rear_wheel = assemble_wheel(spec.rear_wheel)
        tire_radius_r = spec.rear_wheel.tire_radius_mm
        tire_width_r = spec.rear_wheel.tire_section_width_mm
    print(f"  front: OD={2*tire_radius_f:.1f} mm  width={tire_width_f}")
    print(f"  rear : OD={2*tire_radius_r:.1f} mm  width={tire_width_r}")

    print("[body] building floor + wheelhouse surfaces ...")
    floor_spec = FloorSpec(
        floor_x_min=layout["floor_x_min"],
        floor_x_max=layout["floor_x_max"],
        floor_width_mm=spec.floor_width_mm,
        ride_height_mm=spec.ride_height_mm,
        floor_thickness_mm=spec.floor_thickness_mm,
        diffuser_start_x_mm=layout["diff_start_x"],
        diffuser_angle_deg=spec.diffuser_angle_deg,
        diffuser_radius_mm=spec.diffuser_radius_mm,
    )
    floor = build_floor(floor_spec)
    below_cropper = build_below_cropper(floor_spec).val()

    wh_specs = _make_wheelhouse_specs(
        spec, layout,
        tire_radius_f, tire_width_f,
        tire_radius_r, tire_width_r,
    )
    wh_solids = [(s, build_wheelhouse_solid(s)) for s in wh_specs]

    if spec.floor_thickness_mm <= 0:
        # Surface mode: cut the floor Face with each wheelhouse solid so the
        # floor's edge picks up the fillet; extract outer wheelhouse faces.
        for _s, solid in wh_solids:
            floor = _cut_face_with_solid(floor, solid.val())
        wh_faces = []
        for s, solid in wh_solids:
            wh_faces.extend(extract_wheelhouse_surfaces(solid, s, below_cropper))
        body = cq.Compound.makeCompound([floor, *wh_faces])
    else:
        # Slab mode: floor slab + closed half-cylinder wheelhouse solids
        # unioned in. (The thin-shell variant from the prototype lives in
        # paramub/debug/underbody_files/build.py if you need it.)
        body = floor
        for _s, solid in wh_solids:
            try:
                body = body.union(solid)
            except Exception as exc:
                print(f"[warn] wheelhouse union failed: {exc}")

    if abs(spec.floor_angle_deg) > 1e-6:
        pivot_x = layout["floor_x_max"]
        body = body.rotate(
            (pivot_x, 0, spec.ride_height_mm),
            (pivot_x, 1, spec.ride_height_mm),
            -spec.floor_angle_deg,
        )

    print("[wheels] placing 4 corners ...")
    wheels = [
        ("wheel_fl", _place_wheel(front_wheel, side="left",
                                    axle_x=layout["front_axle_x"],
                                    track=spec.track_front_mm,
                                    camber_deg=spec.camber_front_deg,
                                    toe_deg=spec.toe_front_deg,
                                    tire_radius=tire_radius_f)),
        ("wheel_fr", _place_wheel(front_wheel, side="right",
                                    axle_x=layout["front_axle_x"],
                                    track=spec.track_front_mm,
                                    camber_deg=spec.camber_front_deg,
                                    toe_deg=spec.toe_front_deg,
                                    tire_radius=tire_radius_f)),
        ("wheel_rl", _place_wheel(rear_wheel, side="left",
                                    axle_x=layout["rear_axle_x"],
                                    track=spec.track_rear_mm,
                                    camber_deg=spec.camber_rear_deg,
                                    toe_deg=spec.toe_rear_deg,
                                    tire_radius=tire_radius_r)),
        ("wheel_rr", _place_wheel(rear_wheel, side="right",
                                    axle_x=layout["rear_axle_x"],
                                    track=spec.track_rear_mm,
                                    camber_deg=spec.camber_rear_deg,
                                    toe_deg=spec.toe_rear_deg,
                                    tire_radius=tire_radius_r)),
    ]

    print("[assemble] body + 4 wheels ...")
    asy = cq.Assembly()
    asy.add(body, name="body", color=cq.Color(0.78, 0.80, 0.84))
    for name, w in wheels:
        asy.add(w, name=name, color=cq.Color(0.18, 0.18, 0.20))

    return asy, layout
