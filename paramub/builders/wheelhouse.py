"""Wheelhouse builder: single arched shell over one wheel.

Shape (cross-section in XZ at axle Y, viewed with car +X = forward, +Z = up):

    ::

           dome (semicircular, radius = arch_r, centered at tire_r)
              ___
             /   \
            |     |       <- vertical sidewalls down to z = ride_height
        ----|     |----        (chord at the floor)

Extruded in Y over a length of (tire_width + 2 * lateral_clearance), then:

    1. an optional 3D tangent fillet wraps around the inboard cap's full
       upper outline (dome + both sidewalls); the floor cut downstream
       turns the fillet's tail into a clean edge.
    2. the outboard end is trimmed at the floor's lateral perimeter
       (Y = ± floor_width / 2) so the wheelhouse never sticks past the
       floor's plan outline.

The solid form (with the flat floor chord intact) is the right cutter for
the floor surface — that's why this module returns the solid. For the
final surface-mode underbody, :func:`extract_wheelhouse_surfaces` peels
off only the outer wall + cap + fillet faces and crops them against the
diffuser-following ``below_cropper`` solid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cadquery as cq


@dataclass
class WheelhouseSpec:
    """Per-corner inputs for one wheelhouse.

    Geometry & placement
        axle_x:              X (forward) coordinate of the wheel center.
        y_track:             Y (right) coordinate of the wheel center.
        side:                "left" or "right" — picks which face is inboard
                             and which Y direction the outboard trim cuts.
        tire_radius_mm:      tire OD / 2.
        tire_width_mm:       tire axial width (section width).

    Clearances
        axial_clearance_mm:  radial gap between tire OD and arch inner wall.
        lateral_clearance_mm: extra Y per side (includes steering clearance
                              on a front wheel).
        thickness_mm:        intermediate shell thickness; only used inside
                              the wheelhouse solid construction (the final
                              surface-mode output is zero-thickness, with
                              outer faces extracted from the intermediate
                              solid).
        fillet_mm:           tangent fillet radius on the inboard cap's
                              upper outline. 0 disables.

    Underbody reference
        ride_height_mm:      Z of the flat floor underside.
        floor_edge_y:        signed Y of the floor's lateral edge on this
                              side (= ± floor_width / 2). Outboard trim
                              uses this so the arch never sticks past
                              the plan outline.
    """
    axle_x: float
    y_track: float
    side: str
    tire_radius_mm: float
    tire_width_mm: float

    axial_clearance_mm: float = 30.0
    lateral_clearance_mm: float = 35.0
    thickness_mm: float = 6.0
    fillet_mm: float = 100.0

    ride_height_mm: float = 100.0
    floor_edge_y: float = 900.0

    def validate(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        if self.tire_radius_mm <= self.ride_height_mm:
            raise ValueError(
                "tire_radius_mm must exceed ride_height_mm or the wheelhouse "
                "cross-section collapses to a line.")

    # ----- derived -----
    @property
    def arch_radius_mm(self) -> float:
        return self.tire_radius_mm + self.axial_clearance_mm

    @property
    def arch_length_mm(self) -> float:
        return self.tire_width_mm + 2.0 * self.lateral_clearance_mm


def build_wheelhouse_solid(spec: WheelhouseSpec):
    """Closed half-cylinder solid for one wheelhouse, with optional 3D fillet
    on the inboard arc edge and outboard trim at the floor's lateral edge.

    Bottom is FLAT at z = ride_height_mm (the chord that will be cut away by
    the floor downstream). Returns a cq.Workplane wrapping the resulting
    Solid.
    """
    spec.validate()
    sign = +1.0 if spec.side == "right" else -1.0
    arch_r = spec.arch_radius_mm
    arch_length = spec.arch_length_mm
    y_outboard = spec.y_track + sign * arch_length / 2.0

    cross_section = (
        cq.Workplane("XZ")
        .moveTo(spec.axle_x - arch_r, spec.ride_height_mm)
        .lineTo(spec.axle_x - arch_r, spec.tire_radius_mm)
        .threePointArc((spec.axle_x, spec.tire_radius_mm + arch_r),
                       (spec.axle_x + arch_r, spec.tire_radius_mm))
        .lineTo(spec.axle_x + arch_r, spec.ride_height_mm)
        .close()
    )
    solid_wp = (cross_section
                .extrude(arch_length / 2.0, both=True)
                .translate((0, spec.y_track, 0)))

    fillet_r = max(min(spec.fillet_mm, arch_r * 0.9), 0.0)
    if fillet_r > 0:
        inboard_sel = "<Y" if spec.side == "right" else ">Y"
        try:
            inboard_face = solid_wp.faces(inboard_sel).val()
            upper_edges = [e for e in inboard_face.Edges()
                           if e.Center().z > spec.ride_height_mm + 1.0]
            if upper_edges:
                solid_wp = solid_wp.newObject(upper_edges).fillet(fillet_r)
        except Exception as exc:
            print(f"[warn] inboard fillet failed on {spec.side} wheelhouse: {exc}")

    if (sign > 0 and y_outboard > spec.floor_edge_y) or \
       (sign < 0 and y_outboard < spec.floor_edge_y):
        huge = 10000.0
        if sign > 0:
            trim_box = (cq.Workplane("XY").box(huge, huge, huge)
                        .translate((0, spec.floor_edge_y + huge / 2.0, 0)))
        else:
            trim_box = (cq.Workplane("XY").box(huge, huge, huge)
                        .translate((0, spec.floor_edge_y - huge / 2.0, 0)))
        solid_wp = solid_wp.cut(trim_box)

    return solid_wp


def _cut_face_with_solid(face_shape, solid_shape):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    cut = BRepAlgoAPI_Cut(face_shape.wrapped, solid_shape.wrapped)
    cut.Build()
    return cq.Shape.cast(cut.Shape())


def extract_wheelhouse_surfaces(solid_wp, spec: WheelhouseSpec,
                                 below_cropper: Optional[cq.Shape] = None):
    """Peel off the outer wall + inboard cap + fillet faces.

    Drops the flat bottom chord (the floor takes its place) and the outboard
    end cap (it sits outside the floor). If ``below_cropper`` is given, each
    retained face is cropped against it so rear arches follow the diffuser
    slope instead of leaving material hanging below the underbody underside.
    """
    sign = +1.0 if spec.side == "right" else -1.0
    bbox = solid_wp.val().BoundingBox()

    keep = []
    for f in solid_wp.faces().vals():
        fb = f.BoundingBox()
        # Bottom chord (flat at z = ride_height).
        if abs(fb.zmax - spec.ride_height_mm) < 0.5 and (fb.zmax - fb.zmin) < 0.5:
            continue
        # Outboard cap (flat at most-outboard Y).
        if sign > 0:
            if abs(fb.ymin - bbox.ymax) < 0.5 and (fb.ymax - fb.ymin) < 0.5:
                continue
        else:
            if abs(fb.ymax - bbox.ymin) < 0.5 and (fb.ymax - fb.ymin) < 0.5:
                continue
        keep.append(f)

    if below_cropper is None:
        return keep

    cropped = []
    for f in keep:
        cropped_shape = _cut_face_with_solid(f, below_cropper)
        if cropped_shape is not None and not cropped_shape.isNull():
            cropped.append(cropped_shape)
    return cropped
