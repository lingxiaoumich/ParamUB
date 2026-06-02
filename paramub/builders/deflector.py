"""Front tire (air) deflector builder.

A vertical plate hung from the underbody just ahead of each FRONT
wheelhouse, to deflect air around the leading face of the front tire.

Geometry (ParamUB frame: +X forward, +Y right, +Z up; built per-corner
in the same frame as :mod:`ub_assem`):

    - Rectangular plate, ``thickness_mm`` thick in X (across the flow).
    - Positioned ``front_gap_mm`` forward of the wheelhouse FRONT wall
      (the wall sits at ``axle_x + arch_radius``), so the plate's rear face
      is at ``axle_x + arch_radius + front_gap_mm``.
    - Y span covers MOST OF THE TIRE plus a bit inboard:
        outboard bound = ``outboard_inset_mm`` inboard of the tire's
                         OUTBOARD edge (so it nearly reaches the outer edge);
        inboard bound  = ``inboard_inset_mm`` inboard of the tire's
                         INBOARD edge (so it spans the whole tire width and
                         extends a little toward the centerline, y=0).
    - Bottom flat at ``ride_height_mm - drop_mm`` (hangs below the floor).
    - Top flat at ``ride_height_mm + top_extend_mm`` — deliberately ABOVE
      the floor plane so a later boolean trim into the floor + splitter +
      diffuser surface cuts cleanly (the plate crosses the surface
      transversally rather than ending coincident with it).

No trimming is performed here: this returns the raw plate solid. The
underbody surface trim is applied as a separate step at the end of the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq


@dataclass
class FrontDeflectorSpec:
    """One front-tire deflector, dimensioned off its wheelhouse + tire."""
    axle_x: float          # front axle X (ParamUB frame)
    y_track: float         # wheel center Y (signed)
    side: str              # "left" or "right"
    tire_width_mm: float   # tire axial (Y) width
    arch_radius_mm: float  # wheelhouse arch radius (front wall = axle_x + this)
    ride_height_mm: float  # Z of the flat floor underside

    front_gap_mm: float = 20.0       # gap ahead of the wheelhouse front wall
    thickness_mm: float = 3.0        # plate thickness, along X
    outboard_inset_mm: float = 120.0  # inboard of the tire OUTBOARD edge
    inboard_inset_mm: float = 50.0    # inboard of the tire INBOARD edge
    drop_mm: float = 40.0            # how far below the floor the plate hangs
    top_extend_mm: float = 50.0      # how far above the floor the plate rises

    def validate(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        if self.thickness_mm <= 0 or self.drop_mm <= 0:
            raise ValueError("thickness_mm and drop_mm must be > 0")

    # ----- derived bounds (ParamUB frame) -----
    @property
    def _sign(self) -> float:
        return +1.0 if self.side == "right" else -1.0  # outboard direction

    @property
    def x_bounds_mm(self) -> tuple[float, float]:
        x_rear = self.axle_x + self.arch_radius_mm + self.front_gap_mm
        return x_rear, x_rear + self.thickness_mm

    @property
    def y_bounds_mm(self) -> tuple[float, float]:
        sign = self._sign
        tire_out_edge = self.y_track + sign * self.tire_width_mm / 2.0
        tire_in_edge = self.y_track - sign * self.tire_width_mm / 2.0
        # outboard bound: inboard of the OUTBOARD edge.
        y_out = tire_out_edge - sign * self.outboard_inset_mm
        # inboard bound: inboard of the INBOARD edge (toward centerline).
        y_in = tire_in_edge - sign * self.inboard_inset_mm
        return tuple(sorted((y_out, y_in)))

    @property
    def z_bounds_mm(self) -> tuple[float, float]:
        return (self.ride_height_mm - self.drop_mm,
                self.ride_height_mm + self.top_extend_mm)


def build_front_deflector_solid(spec: FrontDeflectorSpec):
    """Return the raw deflector plate as a ``cq.Workplane`` solid (no trim)."""
    spec.validate()
    x_rear, x_front = spec.x_bounds_mm
    y_lo, y_hi = spec.y_bounds_mm
    z_bot, z_top = spec.z_bounds_mm
    return (cq.Workplane("XY")
            .box(x_front - x_rear, y_hi - y_lo, z_top - z_bot, centered=False)
            .translate((x_rear, y_lo, z_bot)))
