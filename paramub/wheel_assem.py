"""Wheel assembler: composes :mod:`tire_builder` + :mod:`spoke_builder`.

Builds and unions the tire (revolved cross-section), the wheel disc (with
spoke window cutouts), and the rim barrel into a single solid.

The wheel comes out centered on Z = 0 with its spin axis along Z.
Downstream code (e.g. :mod:`ub_assem`) rotates it to put the spin axis
along Y and translates each corner into place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cadquery as cq

from .spoke_builder import (
    SpokeSpec,
    build_rim_barrel,
    build_spoke_disc,
    cut_spoke_windows,
)
from .tire_builder import TireSpec, build_tire


@dataclass
class WheelSpec:
    """Composite spec for a complete wheel (tire + disc + rim).

    `tire` and `spoke` are full sub-specs you can edit independently;
    just be aware that ``tire.rim_diameter_in`` and ``spoke.rim_diameter_in``
    must agree (the assembler enforces this and copies the tire value over
    if they differ, so it's safe to set just one).
    """
    tire: TireSpec = field(default_factory=TireSpec)
    spoke: SpokeSpec = field(default_factory=SpokeSpec)

    def __post_init__(self):
        # The rim radius is shared by both halves of the assembly. Treat
        # tire.rim_diameter_in as authoritative.
        self.spoke.rim_diameter_in = self.tire.rim_diameter_in

    @property
    def tire_radius_mm(self) -> float:
        return self.tire.tire_od_mm / 2.0

    @property
    def tire_section_width_mm(self) -> float:
        return self.tire.section_width_mm


def _safe_union(a, b, label):
    try:
        return a.union(b)
    except Exception as exc:
        raise ValueError(f"union failed at '{label}': {exc}") from exc


def assemble_wheel(spec: WheelSpec):
    """Return a single Workplane wheel: tire ∪ (disc \\ windows) ∪ rim."""
    tire = build_tire(spec.tire, z_bead_mm=spec.spoke.wheel_width_mm / 2.0)
    disc = build_spoke_disc(spec.spoke)
    disc_cut = cut_spoke_windows(disc, spec.spoke)
    barrel = build_rim_barrel(spec.spoke)
    wheel = _safe_union(disc_cut, barrel, "disc + barrel")
    wheel = _safe_union(wheel, tire, "wheel + tire")
    return wheel
