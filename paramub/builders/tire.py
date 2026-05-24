"""Tire builder: revolved cross-section tire with rounded shoulder.

Inputs are a :class:`TireSpec` (cross-section, profile, tread) plus the
axial half-width at the bead (``z_bead_mm``), which the spoke/wheel side
of the assembly provides (it equals ``wheel_width_mm / 2``).

Coordinate convention for the returned solid:
    - revolved about the global Z axis (spin axis)
    - centered axially on Z = 0
    - radial direction = X
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import cadquery as cq


@dataclass
class TireSpec:
    """All tire-only dimensions.

    Cross-section
        section_width_mm:   nominal cross-section width (mm); the user-facing
                            tire-spec width, like the "225" in 225/45 R18.
        aspect_ratio:       sidewall height as a percentage of section width;
                            the "45" in 225/45 R18.
        rim_diameter_in:    bead-seat diameter (inches); the "18".

    Sidewall shape
        sidewall_bulge_mm:  max radial bulge of the sidewall away from the
                            straight line bead -> tread edge.
        sidewall_bulge_pos: 0..1 axial position of the bulge maximum
                            (0 = at the bead, 1 = at the shoulder).
        shoulder_radius_mm: depth of the C1 tangent rounding into the tread.

    Tread
        tread_width_mm:     axial width of the crown arc footprint.
        crown_radius_mm:    radius of the tread crown arc (large => flat).
        rim_flange_mm:      radial height of the bead/flange ledge above the
                            rim radius. (Geometrically belongs to the tire's
                            inner profile.)
    """
    # cross-section
    section_width_mm: float = 225.0
    aspect_ratio: float = 45.0
    rim_diameter_in: float = 18.0

    # sidewall shape
    sidewall_bulge_mm: float = 8.0
    sidewall_bulge_pos: float = 0.45
    shoulder_radius_mm: float = 30.0

    # tread
    tread_width_mm: float = 190.0
    crown_radius_mm: float = 400.0
    rim_flange_mm: float = 18.0

    def validate(self) -> None:
        if not (0.0 <= self.sidewall_bulge_pos <= 1.0):
            raise ValueError("sidewall_bulge_pos must be in [0, 1]")
        if self.crown_radius_mm <= self.tread_width_mm / 2.0 + 5.0:
            raise ValueError("crown_radius_mm must exceed tread_width_mm/2 + 5")

    # ---- derived values consumers (e.g. wheel_assem) need to know about ----
    @property
    def rim_radius_mm(self) -> float:
        return self.rim_diameter_in * 25.4 / 2.0

    @property
    def sidewall_height_mm(self) -> float:
        return self.section_width_mm * self.aspect_ratio / 100.0

    @property
    def tire_od_mm(self) -> float:
        return self.rim_radius_mm * 2.0 + self.sidewall_height_mm * 2.0


def _bezier_point(P0, P1, P2, P3, t):
    u = 1.0 - t
    return (
        u * u * u * P0[0] + 3 * u * u * t * P1[0]
        + 3 * u * t * t * P2[0] + t * t * t * P3[0],
        u * u * u * P0[1] + 3 * u * u * t * P1[1]
        + 3 * u * t * t * P2[1] + t * t * t * P3[1],
    )


def build_tire(spec: TireSpec, z_bead_mm: float):
    """Return a revolved tire solid centered on Z = 0.

    z_bead_mm is the axial offset of each bead from the spin-axis midplane
    (in the wheel assembly this is ``wheel_width_mm / 2``).
    """
    spec.validate()

    rim_radius_mm = spec.rim_radius_mm
    sidewall_height_mm = spec.sidewall_height_mm

    z_bead = z_bead_mm
    z_tread_edge = spec.tread_width_mm / 2.0

    R_bead_base = rim_radius_mm
    R_flange_top = rim_radius_mm + spec.rim_flange_mm
    R_tread_center = rim_radius_mm + sidewall_height_mm
    crown_center_R = R_tread_center - spec.crown_radius_mm
    R_tread_edge = crown_center_R + sqrt(spec.crown_radius_mm ** 2 - z_tread_edge ** 2)

    # Tangent to the tread arc at (R_tread_edge, z_tread_edge), pointing
    # toward the tread center (R increasing, Z decreasing).
    tan_R = z_tread_edge / spec.crown_radius_mm
    tan_Z = -sqrt(spec.crown_radius_mm ** 2 - z_tread_edge ** 2) / spec.crown_radius_mm

    # Bezier control points. P3-P2 parallel to the tread tangent gives C1
    # continuity at the tread arc; P1 controls the bulge.
    P0 = (R_flange_top, z_bead)
    P3 = (R_tread_edge, z_tread_edge)
    P2 = (R_tread_edge - spec.shoulder_radius_mm * tan_R,
          z_tread_edge - spec.shoulder_radius_mm * tan_Z)
    P1 = (R_flange_top,
          z_bead + spec.sidewall_bulge_mm * (0.5 + 0.5 * spec.sidewall_bulge_pos))

    N = 32
    sidewall_pos = [_bezier_point(P0, P1, P2, P3, k / N) for k in range(N + 1)]
    sidewall_neg = [(p[0], -p[1]) for p in sidewall_pos]

    z_tread_mid = z_tread_edge / 2.0
    R_tread_mid = crown_center_R + sqrt(spec.crown_radius_mm ** 2 - z_tread_mid ** 2)

    wp = (
        cq.Workplane("XZ")
        .moveTo(R_bead_base, z_bead)
        .lineTo(R_flange_top, z_bead)
        .spline(sidewall_pos[1:], includeCurrent=True)
        .threePointArc((R_tread_mid, z_tread_mid), (R_tread_center, 0))
        .threePointArc((R_tread_mid, -z_tread_mid), (R_tread_edge, -z_tread_edge))
        .spline(list(reversed(sidewall_neg))[1:], includeCurrent=True)
        .lineTo(R_bead_base, -z_bead)
        .close()
    )
    return wp.revolve(axisStart=(0, 0), axisEnd=(0, 1))
