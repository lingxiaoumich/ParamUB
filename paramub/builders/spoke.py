"""Spoke / wheel-disc + rim-barrel builder.

Three independent geometry helpers driven by a single :class:`SpokeSpec`:

- :func:`build_spoke_disc`   — revolved curved disc (hub + spokes + crown cap)
- :func:`cut_spoke_windows`  — punches through the disc to leave N spokes
- :func:`build_rim_barrel`   — hollow C-channel rim with flanges + bead seats

Solids are emitted in the FINAL wheel coordinate frame (spin axis = Z,
center at Z = 0; inboard mounting face at z = +wheel_offset_mm). The tire
sits at the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, degrees, radians, sin

import cadquery as cq


@dataclass
class SpokeSpec:
    """All wheel/disc/rim/spoke-window dimensions.

    Rim & wheel body
        wheel_width_mm:    full axial width of the wheel (flange-tip to
                           flange-tip). Defines the bead axial position
                           (z_bead = wheel_width_mm / 2).
        rim_diameter_in:   bead-seat diameter (same value as TireSpec.rim_diameter_in;
                           re-stated here because the disc / rim are
                           dimensioned from the rim radius independently).
        rim_flange_mm:     radial height of the flange tip above the rim radius.
        hub_bore_mm:       diameter of the central hub bore.
        wheel_offset_mm:   ET — axial position of the mounting face,
                           measured from the spin-axis midplane;
                           positive = mounting face outboard.

    Outboard "dish" face shape
        dish_depth_mm:     axial sag of the outboard face vs. the rim attach
                           plane (0 = flat).
        dish_profile:      "spherical" (smooth bezier) or "conical" (straight).

    Spoke window cutouts
        num_spokes:                number of spokes (cuts = same).
        spoke_width_hub_mm:        chord width of each spoke at its inner end.
        spoke_width_rim_mm:        chord width of each spoke at its outer end.
        window_inner_gap_mm:       radial gap between hub OD and window inner arc.
        window_outer_gap_mm:       radial gap between rim ID and window outer arc.
        window_corner_radius_mm:   2D fillet on each window corner.

    Rim construction
        rim_band_width_mm:         radial wall thickness of the rim band
                                   (also sets the spoke OD).
        flange_axial_thickness_mm: axial thickness of each flange "lip".
        spoke_thickness_mm:        axial thickness of the disc at its OD
                                   (the disc gets thicker toward the hub).
        spoke_outer_crown_mm:      crown height on the OD cap (R direction).
    """
    # rim/wheel body
    wheel_width_mm: float = 215.0
    rim_diameter_in: float = 18.0
    rim_flange_mm: float = 18.0
    hub_bore_mm: float = 72.0
    wheel_offset_mm: float = 35.0

    # outboard face shape
    dish_depth_mm: float = 15.0
    dish_profile: str = "spherical"

    # spoke windows
    num_spokes: int = 7
    spoke_width_hub_mm: float = 28.0
    spoke_width_rim_mm: float = 18.0
    window_inner_gap_mm: float = 12.0
    window_outer_gap_mm: float = 15.0
    window_corner_radius_mm: float = 10.0

    # construction
    rim_band_width_mm: float = 20.0
    flange_axial_thickness_mm: float = 12.0
    spoke_thickness_mm: float = 150.0
    spoke_outer_crown_mm: float = 4.0

    # hub
    # fill_hub_bore: when True the revolved disc profile reaches the spin
    # axis (R=0) so the disc is solid through its centre — i.e. the
    # central axle bore is closed and there is no hole at the centre of
    # the spoke. When False (default) the disc keeps the hub bore.
    fill_hub_bore: bool = False

    def validate(self) -> None:
        if self.num_spokes < 3:
            raise ValueError("num_spokes must be >= 3")
        if self.dish_profile not in {"spherical", "conical"}:
            raise ValueError("dish_profile must be 'spherical' or 'conical'")
        # window radii must leave room for at least one valid spoke
        if self.spoke_outer_r_mm <= self.spoke_inner_r_mm + 12.0:
            raise ValueError(
                "window geometry collapses: increase rim_diameter_in or reduce "
                "window_outer_gap_mm/rim_band_width_mm")

    # ----- derived geometry the builders use -----
    @property
    def rim_radius_mm(self) -> float:
        return self.rim_diameter_in * 25.4 / 2.0

    @property
    def hub_ring_od_mm(self) -> float:
        # 30 mm annular hub ring around the bore
        return self.hub_bore_mm / 2.0 + 30.0

    @property
    def spoke_inner_r_mm(self) -> float:
        return self.hub_ring_od_mm + self.window_inner_gap_mm

    @property
    def spoke_outer_r_mm(self) -> float:
        return self.rim_radius_mm - self.window_outer_gap_mm - self.rim_band_width_mm


def _bezier_point(P0, P1, P2, P3, t):
    u = 1.0 - t
    return (
        u * u * u * P0[0] + 3 * u * u * t * P1[0]
        + 3 * u * t * t * P2[0] + t * t * t * P3[0],
        u * u * u * P0[1] + 3 * u * u * t * P1[1]
        + 3 * u * t * t * P2[1] + t * t * t * P3[1],
    )


def build_spoke_disc(spec: SpokeSpec):
    """Revolved disc: thick hub + curved spoke profile + thin rim attach.

    The disc is asymmetric (biased outboard like a real positive-ET wheel).
    Cross-section walks closed CCW from the hub bore around the outer profile
    and back. Bezier segments give C1 continuity at the inboard spoke /
    OD-cap / outboard spoke joins.
    """
    spec.validate()
    R_hub = spec.hub_bore_mm / 2.0
    # Inner radius of the revolved profile. With fill_hub_bore the profile
    # runs all the way to the spin axis (R=0), so the revolve produces a
    # solid disc with no central hole; otherwise it stops at the hub bore.
    R_inner = 0.0 if spec.fill_hub_bore else R_hub
    R_hub_outer = spec.hub_ring_od_mm
    # 0.5 mm overlap into the rim so the boolean union closes cleanly.
    R_disc_outer = spec.rim_radius_mm - spec.rim_band_width_mm + 0.5

    z_mount = spec.wheel_offset_mm
    z_attach_outboard = spec.wheel_width_mm / 2.0
    z_attach_inboard = z_attach_outboard - spec.spoke_thickness_mm
    z_hub_outboard = z_attach_outboard - spec.dish_depth_mm
    if z_hub_outboard <= z_mount + 1.0:
        z_hub_outboard = z_mount + max(spec.spoke_thickness_mm, 5.0)

    dR_in = (R_disc_outer - R_hub_outer) * 0.55
    dR_cap_blend = min(
        max(spec.spoke_outer_crown_mm * 2.5, 8.0),
        max((R_disc_outer - R_hub_outer) * 0.35, 8.0),
    )
    P0_in = (R_hub_outer, z_mount)
    P1_in = (R_hub_outer + dR_in, z_mount)
    P2_in = (R_disc_outer - dR_cap_blend, z_attach_inboard)
    P3_in = (R_disc_outer, z_attach_inboard)

    cap_handle_R = max(spec.spoke_outer_crown_mm, 0.0) / 0.75
    P0_cap = P3_in
    P1_cap = (R_disc_outer + cap_handle_R, z_attach_inboard)
    P2_cap = (R_disc_outer + cap_handle_R, z_attach_outboard)
    P3_cap = (R_disc_outer, z_attach_outboard)

    dR_out = (R_disc_outer - R_hub_outer) * 0.55
    dR_out_start = (R_disc_outer - R_hub_outer) * 0.30
    P0_out = (R_disc_outer, z_attach_outboard)
    P1_out = (R_disc_outer - dR_out_start, z_attach_outboard)
    P2_out = (R_hub_outer + dR_out, z_hub_outboard)
    P3_out = (R_hub_outer, z_hub_outboard)

    N = 28
    if spec.dish_profile == "spherical":
        inboard_pts = [_bezier_point(P0_in, P1_in, P2_in, P3_in, k / N)
                       for k in range(N + 1)]
        outboard_pts = [_bezier_point(P0_out, P1_out, P2_out, P3_out, k / N)
                        for k in range(N + 1)]
    else:  # conical
        inboard_pts = [(P0_in[0] + (P3_in[0] - P0_in[0]) * k / N,
                        P0_in[1] + (P3_in[1] - P0_in[1]) * k / N)
                       for k in range(N + 1)]
        outboard_pts = [(P0_out[0] + (P3_out[0] - P0_out[0]) * k / N,
                         P0_out[1] + (P3_out[1] - P0_out[1]) * k / N)
                        for k in range(N + 1)]
    cap_pts = [_bezier_point(P0_cap, P1_cap, P2_cap, P3_cap, k / N)
               for k in range(N + 1)]

    wp = (
        cq.Workplane("XZ")
        .moveTo(R_inner, z_mount)
        .lineTo(R_hub_outer, z_mount)
        .spline(inboard_pts[1:], includeCurrent=True)
        .spline(cap_pts[1:], includeCurrent=True)
        .spline(outboard_pts[1:], includeCurrent=True)
        .lineTo(R_inner, z_hub_outboard)
        .close()
    )
    return wp.revolve(axisStart=(0, 0), axisEnd=(0, 1))


def _window_face(center_angle_deg: float, spec: SpokeSpec):
    spoke_inner_r = spec.spoke_inner_r_mm
    spoke_outer_r = spec.spoke_outer_r_mm

    spoke_pitch_deg = 360.0 / spec.num_spokes
    spoke_inner_deg = degrees(2 * atan(spec.spoke_width_hub_mm / 2 / spoke_inner_r))
    spoke_outer_deg = degrees(2 * atan(spec.spoke_width_rim_mm / 2 / spoke_outer_r))
    window_inner_half_deg = (spoke_pitch_deg - spoke_inner_deg) / 2
    window_outer_half_deg = (spoke_pitch_deg - spoke_outer_deg) / 2

    a_inner_lo = radians(center_angle_deg - window_inner_half_deg)
    a_inner_hi = radians(center_angle_deg + window_inner_half_deg)
    a_outer_lo = radians(center_angle_deg - window_outer_half_deg)
    a_outer_hi = radians(center_angle_deg + window_outer_half_deg)
    a_mid = radians(center_angle_deg)

    p_inner_lo = (spoke_inner_r * cos(a_inner_lo), spoke_inner_r * sin(a_inner_lo))
    p_inner_hi = (spoke_inner_r * cos(a_inner_hi), spoke_inner_r * sin(a_inner_hi))
    p_outer_lo = (spoke_outer_r * cos(a_outer_lo), spoke_outer_r * sin(a_outer_lo))
    p_outer_hi = (spoke_outer_r * cos(a_outer_hi), spoke_outer_r * sin(a_outer_hi))
    p_inner_mid = (spoke_inner_r * cos(a_mid), spoke_inner_r * sin(a_mid))
    p_outer_mid = (spoke_outer_r * cos(a_mid), spoke_outer_r * sin(a_mid))

    raw_wire = (
        cq.Workplane("XY")
        .moveTo(*p_inner_lo)
        .threePointArc(p_inner_mid, p_inner_hi)
        .lineTo(*p_outer_hi)
        .threePointArc(p_outer_mid, p_outer_lo)
        .close()
        .val()
    )
    if spec.window_corner_radius_mm <= 0:
        return cq.Face.makeFromWires(raw_wire)
    for scale in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
        try:
            filleted = raw_wire.fillet2D(spec.window_corner_radius_mm * scale,
                                          raw_wire.Vertices())
            return cq.Face.makeFromWires(filleted)
        except Exception:
            continue
    return cq.Face.makeFromWires(raw_wire)


def cut_spoke_windows(disc, spec: SpokeSpec):
    """Punch ``num_spokes`` windows through ``disc``, leaving spokes between."""
    cut_depth = spec.wheel_width_mm * 4.0
    pitch_deg = 360.0 / spec.num_spokes
    base_center_angle_deg = 180.0 / spec.num_spokes
    face = _window_face(base_center_angle_deg, spec)
    solid = cq.Solid.extrudeLinear(face, cq.Vector(0, 0, cut_depth))
    solid = solid.translate(cq.Vector(0, 0, -cut_depth / 2))
    for i in range(spec.num_spokes):
        rotated = solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), i * pitch_deg)
        disc = disc.cut(rotated)
    return disc


def build_rim_barrel(spec: SpokeSpec):
    """Hollow C-channel rim with flanges + bead seats at each end."""
    R_outer = spec.rim_radius_mm
    R_flange = spec.rim_radius_mm + spec.rim_flange_mm
    R_inner = spec.rim_radius_mm - spec.rim_band_width_mm
    z_out = spec.wheel_width_mm / 2.0
    z_in = -spec.wheel_width_mm / 2.0
    f = spec.flange_axial_thickness_mm

    profile = (
        cq.Workplane("XZ")
        .moveTo(R_inner, z_out)
        .lineTo(R_flange, z_out)
        .lineTo(R_flange, z_out - f)
        .lineTo(R_outer, z_out - f)
        .lineTo(R_outer, z_in + f)
        .lineTo(R_flange, z_in + f)
        .lineTo(R_flange, z_in)
        .lineTo(R_inner, z_in)
        .close()
    )
    return profile.revolve(axisStart=(0, 0), axisEnd=(0, 1))
