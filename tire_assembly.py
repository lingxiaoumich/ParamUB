"""Parametric tire + wheel assembly built with CadQuery.

The script revolves a 2D cross-section profile to make the tire, builds a
revolved wheel disc with a curved outboard face, cuts spoke windows out of
the disc, unions the disc with a rim barrel and the tire, exports an STL,
validates watertightness with trimesh, and renders preview images via
PyVista.
"""

import os
import sys
from contextlib import contextmanager
from math import sqrt, sin, cos, atan, radians, degrees

import cadquery as cq
from cadquery import exporters

# ---------------------------------------------------------------------------
# PARAMETERS (every dimension lives here)
# ---------------------------------------------------------------------------

# Tire
section_width_mm   = 225
aspect_ratio       = 45
rim_diameter_in    = 18

# Sidewall shape
sidewall_bulge_mm  = 8
sidewall_bulge_pos = 0.45
shoulder_radius_mm = 30

# Tread
tread_width_mm     = 190
crown_radius_mm    = 400

# Wheel dimensions
wheel_width_mm     = 215
rim_flange_mm      = 18
hub_bore_mm        = 72
wheel_offset_mm    = 35     # ET value; positive = mounting face outboard of centerline

# Wheel face dish (curved outboard surface)
dish_depth_mm      = 15     # axial depth of the outboard face curvature; 0 = flat
dish_profile       = "spherical"  # "spherical" or "conical"

# Spokes (cutout method)
num_spokes              = 7
spoke_width_hub_mm      = 28
spoke_width_rim_mm      = 18
window_inner_gap_mm     = 12
window_outer_gap_mm     = 15
window_corner_radius_mm = 10

# Wheel construction (not in the original kickoff parameter list but needed
# to give the wheel a realistic C-channel rim with a curved spoke face
# rather than a solid block):
rim_band_width_mm         = 20  # radial thickness of the rim wall (also matches
                                # the implicit 20 mm in spoke_outer_r_mm derivation)
flange_axial_thickness_mm = 12  # axial thickness of each flange "lip"
spoke_thickness_mm        = 150 # axial thickness of the disc at the rim attachment
                                # (the disc is thicker than this at the hub end)
spoke_outer_crown_mm      = 4   # radial crown on the spoke OD cap where it
                                # blends into the rim-side attachment


PARAMETER_GROUPS = {
    "tire": [
        "section_width_mm",
        "aspect_ratio",
        "rim_diameter_in",
        "sidewall_bulge_mm",
        "sidewall_bulge_pos",
        "shoulder_radius_mm",
        "tread_width_mm",
        "crown_radius_mm",
    ],
    "wheel": [
        "wheel_width_mm",
        "rim_flange_mm",
        "hub_bore_mm",
        "wheel_offset_mm",
        "dish_depth_mm",
        "dish_profile",
    ],
    "spokes": [
        "num_spokes",
        "spoke_width_hub_mm",
        "spoke_width_rim_mm",
        "window_inner_gap_mm",
        "window_outer_gap_mm",
        "window_corner_radius_mm",
    ],
    "construction": [
        "rim_band_width_mm",
        "flange_axial_thickness_mm",
        "spoke_thickness_mm",
        "spoke_outer_crown_mm",
    ],
}

ALL_PARAMETER_NAMES = [
    name
    for group_names in PARAMETER_GROUPS.values()
    for name in group_names
]

DEFAULT_PARAMETERS = {name: globals()[name] for name in ALL_PARAMETER_NAMES}


def get_current_parameters():
    return {name: globals()[name] for name in ALL_PARAMETER_NAMES}


def normalize_parameters(overrides=None):
    spec = DEFAULT_PARAMETERS.copy()
    if overrides:
        unknown = sorted(set(overrides) - set(ALL_PARAMETER_NAMES))
        if unknown:
            raise KeyError(f"unknown parameter(s): {', '.join(unknown)}")
        spec.update(overrides)

    notes = []

    spec["num_spokes"] = int(round(spec["num_spokes"]))
    if spec["num_spokes"] < 3:
        notes.append(f"num_spokes raised from {spec['num_spokes']} to 3")
        spec["num_spokes"] = 3

    clamped_bulge_pos = min(max(float(spec["sidewall_bulge_pos"]), 0.0), 1.0)
    if clamped_bulge_pos != spec["sidewall_bulge_pos"]:
        notes.append(
            f"sidewall_bulge_pos clamped from {spec['sidewall_bulge_pos']} to {clamped_bulge_pos}"
        )
        spec["sidewall_bulge_pos"] = clamped_bulge_pos

    min_wheel_width = spec["tread_width_mm"] + 8.0
    if spec["wheel_width_mm"] <= min_wheel_width:
        notes.append(
            f"wheel_width_mm raised from {spec['wheel_width_mm']} to {min_wheel_width:.1f}"
        )
        spec["wheel_width_mm"] = min_wheel_width

    min_crown_radius = spec["tread_width_mm"] / 2.0 + 5.0
    if spec["crown_radius_mm"] <= min_crown_radius:
        notes.append(
            f"crown_radius_mm raised from {spec['crown_radius_mm']} to {min_crown_radius:.1f}"
        )
        spec["crown_radius_mm"] = min_crown_radius

    if spec["window_corner_radius_mm"] < 0:
        notes.append(
            f"window_corner_radius_mm raised from {spec['window_corner_radius_mm']} to 0"
        )
        spec["window_corner_radius_mm"] = 0

    if spec["spoke_outer_crown_mm"] < 0:
        notes.append(
            f"spoke_outer_crown_mm raised from {spec['spoke_outer_crown_mm']} to 0"
        )
        spec["spoke_outer_crown_mm"] = 0

    if spec["dish_profile"] not in {"spherical", "conical"}:
        raise ValueError(
            f"dish_profile must be 'spherical' or 'conical', got {spec['dish_profile']!r}"
        )

    rim_radius_mm = spec["rim_diameter_in"] * 25.4 / 2.0
    hub_ring_od_mm = spec["hub_bore_mm"] / 2.0 + 30.0
    spoke_inner_r_mm = hub_ring_od_mm + spec["window_inner_gap_mm"]
    spoke_outer_r_mm = rim_radius_mm - spec["window_outer_gap_mm"] - spec["rim_band_width_mm"]
    min_spoke_span_mm = 12.0
    if spoke_outer_r_mm <= spoke_inner_r_mm + min_spoke_span_mm:
        target_rim_band_mm = max(
            8.0,
            rim_radius_mm - spec["window_outer_gap_mm"] - spoke_inner_r_mm - min_spoke_span_mm,
        )
        if target_rim_band_mm < spec["rim_band_width_mm"]:
            notes.append(
                "rim_band_width_mm reduced from "
                f"{spec['rim_band_width_mm']} to {target_rim_band_mm:.1f} to preserve spoke window span"
            )
            spec["rim_band_width_mm"] = target_rim_band_mm
        else:
            raise ValueError(
                "window geometry invalid: spoke window span collapses before cuts can be made"
            )

    return spec, notes


def apply_parameters(spec):
    for name in ALL_PARAMETER_NAMES:
        globals()[name] = spec[name]


@contextmanager
def temporary_parameters(overrides=None):
    original = get_current_parameters()
    spec, notes = normalize_parameters(overrides)
    apply_parameters(spec)
    try:
        yield spec, notes
    finally:
        apply_parameters(original)


# ---------------------------------------------------------------------------
# STEP 1 — derived dimensions
# ---------------------------------------------------------------------------

def compute_derived():
    rim_radius_mm      = rim_diameter_in * 25.4 / 2
    sidewall_height_mm = section_width_mm * aspect_ratio / 100
    tire_od_mm         = rim_radius_mm * 2 + sidewall_height_mm * 2
    hub_ring_od_mm     = hub_bore_mm / 2 + 30            # 30 mm annular hub ring
    spoke_inner_r_mm   = hub_ring_od_mm + window_inner_gap_mm
    spoke_outer_r_mm   = rim_radius_mm - window_outer_gap_mm - rim_band_width_mm

    d = dict(
        rim_radius_mm=rim_radius_mm,
        sidewall_height_mm=sidewall_height_mm,
        tire_od_mm=tire_od_mm,
        hub_ring_od_mm=hub_ring_od_mm,
        spoke_inner_r_mm=spoke_inner_r_mm,
        spoke_outer_r_mm=spoke_outer_r_mm,
    )

    print("[derived dimensions]")
    for k, v in d.items():
        print(f"  {k:<22} = {v:10.3f}")
    return d


# ---------------------------------------------------------------------------
# STEP 2 — tire (revolved cross-section)
# ---------------------------------------------------------------------------

def _bezier_point(P0, P1, P2, P3, t):
    u = 1.0 - t
    return (
        u * u * u * P0[0] + 3 * u * u * t * P1[0] + 3 * u * t * t * P2[0] + t * t * t * P3[0],
        u * u * u * P0[1] + 3 * u * u * t * P1[1] + 3 * u * t * t * P2[1] + t * t * t * P3[1],
    )


def build_tire(d):
    """Revolved tire solid with a rounded shoulder.

    Profile lives in the workplane "XZ" so local x = global X (radial) and
    local y = global Z (axial). Revolve around the workplane's Y axis (=
    global Z) for a tire centred at Z = 0.

    The kickoff spec called for an explicit sidewall Bezier + separate
    shoulder arc + tread arc, but with the given parameter values the
    shoulder arc cannot be made tangent to the tread arc, which leaves a
    visible kink ("sharp shoulder"). To produce a rounded shoulder we
    extend the sidewall Bezier all the way to the tread-arc start point
    and choose the bezier's last control point so that the bezier's
    tangent at that endpoint matches the tread arc's tangent (C1
    continuity). The shoulder_radius_mm parameter now controls how far
    back P2 sits from P3, which sets the depth of the shoulder rounding.
    """
    rim_radius_mm      = d["rim_radius_mm"]
    sidewall_height_mm = d["sidewall_height_mm"]

    # Axial positions (Z >= 0 half).
    z_bead       = wheel_width_mm / 2              # bead at the rim flange axial position
    z_tread_edge = tread_width_mm / 2              # tread arc starts here

    # Radial positions.
    R_bead_base     = rim_radius_mm
    R_flange_top    = rim_radius_mm + rim_flange_mm
    R_tread_center  = rim_radius_mm + sidewall_height_mm   # = tire_od / 2

    # Tread crown arc — centred on the spin axis at (crown_center_R, 0).
    crown_center_R = R_tread_center - crown_radius_mm
    R_tread_edge   = crown_center_R + sqrt(crown_radius_mm ** 2 - z_tread_edge ** 2)

    # Tread arc tangent at (R_tread_edge, z_tread_edge), oriented going from
    # the tread edge toward the tread centre (R increasing, Z decreasing).
    #   tangent = (z_tread_edge / R_crown, -sqrt(R_crown^2 - z_tread_edge^2) / R_crown)
    tan_R = z_tread_edge / crown_radius_mm
    tan_Z = -sqrt(crown_radius_mm ** 2 - z_tread_edge ** 2) / crown_radius_mm

    # Sidewall Bezier (4 control points). bead -> tread edge, with the
    # bulge driven by P1 and tangent continuity to the tread arc enforced
    # via P2.
    #   P3 - P2 must be parallel to (tan_R, tan_Z), so
    #   P2 = P3 - shoulder_radius_mm * (tan_R, tan_Z)
    P0 = (R_flange_top, z_bead)
    P1 = (R_flange_top, z_bead + sidewall_bulge_mm)               # initial tangent is +Z
    P3 = (R_tread_edge, z_tread_edge)
    P2 = (R_tread_edge - shoulder_radius_mm * tan_R,
          z_tread_edge - shoulder_radius_mm * tan_Z)

    # sidewall_bulge_pos still informs WHERE the maximum bulge sits — bias P1
    # axially by the bulge_pos fraction of the bulge magnitude so the user
    # parameter still has an effect on the bezier shape.
    P1 = (P1[0], z_bead + sidewall_bulge_mm * (0.5 + 0.5 * sidewall_bulge_pos))

    N_SAMPLES = 32
    sidewall_pos = [_bezier_point(P0, P1, P2, P3, k / N_SAMPLES)
                    for k in range(N_SAMPLES + 1)]
    sidewall_neg = [(p[0], -p[1]) for p in sidewall_pos]

    # Tread arc midpoint (for threePointArc)
    z_tread_mid = z_tread_edge / 2.0
    R_tread_mid = crown_center_R + sqrt(crown_radius_mm ** 2 - z_tread_mid ** 2)

    wp = (
        cq.Workplane("XZ")
        .moveTo(R_bead_base, z_bead)
        .lineTo(R_flange_top, z_bead)                   # flange face (radial)
        .spline(sidewall_pos[1:], includeCurrent=True)  # sidewall + shoulder rounding
        .threePointArc((R_tread_mid, z_tread_mid), (R_tread_center, 0))  # tread crown
        .threePointArc((R_tread_mid, -z_tread_mid), (R_tread_edge, -z_tread_edge))
        .spline(list(reversed(sidewall_neg))[1:], includeCurrent=True)
        .lineTo(R_bead_base, -z_bead)
        .close()
    )

    return wp.revolve(axisStart=(0, 0), axisEnd=(0, 1))


# ---------------------------------------------------------------------------
# STEP 3 — wheel disc with curved outboard face
# ---------------------------------------------------------------------------

def build_wheel_disc(d):
    """Curved wheel disc with an inboard mounting face.

    Built directly in FINAL world coordinates (no later offset shift):
      - flat inboard mounting face at z = +wheel_offset_mm
        (= +ET), spanning radially from the hub bore to the hub ring OD
      - the spoke section curves axially outboard as it goes radially
        outward, attaching to the rim's inner wall near the OUTBOARD rim
        end at z ≈ +(wheel_width / 2 − flange_axial_thickness_mm)
      - so the disc is asymmetric: thick at the hub, thin near the rim,
        biased toward the outboard side (like a real positive-offset
        wheel — see the cross-section reference in the kickoff thread).

    Cross-section profile, walking clockwise around the closed wire:

           R_hub               R_hub_outer                R_disc_outer
             |                      |                           |
       z_hub_out  +--- bezier  outboard -------------- bezier --+ z_attach_out
                 |              spoke surface                   |
                 |                                              |
       z_mount   +--- mounting face --+                         |
                                      |                         |
                                      +---- bezier  inboard ----+ z_attach_in
                                            spoke surface

    dish_depth_mm sets z_hub_outboard = z_attach_outboard − dish_depth_mm,
    i.e., how far axially inward the hub's outboard face sits relative to
    the rim attachment. dish_profile selects bezier (smooth, "spherical")
    vs. a straight line ("conical").
    """
    rim_radius_mm = d["rim_radius_mm"]

    R_hub        = hub_bore_mm / 2
    R_hub_outer  = d["hub_ring_od_mm"]                  # = R_hub + 30 (per spec)
    # Push the disc 0.5 mm into the rim so the boolean union closes cleanly
    # — otherwise the two surfaces are exactly tangent and the STL tessellation
    # leaves a hairline gap that defeats watertight checks.
    R_disc_outer = rim_radius_mm - rim_band_width_mm + 0.5

    # All Z values are FINAL coords (centre of wheel = Z = 0).
    z_mount           = wheel_offset_mm                                       # +ET
    # The disc OD must be FLUSH with the rim's OB face so the cross-section
    # shows one continuous tangent surface (no axial step in front of the
    # rim flange). The OD cap therefore extends all the way out to
    # +wheel_width/2 instead of stopping at the inboard side of the
    # outboard flange.
    z_attach_outboard = wheel_width_mm / 2
    z_attach_inboard  = z_attach_outboard - spoke_thickness_mm                # disc thickness at rim
    z_hub_outboard    = z_attach_outboard - dish_depth_mm                     # dish-depth sag

    if z_hub_outboard <= z_mount + 1.0:
        # If the parameters would collapse the hub thickness, give it a
        # minimum so the closed wire stays valid.
        z_hub_outboard = z_mount + max(spoke_thickness_mm, 5.0)

    # Inboard spoke surface: bezier from the hub outer ring to the IB end
    # of the spoke OD cap. The end tangent is radial (+R) so the surface
    # rolls into the crowned OD cap with C1 continuity at z_attach_inboard.
    dR_in = (R_disc_outer - R_hub_outer) * 0.55
    dR_cap_blend = min(
        max(spoke_outer_crown_mm * 2.5, 8.0),
        max((R_disc_outer - R_hub_outer) * 0.35, 8.0),
    )
    P0_in = (R_hub_outer, z_mount)
    P1_in = (R_hub_outer + dR_in, z_mount)
    P2_in = (R_disc_outer - dR_cap_blend, z_attach_inboard)
    P3_in = (R_disc_outer, z_attach_inboard)

    # OD cap with a small outward crown.
    #
    # Both endpoints sit on R = R_disc_outer (the rim's inner wall, with
    # the 0.5 mm overlap baked in), and BOTH ends carry a radial tangent
    # — +R at the inboard end (so it blends into the inboard spoke
    # surface, whose end tangent is also +R) and -R at the outboard end
    # (so the cap rolls into the rim's OB face plane, which is itself
    # radial, with no axial step). Control-point R offset drives the
    # crown magnitude: the cubic bezier midpoint sits at
    #   R_mid = R_disc_outer + 0.375 * (cap_p1_offset + cap_p2_offset)
    # so picking each handle equal to spoke_outer_crown_mm / 0.75 lands
    # the midpoint exactly spoke_outer_crown_mm proud of R_disc_outer.
    cap_handle_R = max(spoke_outer_crown_mm, 0.0) / 0.75
    P0_cap = P3_in
    P1_cap = (R_disc_outer + cap_handle_R, z_attach_inboard)   # tangent +R at IB
    P2_cap = (R_disc_outer + cap_handle_R, z_attach_outboard)  # tangent -R at OB
    P3_cap = (R_disc_outer, z_attach_outboard)

    # Outboard spoke surface: bezier from the OB end of the cap back to
    # the hub. It starts with a radial -R tangent so the cap and the
    # outboard surface form one continuous C1 curve at the OB rim attach
    # — there is no step and no kink at z = +wheel_width/2.
    dR_out = (R_disc_outer - R_hub_outer) * 0.55
    dR_out_start = (R_disc_outer - R_hub_outer) * 0.30
    P0_out = (R_disc_outer, z_attach_outboard)
    P1_out = (R_disc_outer - dR_out_start, z_attach_outboard)   # tangent -R at OB
    P2_out = (R_hub_outer + dR_out, z_hub_outboard)
    P3_out = (R_hub_outer, z_hub_outboard)

    N = 28
    if dish_profile == "spherical":
        inboard_pts  = [_bezier_point(P0_in,  P1_in,  P2_in,  P3_in,  k / N)
                        for k in range(N + 1)]
        cap_pts      = [_bezier_point(P0_cap, P1_cap, P2_cap, P3_cap, k / N)
                        for k in range(N + 1)]
        outboard_pts = [_bezier_point(P0_out, P1_out, P2_out, P3_out, k / N)
                        for k in range(N + 1)]
    elif dish_profile == "conical":
        # Straight-ish spoke sides, but keep the OD crown/round so the rim
        # interface stays smooth in section.
        inboard_pts  = [(P0_in[0]  + (P3_in[0]  - P0_in[0])  * k / N,
                         P0_in[1]  + (P3_in[1]  - P0_in[1])  * k / N)
                        for k in range(N + 1)]
        cap_pts      = [_bezier_point(P0_cap, P1_cap, P2_cap, P3_cap, k / N)
                        for k in range(N + 1)]
        outboard_pts = [(P0_out[0] + (P3_out[0] - P0_out[0]) * k / N,
                         P0_out[1] + (P3_out[1] - P0_out[1]) * k / N)
                        for k in range(N + 1)]
    else:
        raise ValueError(f"unknown dish_profile: {dish_profile!r}")

    wp = (
        cq.Workplane("XZ")
        .moveTo(R_hub, z_mount)                              # mounting face hub-bore corner
        .lineTo(R_hub_outer, z_mount)                        # along the mounting face
        .spline(inboard_pts[1:],  includeCurrent=True)       # inboard spoke surface
        .spline(cap_pts[1:], includeCurrent=True)            # crowned OD cap
        .spline(outboard_pts[1:], includeCurrent=True)       # outboard spoke surface
        .lineTo(R_hub, z_hub_outboard)                       # over to hub bore at outboard
        .close()                                             # close along hub bore wall
    )

    return wp.revolve(axisStart=(0, 0), axisEnd=(0, 1))


# ---------------------------------------------------------------------------
# STEP 4 — spoke window cutouts
# ---------------------------------------------------------------------------

def _window_face(center_angle_deg, d):
    spoke_inner_r = d["spoke_inner_r_mm"]
    spoke_outer_r = d["spoke_outer_r_mm"]

    spoke_pitch_deg = 360.0 / num_spokes
    spoke_inner_deg = degrees(2 * atan(spoke_width_hub_mm / 2 / spoke_inner_r))
    spoke_outer_deg = degrees(2 * atan(spoke_width_rim_mm / 2 / spoke_outer_r))
    window_inner_half_deg = (spoke_pitch_deg - spoke_inner_deg) / 2
    window_outer_half_deg = (spoke_pitch_deg - spoke_outer_deg) / 2

    a_inner_lo = radians(center_angle_deg - window_inner_half_deg)
    a_inner_hi = radians(center_angle_deg + window_inner_half_deg)
    a_outer_lo = radians(center_angle_deg - window_outer_half_deg)
    a_outer_hi = radians(center_angle_deg + window_outer_half_deg)
    a_mid      = radians(center_angle_deg)

    p_inner_lo  = (spoke_inner_r * cos(a_inner_lo), spoke_inner_r * sin(a_inner_lo))
    p_inner_hi  = (spoke_inner_r * cos(a_inner_hi), spoke_inner_r * sin(a_inner_hi))
    p_outer_lo  = (spoke_outer_r * cos(a_outer_lo), spoke_outer_r * sin(a_outer_lo))
    p_outer_hi  = (spoke_outer_r * cos(a_outer_hi), spoke_outer_r * sin(a_outer_hi))
    p_inner_mid = (spoke_inner_r * cos(a_mid),      spoke_inner_r * sin(a_mid))
    p_outer_mid = (spoke_outer_r * cos(a_mid),      spoke_outer_r * sin(a_mid))

    raw_wire = (
        cq.Workplane("XY")
        .moveTo(*p_inner_lo)
        .threePointArc(p_inner_mid, p_inner_hi)
        .lineTo(*p_outer_hi)
        .threePointArc(p_outer_mid, p_outer_lo)
        .close()
        .val()
    )
    if window_corner_radius_mm <= 0:
        return cq.Face.makeFromWires(raw_wire), 0.0

    last_exc = None
    for scale in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
        radius = window_corner_radius_mm * scale
        try:
            filleted = raw_wire.fillet2D(radius, raw_wire.Vertices())
            return cq.Face.makeFromWires(filleted), radius
        except Exception as exc:
            last_exc = exc

    print(
        "  [warn] spoke window fillet could not be applied; "
        "falling back to sharp-corner windows"
    )
    if last_exc:
        print(f"         last fillet error: {last_exc}")
    return cq.Face.makeFromWires(raw_wire), 0.0


def cut_spoke_windows(disc, d):
    cut_depth = wheel_width_mm * 4.0  # generous, "through all"
    pitch_deg = 360.0 / num_spokes
    base_center_angle_deg = 180.0 / num_spokes
    face, actual_corner_radius_mm = _window_face(base_center_angle_deg, d)
    if actual_corner_radius_mm != window_corner_radius_mm:
        print(
            "  [warn] window_corner_radius_mm adjusted from "
            f"{window_corner_radius_mm:.3f} to {actual_corner_radius_mm:.3f} mm"
        )

    solid = cq.Solid.extrudeLinear(face, cq.Vector(0, 0, cut_depth))
    solid = solid.translate(cq.Vector(0, 0, -cut_depth / 2))
    for i in range(num_spokes):
        rotated = solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), i * pitch_deg)
        disc = disc.cut(rotated)
    return disc


# ---------------------------------------------------------------------------
# STEP 6 — rim barrel
# ---------------------------------------------------------------------------

def build_rim_barrel(d):
    """Hollow C-channel rim with flanges + bead seats at each axial end.

    Cross-section in (R, Z):

        z = +wheel_width/2          inner -- flange tip
                                      |        |
        z = +(ww/2 - flange_th)       |    flange root, drops to bead seat
                                      |        |
                                      |     bead seat / barrel (R = rim_radius)
                                      |        |
        z = -(ww/2 - flange_th)       |     bead seat / barrel
                                      |        |
        z = -wheel_width/2          inner -- flange tip
    """
    rim_radius_mm = d["rim_radius_mm"]

    R_outer  = rim_radius_mm
    R_flange = rim_radius_mm + rim_flange_mm
    R_inner  = rim_radius_mm - rim_band_width_mm

    z_out = wheel_width_mm / 2
    z_in  = -wheel_width_mm / 2
    f     = flange_axial_thickness_mm

    profile = (
        cq.Workplane("XZ")
        .moveTo(R_inner,  z_out)
        .lineTo(R_flange, z_out)            # top of outboard flange
        .lineTo(R_flange, z_out - f)        # axial face of outboard flange
        .lineTo(R_outer,  z_out - f)        # drop to bead seat
        .lineTo(R_outer,  z_in  + f)        # along the barrel
        .lineTo(R_flange, z_in  + f)        # rise to inboard flange
        .lineTo(R_flange, z_in)             # axial face of inboard flange
        .lineTo(R_inner,  z_in)             # over to inner corner
        .close()                            # back along the inner wall to start
    )

    return profile.revolve(axisStart=(0, 0), axisEnd=(0, 1))


# ---------------------------------------------------------------------------
# STEP 7 — assembly
# ---------------------------------------------------------------------------

def _safe_union(a, b, label):
    try:
        return a.union(b)
    except Exception as exc:
        a_bb = a.val().BoundingBox()
        b_bb = b.val().BoundingBox()
        print(f"  bounding box A: {a_bb.xmin:.2f},{a_bb.ymin:.2f},{a_bb.zmin:.2f} -> "
              f"{a_bb.xmax:.2f},{a_bb.ymax:.2f},{a_bb.zmax:.2f}")
        print(f"  bounding box B: {b_bb.xmin:.2f},{b_bb.ymin:.2f},{b_bb.zmin:.2f} -> "
              f"{b_bb.xmax:.2f},{b_bb.ymax:.2f},{b_bb.zmax:.2f}")
        raise ValueError(f"union failed at step '{label}': {exc}") from exc


def build_assembly(tire, disc_shifted, barrel_shifted):
    wheel = _safe_union(disc_shifted, barrel_shifted, "disc + barrel")
    assembly = _safe_union(wheel, tire, "wheel + tire")
    return assembly


# ---------------------------------------------------------------------------
# STEP 8C/D — PyVista rendering (runs in a separate process; see render_views)
# ---------------------------------------------------------------------------

def render_views(stl_path: str, meta_lines, output_dir=None, preview_name="tire_preview.png",
                 keep_views=False):
    """Run the PyVista renderer in a clean subprocess.

    CadQuery's OCP backend leaves OpenGL / VTK state initialised in this
    process which makes subsequent off-screen PyVista renders draw solid
    black frames on llvmpipe / OSMesa. Spawning a fresh Python process
    (that never imports CadQuery) avoids the conflict.
    """
    import json
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    renderer = os.path.join(here, "_render_views.py")
    output_dir = os.path.abspath(output_dir or os.path.dirname(os.path.abspath(stl_path)))
    payload = json.dumps({"stl_path": os.path.abspath(stl_path),
                          "meta_lines": meta_lines,
                          "output_dir": output_dir,
                          "preview_name": preview_name,
                          "keep_views": keep_views})
    result = subprocess.run(
        [sys.executable, renderer, payload],
        check=False,
        cwd=output_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(f"render subprocess failed with code {result.returncode}")


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(output_dir=".", stem="tire_assembly", preview_name="tire_preview.png",
                 keep_views=False, render=True):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stl_path = os.path.join(output_dir, f"{stem}.stl")
    preview_path = os.path.join(output_dir, preview_name) if render else None

    print("=" * 60)
    print("STEP 1: derived dimensions")
    d = compute_derived()

    print("\nSTEP 2: tire")
    tire = build_tire(d)
    print(f"  tire bounding box: {tire.val().BoundingBox().DiagonalLength:.2f} mm diagonal")

    print("\nSTEP 3: wheel disc (pre-cut)")
    disc = build_wheel_disc(d)
    print(f"  disc bounding box: {disc.val().BoundingBox().DiagonalLength:.2f} mm diagonal")

    print("\nSTEP 4: spoke window cutouts")
    disc_cut = cut_spoke_windows(disc, d)
    print(f"  disc-cut bounding box: {disc_cut.val().BoundingBox().DiagonalLength:.2f} mm diagonal")

    print("\nSTEP 5: wheel offset shift")
    # The disc is now built directly in final coords (mounting face at
    # z = +wheel_offset_mm), so the spec's post-cut translation is a
    # no-op. The shift value is reported here so the parameter is still
    # surfaced for the user.
    axial_shift = wheel_offset_mm - (wheel_width_mm / 2)
    print(f"  axial_shift = {axial_shift:.3f} mm (already baked into disc geometry)")
    disc_shifted = disc_cut

    print("\nSTEP 6: rim barrel")
    barrel = build_rim_barrel(d)   # centred on Z = 0, mates with the tire bead

    print("\nSTEP 7: assembly")
    assembly = build_assembly(tire, disc_shifted, barrel)

    print("\nSTEP 8A: export STL")
    exporters.export(
        assembly,
        stl_path,
        exporters.ExportTypes.STL,
        tolerance=0.1,
        angularTolerance=0.1,
    )
    print(f"  wrote {stl_path}")

    print("\nSTEP 8B: trimesh validation")
    import trimesh
    mesh = trimesh.load(stl_path)
    is_wt = bool(mesh.is_watertight)
    volume_cm3 = float(round(mesh.volume / 1000.0, 1))
    face_count = len(mesh.faces)
    repaired = False
    print(f"  Watertight:   {is_wt}")
    print(f"  Volume cm:   {volume_cm3}")
    print(f"  Face count:   {face_count}")
    print(f"  Bounds (mm):  {mesh.bounds}")
    if not is_wt:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        mesh.export(stl_path)
        is_wt = bool(mesh.is_watertight)
        volume_cm3 = float(round(mesh.volume / 1000.0, 1))
        repaired = True
        print("  Repaired and re-exported.")
        print(f"  Watertight (post-repair): {is_wt}")

    if render:
        print("\nSTEP 8C/D: rendering views")
    meta_lines = [
        f"section_width_mm   = {section_width_mm}",
        f"aspect_ratio       = {aspect_ratio}",
        f"rim_diameter_in    = {rim_diameter_in}",
        f"num_spokes         = {num_spokes}",
        f"wheel_offset_mm    = {wheel_offset_mm}",
        f"dish_depth_mm      = {dish_depth_mm}",
        f"sidewall_bulge_mm  = {sidewall_bulge_mm}",
        f"dish_profile       = {dish_profile}",
        "",
        f"Watertight: {is_wt}",
        f"Volume: {volume_cm3} cm^3",
    ]
    if render:
        render_views(
            stl_path,
            meta_lines,
            output_dir=output_dir,
            preview_name=preview_name,
            keep_views=keep_views,
        )

    print("""
=== VISUAL REVIEW CHECKLIST ===
outboard_face  : spoke count correct? window shapes symmetric?
                 dish curvature visible at rim vs hub?
side_profile   : sidewall bulge present? tread crown visible?
                 tire width matches section_width_mm?
isometric      : overall proportions look correct?
                 no obvious geometry artifacts?
top_down       : spokes evenly spaced? hub bore visible?
cross_section  : sidewall Bezier curve smooth?
                 rim flange geometry correct? dish depth visible?
===============================
""")
    return {
        "assembly": assembly,
        "stl_path": stl_path,
        "preview_path": preview_path,
        "parameters": get_current_parameters(),
        "derived": d,
        "watertight": is_wt,
        "repair_applied": repaired,
        "volume_cm3": volume_cm3,
        "face_count": face_count,
        "bounds_mm": mesh.bounds.tolist(),
    }


def generate_case(spec_overrides=None, output_dir=".", stem="tire_assembly",
                  preview_name="tire_preview.png", keep_views=False, render=True):
    with temporary_parameters(spec_overrides) as (spec, notes):
        if notes:
            print("[parameter normalization]")
            for note in notes:
                print(f"  - {note}")
        result = run_pipeline(
            output_dir=output_dir,
            stem=stem,
            preview_name=preview_name,
            keep_views=keep_views,
            render=render,
        )
        result["normalized_parameters"] = spec
        result["normalization_notes"] = notes
        return result


# ---------------------------------------------------------------------------
# CQ-editor preview hook
# ---------------------------------------------------------------------------

if __name__ == "__cq_main__":
    _d = compute_derived()
    _tire = build_tire(_d)
    _disc = build_wheel_disc(_d)
    _disc_cut = cut_spoke_windows(_disc, _d)
    _barrel = build_rim_barrel(_d)
    _assembly = build_assembly(_tire, _disc_cut, _barrel)
    show_object(_tire,     name="tire",            options={"alpha": 0.3})   # noqa: F821
    show_object(_disc,     name="wheel_disc_pre",  options={"alpha": 0.4})   # noqa: F821
    show_object(_disc_cut, name="wheel_disc_post")                            # noqa: F821
    show_object(_assembly, name="assembly")                                   # noqa: F821


if __name__ == "__main__":
    run_pipeline()
