"""ParamUB example — every parameter exposed for tweaking.

Edit any value below and run:

    conda activate paramub
    python makeUB.py

Output goes to outputs/UB1/ as:
    ub.stl                  merged assembly (tessellated)
    ub.step                 merged assembly (BREP; only when export_step=True)
    ub_preview.png          8-panel composite preview with parameter footer
    ub.log                  mirrored stdout/stderr
    ub.json                 full spec dump
    FLOOR.STL  / .STEP                              (only when output_parts=True;
    WHEEL_FL.STL / .STEP, WHEEL_FR.STL / .STEP,      .STEP files require
    WHEEL_RL.STL / .STEP, WHEEL_RR.STL / .STEP       export_step=True)
"""

from paramub import (
    generate,
    UnderbodySpec,
    WheelSpec,
    TireSpec,
    SpokeSpec,
)


# =====================================================================
# TIRE  —  rubber profile only (see paramub/docs/tire_builder.html)
# =====================================================================
tire = TireSpec(
    # cross-section
    section_width_mm   = 245.0,     # "225" in 225/45 R18
    aspect_ratio       = 45.0,      # "45"  (= sidewall_height / section_width × 100)
    rim_diameter_in    = 18.0,      # "18"

    # sidewall shape
    sidewall_bulge_mm  = 8.0,       # max radial bow of the sidewall
    sidewall_bulge_pos = 0.45,      # 0..1, axial position of the bulge max
    shoulder_radius_mm = 30.0,      # depth of C1 tangent rounding into tread

    # tread
    tread_width_mm     = 190.0,
    crown_radius_mm    = 400.0,     # larger = flatter crown
    rim_flange_mm      = 18.0,      # height of flange ledge above rim radius
)


# =====================================================================
# SPOKE / WHEEL  —  rim, disc, spoke windows  (see spoke_builder.html)
# =====================================================================
spoke = SpokeSpec(
    # rim / wheel body
    wheel_width_mm     = 235.0,     # flange-tip to flange-tip
    rim_diameter_in    = 18.0,      # MUST match tire.rim_diameter_in
    rim_flange_mm      = 18.0,
    hub_bore_mm        = 72.0,
    wheel_offset_mm    = 35.0,      # ET; +ve = mounting face outboard

    # outboard "dish" face
    dish_depth_mm      = 15.0,      # 0 = flat
    dish_profile       = "spherical",   # "spherical" or "conical"

    # spoke window cutouts
    num_spokes              = 17,
    spoke_width_hub_mm      = 28.0,
    spoke_width_rim_mm      = 18.0,
    window_inner_gap_mm     = 12.0,
    window_outer_gap_mm     = 15.0,
    window_corner_radius_mm = 10.0,

    # construction
    rim_band_width_mm         = 20.0,
    flange_axial_thickness_mm = 12.0,
    spoke_thickness_mm        = 150.0,
    spoke_outer_crown_mm      = 4.0,
)

front_wheel = WheelSpec(tire=tire, spoke=spoke)
# Set rear_wheel below if you want different front/rear wheels:
rear_wheel = None    # None -> reuse front wheel for all 4 corners


# =====================================================================
# UNDERBODY  —  vehicle plan + ride + diffuser + wheelhouses + alignment
# =====================================================================
spec = UnderbodySpec(
    # ----- vehicle plan -----------------------------------------------
    wheelbase_mm       = 2700.0,
    front_overhang_mm  = 850.0,
    rear_overhang_mm   = 800.0,
    track_front_mm     = 1550.0,
    track_rear_mm      = 1540.0,

    # ----- ride / floor / diffuser ------------------------------------
    ride_height_mm     = 100.0,     # ground -> floor underside
    floor_width_mm     = 1800.0,    # plan width of the rectangular floor
    floor_thickness_mm = 0.0,       # 0 -> zero-thickness surface (CFD)
                                    # >0 -> solid slab of that thickness
    floor_angle_deg    = 0.0,       # rake about the front edge

    diffuser_start_x_mm = None,     # None -> rear axle
    diffuser_angle_deg  = 7.0,
    diffuser_radius_mm  = 500.0,    # tangent fillet between floor + ramp

    # ----- wheelhouses ------------------------------------------------
    wheel_house_axial_clearance_mm   = 30.0,    # tire OD -> arch ID
    wheel_house_thickness_mm         = 6.0,     # solid mode only
    wheel_house_lateral_clearance_mm = 35.0,    # base Y gap per side
    front_steering_clearance_mm      = 150.0,   # extra Y on front arches
    rear_steering_clearance_mm       = 0.0,
    front_wheel_house_fillet_mm      = 100.0,   # inboard cap fillet R
    rear_wheel_house_fillet_mm       = 30.0,

    # ----- wheel alignment --------------------------------------------
    camber_front_deg   = -1.5,      # -ve = top of wheel inward
    camber_rear_deg    = -1.0,
    toe_front_deg      = 0.10,      # +ve = toe-in
    toe_rear_deg       = 0.0,

    # ----- wheels (composed) ------------------------------------------
    wheel      = front_wheel,
    rear_wheel = rear_wheel,
)


# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    generate(
        spec,
        output_dir       = "outputs/UB1",
        stem             = "ub",
        output_mode      = "all",          # "stl" for just the STL
        output_parts     = True,           # also emit FLOOR.STL + WHEEL_*.STL
        export_step      = True,          # True -> also emit <stem>.step
                                           # (+ FLOOR.STEP, WHEEL_*.STEP when
                                           #  combined with output_parts=True)
        stl_tolerance_mm = 0.1,            # smaller = finer mesh, larger STL
    )
