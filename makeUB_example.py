"""Example / smoke-test driver for the multisection splitter and diffuser.

Run with one of these modes (default: ``diffuser_n3``)::

    python makeUB_example.py diffuser_n2  | diffuser_n3
                             splitter_n2 | splitter_n3
                             both_n3 | both_mixed | varying_x

  diffuser_n2 / diffuser_n3 — diffuser only, 1 or 2 user sections
  splitter_n2 / splitter_n3 — splitter only, 1 or 2 user sections
  both_n3                   — splitter + diffuser, 2 user sections each
  both_mixed                — splitter N=3, diffuser N=2 (independent N test)
  varying_x                 — splitter + diffuser with varying X at BOTH
                              the kicklines and the front/rear endpoints
"""
import sys

from paramub import (
    DiffuserSection,
    SplitterSection,
    generate,
    UnderbodySpec,
    WheelSpec,
    TireSpec,
    SpokeSpec,
)


tire = TireSpec(
    section_width_mm=245.0, aspect_ratio=45.0, rim_diameter_in=18.0,
    sidewall_bulge_mm=8.0, sidewall_bulge_pos=0.45, shoulder_radius_mm=30.0,
    tread_width_mm=190.0, crown_radius_mm=400.0, rim_flange_mm=18.0,
)
spoke = SpokeSpec(
    wheel_width_mm=235.0, rim_diameter_in=18.0, rim_flange_mm=18.0,
    hub_bore_mm=72.0, wheel_offset_mm=35.0,
    dish_depth_mm=15.0, dish_profile="spherical",
    num_spokes=17, spoke_width_hub_mm=28.0, spoke_width_rim_mm=18.0,
    window_inner_gap_mm=12.0, window_outer_gap_mm=15.0,
    window_corner_radius_mm=10.0,
    rim_band_width_mm=20.0, flange_axial_thickness_mm=12.0,
    spoke_thickness_mm=150.0, spoke_outer_crown_mm=4.0,
)
wheel = WheelSpec(tire=tire, spoke=spoke)


# Reusable section definitions.
DIFF_OUT = DiffuserSection(
    y_mm=900.0, kick_x_mm=-1200.0, rear_x_mm=-2150.0,
    rear_z_mm=300.0, rear_angle_deg=12.0,
    start_strength=0.26, end_strength=0.26,
)
DIFF_CTR = DiffuserSection(
    y_mm=0.0, kick_x_mm=-1350.0, rear_x_mm=-2150.0,
    rear_z_mm=200.0, rear_angle_deg=8.0,
    start_strength=0.37, end_strength=0.37,
)
SPLIT_OUT = SplitterSection(
    y_mm=900.0, kick_x_mm=1400.0, front_x_mm=2200.0,
    front_z_mm=180.0, front_angle_deg=8.0,
    start_strength=0.25, end_strength=0.25,
)
SPLIT_CTR = SplitterSection(
    y_mm=0.0, kick_x_mm=1450.0, front_x_mm=2200.0,
    front_z_mm=150.0, front_angle_deg=5.0,
    start_strength=0.30, end_strength=0.30,
)
SPLIT_MID = SplitterSection(
    y_mm=450.0, kick_x_mm=1430.0, front_x_mm=2200.0,
    front_z_mm=160.0, front_angle_deg=6.0,
    start_strength=0.28, end_strength=0.28,
)

# ---- varying_x set: BOTH kicklines and end points X positions vary
#      across sections, to exercise the independent X-control. The
#      flat floor's plan boundary is no longer a simple rectangle.

# Splitter: centerline kicks later (further forward of front axle) and
# extends past floor_x_max; outboard kicks earlier (less forward of
# axle) and ends inside floor_x_max with a steeper lift.
SPLIT_CTR_VAR = SplitterSection(
    y_mm=0.0,
    kick_x_mm=1500.0,    # 150 mm forward of front axle (1350)
    front_x_mm=2300.0,   # extends 100 mm past floor_x_max (2200)
    front_z_mm=130.0,
    front_angle_deg=3.0,
    start_strength=0.30, end_strength=0.30,
)
SPLIT_OUT_VAR = SplitterSection(
    y_mm=900.0,
    kick_x_mm=1300.0,    # 50 mm rearward of front axle
    front_x_mm=2150.0,   # ends 50 mm inside floor_x_max
    front_z_mm=200.0,
    front_angle_deg=10.0,
    start_strength=0.25, end_strength=0.25,
)

# Diffuser: centerline kicks rearward of rear axle, ends past
# floor_x_min; outboard kicks much further forward and ends earlier
# (so the diffuser is shorter and steeper outboard).
DIFF_CTR_VAR = DiffuserSection(
    y_mm=0.0,
    kick_x_mm=-1400.0,   # 50 mm rearward of rear axle (-1350)
    rear_x_mm=-2250.0,   # extends 100 mm past floor_x_min (-2150)
    rear_z_mm=200.0,
    rear_angle_deg=8.0,
    start_strength=0.37, end_strength=0.37,
)
DIFF_OUT_VAR = DiffuserSection(
    y_mm=900.0,
    kick_x_mm=-1100.0,   # 250 mm forward of rear axle (early kick outboard)
    rear_x_mm=-2050.0,   # ends 100 mm inside floor_x_min
    rear_z_mm=350.0,     # higher exit outboard
    rear_angle_deg=15.0,
    start_strength=0.22, end_strength=0.28,
)


mode = sys.argv[1] if len(sys.argv) > 1 else "diffuser_n3"

splitter_sections = None
diffuser_sections = None

if mode == "diffuser_n2":
    diffuser_sections = [DIFF_OUT]
elif mode == "diffuser_n3":
    diffuser_sections = [DIFF_CTR, DIFF_OUT]
elif mode == "splitter_n2":
    splitter_sections = [SPLIT_OUT]
elif mode == "splitter_n3":
    splitter_sections = [SPLIT_CTR, SPLIT_OUT]
elif mode == "both_n3":
    splitter_sections = [SPLIT_CTR, SPLIT_OUT]
    diffuser_sections = [DIFF_CTR, DIFF_OUT]
elif mode == "both_mixed":
    splitter_sections = [SPLIT_CTR, SPLIT_MID, SPLIT_OUT]   # N=3
    diffuser_sections = [DIFF_OUT]                          # N=2 (mirrored)
elif mode == "varying_x":
    splitter_sections = [SPLIT_CTR_VAR, SPLIT_OUT_VAR]
    diffuser_sections = [DIFF_CTR_VAR, DIFF_OUT_VAR]
else:
    raise SystemExit(f"unknown mode: {mode}")

out_dir = f"outputs/UB1_{mode}"


spec = UnderbodySpec(
    wheelbase_mm=2700.0,
    front_overhang_mm=850.0,
    rear_overhang_mm=800.0,
    track_front_mm=1550.0,
    track_rear_mm=1540.0,
    ride_height_mm=100.0,
    floor_width_mm=1800.0,
    floor_angle_deg=0.0,
    diffuser_sections=diffuser_sections,
    splitter_sections=splitter_sections,
    wheel_house_axial_clearance_mm=30.0,
    wheel_house_thickness_mm=6.0,
    wheel_house_lateral_clearance_mm=35.0,
    front_steering_clearance_mm=150.0,
    rear_steering_clearance_mm=0.0,
    front_wheel_house_fillet_mm=100.0,
    rear_wheel_house_fillet_mm=100.0,
    camber_front_deg=-1.5,
    camber_rear_deg=-1.0,
    toe_front_deg=0.10,
    toe_rear_deg=0.0,
    wheel=wheel,
    rear_wheel=None,
)


if __name__ == "__main__":
    generate(
        spec,
        output_dir=out_dir,
        stem="ub",
        output_mode="all",
        output_parts=True,
        export_step=True,
        stl_tolerance_mm=0.1,
        half_only=True,
    )
