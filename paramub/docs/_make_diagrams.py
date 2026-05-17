"""Generate matplotlib PNG diagrams that the HTML user guides reference.

Outputs one PNG per builder under ``paramub/docs/diagrams/``:

    tire_section.png        — radial cross-section + dimension callouts
    spoke_section.png       — disc cross-section + cap/dish callouts
    spoke_window.png        — plan view of one spoke window
    wheelhouse_section.png  — XZ section through the arch with sidewalls + dome
    wheelhouse_plan.png     — XY footprint with fillet + outboard trim
    floor_section.png       — XZ underside contour: flat + fillet + ramp
    floor_plan.png          — XY plan with wheelhouse cut-outs

Re-run after changing any default geometry in the builder modules:

    python paramub/docs/_make_diagrams.py
"""

from __future__ import annotations

import sys
from math import cos, radians, sin, sqrt, tan
from pathlib import Path

# Make `paramub` importable when this script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from paramub.floor_builder import FloorSpec
from paramub.spoke_builder import SpokeSpec
from paramub.tire_builder import TireSpec
from paramub.wheelhouse_builder import WheelhouseSpec

OUT = Path(__file__).resolve().parent / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _annot(ax, p_a, p_b, label, *, offset=(0, 0), color="#1f4ed8", lw=1.5):
    """Draw a double-arrow dimension line and a label."""
    ax.annotate(
        "", xy=p_b, xytext=p_a,
        arrowprops=dict(arrowstyle="<->", color=color, lw=lw),
    )
    mx = (p_a[0] + p_b[0]) / 2 + offset[0]
    my = (p_a[1] + p_b[1]) / 2 + offset[1]
    ax.text(mx, my, label, color=color, ha="center", va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec=color, lw=0.6))


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT.parent)}")


# ---------------------------------------------------------------------------
# tire_builder: radial cross-section
# ---------------------------------------------------------------------------

def make_tire_section():
    s = TireSpec()
    rim_r = s.rim_radius_mm
    sh = s.sidewall_height_mm
    bw = s.section_width_mm
    tw = s.tread_width_mm

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(0, color="#555", lw=0.6, ls="--")

    rim_band = mpatches.Rectangle(
        (-bw / 2, rim_r - 6), bw, 12, color="#cfd5dc",
        ec="#7d8492", lw=0.8)
    ax.add_patch(rim_band)
    ax.text(0, rim_r - 16, "rim", ha="center", color="#5b6471", fontsize=9)

    pts = _tire_profile_points(s)
    xs, zs = zip(*pts)
    xs_full = [-x for x in reversed(xs)] + list(xs)
    zs_full = [z for z in reversed(zs)] + list(zs)
    ax.fill(xs_full, zs_full, color="#1f2937", alpha=0.18,
            ec="#1f2937", lw=1.2)
    ax.plot(xs_full, zs_full, color="#1f2937", lw=1.5)

    # axial dimensions across the bottom
    _annot(ax, (-bw / 2, rim_r - 55), (bw / 2, rim_r - 55),
           f"section_width_mm = {bw:.0f}", offset=(0, -16))
    # tread_width sits on top inside the tire (clear of edge)
    _annot(ax, (-tw / 2, rim_r + sh + 22),
           (tw / 2, rim_r + sh + 22),
           f"tread_width_mm = {tw:.0f}", offset=(0, 16))

    # radial dimensions on the left
    _annot(ax, (-bw / 2 - 50, 0), (-bw / 2 - 50, rim_r),
           f"rim_radius = {rim_r:.0f}\n(rim_diameter_in × 25.4 / 2)",
           offset=(-95, 0))
    _annot(ax, (-bw / 2 - 200, 0), (-bw / 2 - 200, rim_r + sh),
           f"tire_OD/2 = {rim_r + sh:.0f}",
           offset=(-60, 0))

    # radial dimensions on the right, stacked
    _annot(ax, (bw / 2 + 30, rim_r), (bw / 2 + 30, rim_r + sh),
           f"sidewall_height = {sh:.0f}\n(= section_width × aspect/100)",
           offset=(115, 0))
    flange_top = rim_r + s.rim_flange_mm
    _annot(ax, (bw / 2 + 8, rim_r), (bw / 2 + 8, flange_top),
           f"rim_flange = {s.rim_flange_mm:.0f}",
           offset=(45, -8))

    # crown radius (centered, well above tread)
    ax.text(0, rim_r + sh + 65,
            f"crown_radius = {s.crown_radius_mm:.0f}",
            ha="center", color="#1f4ed8", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#1f4ed8", lw=0.6))

    # shoulder rounding — pulled clear of sidewall_height callout
    ax.annotate("shoulder_radius =\nC1 tangent rounding\ninto tread",
                xy=(tw / 2 + 4, rim_r + sh - 4),
                xytext=(tw / 2 - 90, rim_r + sh + 70),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))

    # sidewall_bulge — point at the bulge area, label below the side callout
    ax.annotate(f"sidewall_bulge = {s.sidewall_bulge_mm:.0f}\n(max outward bow)",
                xy=(bw / 2 - 4, rim_r + sh * 0.55),
                xytext=(bw / 2 + 80, rim_r * 0.5),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))

    ax.set_xlim(-bw / 2 - 330, bw / 2 + 290)
    ax.set_ylim(-80, rim_r + sh + 130)
    ax.set_aspect("equal")
    ax.set_xlabel("axial (mm)")
    ax.set_ylabel("radial (mm)")
    ax.set_title("Tire — radial cross-section (one half mirrored)")
    ax.grid(alpha=0.2)
    _save(fig, "tire_section.png")


def _tire_profile_points(s: TireSpec):
    """Recreate the positive-Z half of the tire profile points (axial=x, radial=z)."""
    rim_r = s.rim_radius_mm
    sh = s.sidewall_height_mm
    z_bead = 215.0 / 2  # default wheel_width / 2 just for the diagram
    z_tread_edge = s.tread_width_mm / 2
    R_bead = rim_r
    R_flange = rim_r + s.rim_flange_mm
    R_tread_c = rim_r + sh
    cc_R = R_tread_c - s.crown_radius_mm
    R_te = cc_R + sqrt(s.crown_radius_mm ** 2 - z_tread_edge ** 2)

    tan_R = z_tread_edge / s.crown_radius_mm
    tan_Z = -sqrt(s.crown_radius_mm ** 2 - z_tread_edge ** 2) / s.crown_radius_mm
    P0 = (R_flange, z_bead)
    P3 = (R_te, z_tread_edge)
    P2 = (R_te - s.shoulder_radius_mm * tan_R,
          z_tread_edge - s.shoulder_radius_mm * tan_Z)
    P1 = (R_flange,
          z_bead + s.sidewall_bulge_mm * (0.5 + 0.5 * s.sidewall_bulge_pos))

    def bez(t):
        u = 1 - t
        return (
            u**3 * P0[0] + 3*u*u*t*P1[0] + 3*u*t*t*P2[0] + t**3 * P3[0],
            u**3 * P0[1] + 3*u*u*t*P1[1] + 3*u*t*t*P2[1] + t**3 * P3[1],
        )
    sidewall = [bez(k / 32) for k in range(33)]
    # tread arc: parametric over angle from z_tread_edge to 0
    n_arc = 24
    arc = []
    for k in range(n_arc + 1):
        z = z_tread_edge * (1 - k / n_arc)
        R = cc_R + sqrt(s.crown_radius_mm ** 2 - z**2)
        arc.append((R, z))

    # combine — radial(R) is plotted on Y, axial(z) on X
    pts = [(z_bead, R_bead)]
    pts.append((z_bead, R_flange))
    for R, z in sidewall:
        pts.append((z, R))
    for R, z in arc:
        pts.append((z, R))
    return pts


# ---------------------------------------------------------------------------
# spoke_builder: disc cross-section + window plan
# ---------------------------------------------------------------------------

def make_spoke_section():
    s = SpokeSpec()
    rim_r = s.rim_radius_mm
    R_hub = s.hub_bore_mm / 2
    R_hub_o = s.hub_ring_od_mm
    R_disc_o = rim_r - s.rim_band_width_mm + 0.5
    z_mount = s.wheel_offset_mm
    z_att_out = s.wheel_width_mm / 2
    z_att_in = z_att_out - s.spoke_thickness_mm
    z_hub_out = z_att_out - s.dish_depth_mm

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axhline(0, color="#555", lw=0.6, ls="--")  # spin axis
    ax.axvline(0, color="#555", lw=0.6, ls=":")  # midplane

    rim_x = [-z_att_out, z_att_out, z_att_out, -z_att_out, -z_att_out]
    rim_y = [rim_r - s.rim_band_width_mm, rim_r - s.rim_band_width_mm,
             rim_r + s.rim_flange_mm, rim_r + s.rim_flange_mm,
             rim_r - s.rim_band_width_mm]
    ax.plot(rim_x, rim_y, color="#9aa3b1", lw=1.2)
    ax.fill(rim_x, rim_y, color="#cfd5dc", alpha=0.5)

    pts = [
        (z_mount, R_hub),
        (z_mount, R_hub_o),
        (z_att_in, R_disc_o),
        (z_att_out, R_disc_o),
        (z_hub_out, R_hub),
        (z_mount, R_hub),
    ]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#1f2937", lw=1.6)
    ax.fill(xs, ys, color="#1f2937", alpha=0.18)
    ys_m = [-y for y in ys]
    ax.plot(xs, ys_m, color="#1f2937", lw=1.6)
    ax.fill(xs, ys_m, color="#1f2937", alpha=0.18)

    # Axial extents along the top (well above rim flange)
    top_y = rim_r + s.rim_flange_mm + 60
    _annot(ax, (-z_att_out, top_y), (z_att_out, top_y),
           f"wheel_width_mm = {s.wheel_width_mm:.0f}", offset=(0, 16))
    # spoke_thickness lives BELOW the disc to avoid hub callout zone
    mid_y = -(R_hub + R_hub_o) / 2 - 20
    _annot(ax, (z_att_in, mid_y), (z_att_out, mid_y),
           f"spoke_thickness = {s.spoke_thickness_mm:.0f}", offset=(0, -16))

    # Left-side annotations (hub bore + hub_ring_od) using leader lines
    ax.annotate(f"hub_bore = {s.hub_bore_mm:.0f}",
                xy=(z_mount, R_hub),
                xytext=(z_mount - 170, R_hub + 6),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    ax.annotate(f"hub_ring_od = {s.hub_ring_od_mm:.0f}\n(bore/2 + 30)",
                xy=(z_mount, R_hub_o),
                xytext=(z_mount - 200, R_hub_o + 40),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    # wheel_offset on the negative-radial side (clear space)
    ax.annotate(f"wheel_offset (ET) = {s.wheel_offset_mm:.0f}\n"
                "(mounting face from midplane)",
                xy=(z_mount, -R_hub - 8),
                xytext=(z_mount + 60, -R_hub - 110),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    ax.annotate(f"dish_depth = {s.dish_depth_mm:.0f}",
                xy=(z_hub_out, R_hub + 8),
                xytext=(z_hub_out + 110, R_hub - 90),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    ax.annotate(f"rim_band_width = {s.rim_band_width_mm:.0f}",
                xy=(z_att_in, R_disc_o),
                xytext=(z_att_in - 180, R_disc_o + 35),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    ax.annotate(f"spoke_outer_crown = {s.spoke_outer_crown_mm:.0f}",
                xy=((z_att_in + z_att_out) / 2, R_disc_o + 0.5),
                xytext=(z_att_out + 60, R_disc_o + 50),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    ax.annotate(f"flange_axial_thickness = {s.flange_axial_thickness_mm:.0f}",
                xy=(z_att_out, rim_r + s.rim_flange_mm - 2),
                xytext=(z_att_out + 60, rim_r + s.rim_flange_mm + 25),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))

    ax.set_aspect("equal")
    ax.set_xlim(-z_att_out - 240, z_att_out + 280)
    ax.set_ylim(-rim_r - 60, rim_r + s.rim_flange_mm + 140)
    ax.set_xlabel("axial Z (mm) — outboard = +")
    ax.set_ylabel("radial R (mm)")
    ax.set_title("Spoke / wheel-disc — axial cross-section")
    ax.grid(alpha=0.2)
    _save(fig, "spoke_section.png")


def make_spoke_window():
    s = SpokeSpec()
    inner_r = s.spoke_inner_r_mm
    outer_r = s.spoke_outer_r_mm
    pitch = 360.0 / s.num_spokes

    fig, ax = plt.subplots(figsize=(8, 8))
    # hub ring
    ring = mpatches.Wedge((0, 0), s.hub_ring_od_mm, 0, 360, width=8,
                          color="#9aa3b1", alpha=0.8)
    ax.add_patch(ring)
    # rim ring
    rim_ring = mpatches.Wedge((0, 0), s.rim_radius_mm, 0, 360,
                              width=s.rim_band_width_mm,
                              color="#cfd5dc", alpha=0.8)
    ax.add_patch(rim_ring)

    # Draw each window
    from math import atan, degrees
    pitch_deg = pitch
    spoke_inner_deg = degrees(2 * atan(s.spoke_width_hub_mm / 2 / inner_r))
    spoke_outer_deg = degrees(2 * atan(s.spoke_width_rim_mm / 2 / outer_r))
    win_in_half = (pitch_deg - spoke_inner_deg) / 2
    win_out_half = (pitch_deg - spoke_outer_deg) / 2

    for i in range(s.num_spokes):
        center_deg = (i + 0.5) * pitch_deg
        a_in_lo = radians(center_deg - win_in_half)
        a_in_hi = radians(center_deg + win_in_half)
        a_out_lo = radians(center_deg - win_out_half)
        a_out_hi = radians(center_deg + win_out_half)
        # window polygon
        n_seg = 12
        in_arc = [(inner_r * cos(a_in_lo + (a_in_hi - a_in_lo) * k / n_seg),
                   inner_r * sin(a_in_lo + (a_in_hi - a_in_lo) * k / n_seg))
                  for k in range(n_seg + 1)]
        out_arc = [(outer_r * cos(a_out_hi + (a_out_lo - a_out_hi) * k / n_seg),
                    outer_r * sin(a_out_hi + (a_out_lo - a_out_hi) * k / n_seg))
                   for k in range(n_seg + 1)]
        poly = in_arc + out_arc
        ax.fill([p[0] for p in poly], [p[1] for p in poly],
                color="white", ec="#1f2937", lw=1.2)

    # Spread callouts at clear angles around the wheel
    R = s.rim_radius_mm + 40
    cdict = dict(fontsize=9, color="#1f4ed8",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec="#1f4ed8", lw=0.6),
                 arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    ax.annotate(f"spoke_width_hub = {s.spoke_width_hub_mm:.0f}",
                xy=(inner_r * cos(radians(90)), inner_r * sin(radians(90))),
                xytext=(-R - 40, R + 40), **cdict)
    ax.annotate(f"spoke_width_rim = {s.spoke_width_rim_mm:.0f}",
                xy=(outer_r * cos(radians(90)), outer_r * sin(radians(90))),
                xytext=(R - 40, R + 60), **cdict)
    ax.annotate(f"window_inner_gap = {s.window_inner_gap_mm:.0f}\n"
                "(hub_ring_od → window inner arc)",
                xy=(inner_r, 0), xytext=(R + 30, -40), **cdict)
    ax.annotate(f"window_outer_gap = {s.window_outer_gap_mm:.0f}\n"
                "(rim ID → window outer arc)",
                xy=(outer_r, 0), xytext=(R + 30, 40), **cdict)
    ax.annotate(f"window_corner_radius = {s.window_corner_radius_mm:.0f}",
                xy=(outer_r * cos(radians(180 - 4)),
                    outer_r * sin(radians(180 - 4))),
                xytext=(-R - 130, R - 20), **cdict)
    ax.text(0, -s.rim_radius_mm - 70,
            f"num_spokes = {s.num_spokes}  (pitch = {pitch_deg:.1f}°)",
            ha="center", fontsize=11, color="#1f2937",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#1f2937", lw=0.8))

    ax.set_xlim(-R - 160, R + 160)
    ax.set_ylim(-R - 120, R + 110)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Spoke windows — plan view (looking along spin axis)")
    ax.grid(alpha=0.2)
    _save(fig, "spoke_window.png")


# ---------------------------------------------------------------------------
# wheelhouse_builder
# ---------------------------------------------------------------------------

def make_wheelhouse_section():
    tire_r = 330.0
    ride_h = 100.0
    s = WheelhouseSpec(
        axle_x=0.0, y_track=0.0, side="right",
        tire_radius_mm=tire_r, tire_width_mm=225.0,
        ride_height_mm=ride_h, floor_edge_y=900.0,
    )
    arch_r = s.arch_radius_mm

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_between([-arch_r * 2.0, arch_r * 2.0], -60, 0,
                     color="#e5e7eb", lw=0)
    ax.axhline(0, color="#777", lw=0.8)
    ax.axhline(ride_h, color="#1f2937", lw=1.5, ls="--",
               label="floor underside (z = ride_height)")

    # Walk the polygon counter-clockwise: bottom-left -> up the left sidewall
    # -> dome (left to right) -> down the right sidewall -> back across floor.
    thetas = np.linspace(np.pi, 0, 100)   # from -arch_r side TO +arch_r side
    dome_x = arch_r * np.cos(thetas)
    dome_z = tire_r + arch_r * np.sin(thetas)
    arch_x = [-arch_r, -arch_r] + list(dome_x) + [arch_r, -arch_r]
    arch_z = [ride_h, tire_r] + list(dome_z) + [ride_h, ride_h]
    ax.fill(arch_x, arch_z, color="#b0b8c1", alpha=0.5,
            ec="#1f2937", lw=1.5, label="wheelhouse solid")

    # tire outline
    ct = plt.Circle((0, tire_r), tire_r, color="#1f2937", alpha=0.18,
                    ec="#1f2937", lw=1.3, ls="--")
    ax.add_patch(ct)
    ax.text(0, tire_r, "tire", ha="center", va="center", fontsize=9,
            color="#1f2937")

    # callouts (placed away from the legend, no overlap)
    _annot(ax, (-arch_r - 60, 0), (-arch_r - 60, ride_h),
           f"ride_height = {ride_h:.0f}", offset=(-80, 0))
    _annot(ax, (arch_r + 35, ride_h), (arch_r + 35, tire_r),
           "sidewall section\n(ride_h → tire_r)", offset=(95, 0))
    _annot(ax, (-arch_r, tire_r + arch_r + 35),
           (arch_r, tire_r + arch_r + 35),
           f"2 × arch_radius = 2 × (tire_r + axial_clearance)\n"
           f"          = 2 × {arch_r:.0f}", offset=(0, 22))
    # axial_clearance — point to the gap between dome and tire on the side
    gap_x = arch_r * cos(radians(60))
    gap_z = tire_r + arch_r * sin(radians(60))
    ax.annotate(f"axial_clearance = {s.axial_clearance_mm:.0f}\n"
                "(tire OD → arch ID)",
                xy=(gap_x, gap_z),
                xytext=(-arch_r - 200, tire_r + arch_r * 0.6),
                fontsize=8, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))

    ax.set_xlim(-arch_r * 2.0, arch_r * 2.0)
    ax.set_ylim(-60, tire_r + arch_r + 120)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm) — forward")
    ax.set_ylabel("z (mm) — up")
    ax.set_title("Wheelhouse — XZ cross-section (sidewalls + dome over tire)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, "wheelhouse_section.png")


def make_wheelhouse_plan():
    """XY plan view: arch footprint with fillet edge + outboard trim."""
    tire_r = 330.0
    tire_w = 225.0
    lat = 35.0 + 150.0     # base + front steering clearance
    fillet = 100.0
    floor_edge_y = 900.0
    y_track = 775.0
    axle_x = 1350.0
    s = WheelhouseSpec(
        axle_x=axle_x, y_track=y_track, side="right",
        tire_radius_mm=tire_r, tire_width_mm=tire_w,
        lateral_clearance_mm=lat,
        fillet_mm=fillet,
        ride_height_mm=100.0,
        floor_edge_y=floor_edge_y,
    )
    arch_r = s.arch_radius_mm
    arch_L = s.arch_length_mm

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhline(floor_edge_y, color="#1f2937", lw=1.5,
               label=f"floor lateral edge (y = {floor_edge_y:.0f})")

    # Plan-view footprint: rectangle of the wheelhouse, with the INBOARD
    # cap edge rounded by the 3D fillet's plan projection (rounded corners
    # along the inboard cap edge at y = y_track - arch_L/2).
    x0 = axle_x - arch_r
    x1 = axle_x + arch_r
    y_inb = y_track - arch_L / 2   # inboard cap edge (right side: smaller Y)
    y_out = min(y_track + arch_L / 2, floor_edge_y)
    # Build a polygon with two quarter-circles at the inboard corners
    poly_pts = []
    poly_pts.append((x0, y_out))
    poly_pts.append((x0, y_inb + fillet))
    nq = 20
    for k in range(nq + 1):
        th = np.pi + np.pi / 2 * k / nq    # 180° -> 270°  (sweep into corner)
        poly_pts.append((x0 + fillet + fillet * np.cos(th),
                         y_inb + fillet + fillet * np.sin(th)))
    poly_pts.append((x1 - fillet, y_inb))
    for k in range(nq + 1):
        th = -np.pi / 2 + np.pi / 2 * k / nq   # 270° -> 360°
        poly_pts.append((x1 - fillet + fillet * np.cos(th),
                         y_inb + fillet + fillet * np.sin(th)))
    poly_pts.append((x1, y_out))
    poly = mpatches.Polygon(poly_pts, closed=True,
                             color="#b0b8c1", alpha=0.45,
                             ec="#1f2937", lw=1.5,
                             label="wheelhouse footprint")
    ax.add_patch(poly)

    # tire footprint
    tire_rect = mpatches.Rectangle(
        (axle_x - tire_r, y_track - tire_w / 2),
        2 * tire_r, tire_w,
        color="#1f2937", alpha=0.18, ec="#1f2937", lw=1.0, ls="--")
    ax.add_patch(tire_rect)
    ax.text(axle_x, y_track, "tire footprint", ha="center", va="center",
            fontsize=8, color="#1f2937")

    # arch_length (right side)
    _annot(ax, (x1 + 60, y_inb), (x1 + 60, y_track + arch_L / 2),
           f"arch_length = tire_w + 2 × lateral_clear\n"
           f"     = {tire_w:.0f} + 2 × {lat:.0f} = {arch_L:.0f}",
           offset=(180, 0))
    # arch radius across the top (just below outboard edge)
    y_top = y_out - 40
    _annot(ax, (x0, y_top), (x1, y_top),
           f"2 × arch_radius = {2 * arch_r:.0f}", offset=(0, 18))
    # fillet callout pointing at one of the rounded corners
    ax.annotate(f"fillet = {fillet:.0f}\n(rounds inboard cap → sidewall)",
                xy=(x0 + fillet * (1 - np.cos(np.pi / 4)),
                    y_inb + fillet * (1 - np.sin(np.pi / 4))),
                xytext=(x0 - 280, y_inb - 80),
                fontsize=9, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    ax.text(axle_x, floor_edge_y + 18,
            "outboard trim at floor edge",
            ha="center", fontsize=8, color="#1f2937")

    ax.set_xlim(x0 - 350, x1 + 380)
    ax.set_ylim(y_inb - 180, floor_edge_y + 100)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm) — forward")
    ax.set_ylabel("y (mm) — right")
    ax.set_title("Wheelhouse — XY plan (front-right corner)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    _save(fig, "wheelhouse_plan.png")


# ---------------------------------------------------------------------------
# floor_builder
# ---------------------------------------------------------------------------

def make_floor_section():
    s = FloorSpec(
        floor_x_min=-2150.0, floor_x_max=2200.0,
        diffuser_start_x_mm=-1350.0,
    )
    ride_h = s.ride_height_mm
    angle = radians(s.diffuser_angle_deg)
    diff_len = s.diffuser_start_x_mm - s.floor_x_min
    diff_end_z = ride_h + diff_len * tan(angle)
    d_consume = s.diffuser_radius_mm * tan(angle / 2.0)
    f_tan_x = s.diffuser_start_x_mm + d_consume
    d_tan_x = s.diffuser_start_x_mm - d_consume * cos(angle)
    d_tan_z = ride_h + d_consume * sin(angle)
    arc_mid_x = s.diffuser_start_x_mm + s.diffuser_radius_mm * (
        tan(angle / 2.0) - sin(angle / 2.0))
    arc_mid_z = ride_h + s.diffuser_radius_mm * (1.0 - cos(angle / 2.0))

    fig, ax = plt.subplots(figsize=(12, 5))
    # ground
    ax.fill_between([s.floor_x_min - 200, s.floor_x_max + 200], -40, 0,
                     color="#e5e7eb", lw=0)
    ax.axhline(0, color="#777", lw=0.8, label="ground")
    # NB: aspect intentionally NOT equal (X extent ~4350 mm vs Z ~400 mm)
    # so the floor / diffuser geometry is legible.

    # flat floor
    ax.plot([s.floor_x_max, f_tan_x], [ride_h, ride_h],
            color="#1f2937", lw=2.0, label="floor + diffuser")
    # fillet arc — sample
    arc_t = np.linspace(0, 1, 30)
    arc_x = (1 - arc_t)**2 * f_tan_x + 2 * (1 - arc_t) * arc_t * arc_mid_x + arc_t**2 * d_tan_x
    arc_z = (1 - arc_t)**2 * ride_h + 2 * (1 - arc_t) * arc_t * arc_mid_z + arc_t**2 * d_tan_z
    ax.plot(arc_x, arc_z, color="#1f2937", lw=2.0)
    # diffuser ramp
    ax.plot([d_tan_x, s.floor_x_min], [d_tan_z, diff_end_z],
            color="#1f2937", lw=2.0)

    # callouts
    ax.axvline(s.diffuser_start_x_mm, color="#1f4ed8", lw=0.8, ls="--")
    ax.text(s.diffuser_start_x_mm + 60, diff_end_z + 200,
            f"diffuser_start_x_mm = {s.diffuser_start_x_mm:.0f}\n"
            "(= rear axle if None)",
            ha="left", fontsize=9, color="#1f4ed8",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#1f4ed8", lw=0.6))

    _annot(ax, (s.floor_x_min - 60, 0), (s.floor_x_min - 60, ride_h),
           f"ride_height = {ride_h:.0f}", offset=(-150, 0))
    ax.text((d_tan_x + s.floor_x_min) / 2 - 80, diff_end_z + 50,
            f"diffuser_angle = {s.diffuser_angle_deg:.0f}°\n"
            f"diff_end_z ≈ {diff_end_z:.0f}",
            ha="center", color="#1f4ed8", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="#1f4ed8", lw=0.6))
    ax.annotate(f"diffuser_radius = {s.diffuser_radius_mm:.0f}",
                xy=((f_tan_x + d_tan_x) / 2, (ride_h + d_tan_z) / 2),
                xytext=(arc_mid_x + 400, arc_mid_z + 100),
                fontsize=9, color="#1f4ed8",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#1f4ed8", lw=0.6),
                arrowprops=dict(arrowstyle="->", color="#1f4ed8", lw=1))
    # span line BELOW the ground (out of the geometry)
    _annot(ax, (s.floor_x_min, -30), (s.floor_x_max, -30),
           f"floor_x_min ... floor_x_max  "
           f"(span = {s.floor_x_max - s.floor_x_min:.0f})", offset=(0, -20))

    ax.set_xlim(s.floor_x_min - 400, s.floor_x_max + 400)
    ax.set_ylim(-100, diff_end_z + 350)
    # NOTE: aspect intentionally not 'equal' (X span >> Z span).
    ax.set_xlabel("x (mm) — forward")
    ax.set_ylabel("z (mm) — up  (NOT to scale with X)")
    ax.set_title("Floor + diffuser — XZ underside contour (Y = 0)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    _save(fig, "floor_section.png")


def make_floor_plan():
    s = FloorSpec(
        floor_x_min=-2150.0, floor_x_max=2200.0,
        diffuser_start_x_mm=-1350.0,
    )
    y_half = s.floor_width_mm / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    # floor rect
    rect = mpatches.Rectangle(
        (s.floor_x_min, -y_half), s.floor_x_max - s.floor_x_min,
        s.floor_width_mm, color="#cfd5dc", alpha=0.6,
        ec="#1f2937", lw=1.5, label="floor plan rectangle")
    ax.add_patch(rect)
    # diffuser start line
    ax.axvline(s.diffuser_start_x_mm, color="#1f4ed8", lw=1.2, ls="--",
               label="diffuser_start_x")

    # wheelhouse cutouts (placeholder rectangles at front + rear)
    front_axle = 1350.0
    rear_axle = -1350.0
    track_f = 1550.0
    track_r = 1540.0
    tire_r_arch = 360.0
    arch_L_f = 225 + 2 * (35 + 150)
    arch_L_r = 225 + 2 * 35
    for ax_, track, L in [
        (front_axle, track_f, arch_L_f),
        (rear_axle, track_r, arch_L_r),
    ]:
        for sign in (+1, -1):
            r = mpatches.Rectangle(
                (ax_ - tire_r_arch, sign * track / 2 - L / 2),
                2 * tire_r_arch, L,
                color="white", ec="#dc2626", lw=1.4,
                label=("wheelhouse cutout" if (ax_ == front_axle and sign == 1)
                       else None))
            ax.add_patch(r)

    # callouts
    _annot(ax, (s.floor_x_min, -y_half - 80),
           (s.floor_x_max, -y_half - 80),
           f"floor_x_min ... floor_x_max  "
           f"(= {s.floor_x_max - s.floor_x_min:.0f})", offset=(0, -22))
    _annot(ax, (s.floor_x_max + 80, -y_half), (s.floor_x_max + 80, y_half),
           f"floor_width_mm = {s.floor_width_mm:.0f}", offset=(150, 0))

    ax.set_xlim(s.floor_x_min - 250, s.floor_x_max + 350)
    ax.set_ylim(-y_half - 200, y_half + 120)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm) — forward")
    ax.set_ylabel("y (mm) — right")
    ax.set_title("Floor — XY plan (rectangle minus 4 wheelhouse cut-outs)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    _save(fig, "floor_plan.png")


def main():
    print("Generating diagrams under", OUT)
    make_tire_section()
    make_spoke_section()
    make_spoke_window()
    make_wheelhouse_section()
    make_wheelhouse_plan()
    make_floor_section()
    make_floor_plan()
    print("done.")


if __name__ == "__main__":
    main()
