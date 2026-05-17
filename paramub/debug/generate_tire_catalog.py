"""Generate organized ParamUB outputs, documentation images, and a 30-shape catalog."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import tire_assembly as ta


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
DOCS_DIR = OUTPUT_ROOT / "docs"
EXAMPLE_DIR = OUTPUT_ROOT / "examples" / "default_225-45R18"
CATALOG_DIR = OUTPUT_ROOT / "catalog"
CASES_DIR = CATALOG_DIR / "cases"

PREVIEW_NAME = "preview.png"
STEM_NAME = "tire_assembly"


COMMON_TIRE_CATALOG = [
    {"section": 185, "aspect": 65, "rim": 15, "category": "touring", "label": "compact_sedan"},
    {"section": 195, "aspect": 65, "rim": 15, "category": "touring", "label": "economy_touring"},
    {"section": 205, "aspect": 55, "rim": 16, "category": "touring", "label": "daily_sedan"},
    {"section": 205, "aspect": 60, "rim": 16, "category": "touring", "label": "all_season_sedan"},
    {"section": 215, "aspect": 55, "rim": 17, "category": "touring", "label": "mid_size_sedan"},
    {"section": 215, "aspect": 60, "rim": 17, "category": "suv", "label": "small_cuv"},
    {"section": 225, "aspect": 45, "rim": 17, "category": "performance", "label": "sport_compact"},
    {"section": 225, "aspect": 50, "rim": 17, "category": "touring", "label": "grand_touring"},
    {"section": 225, "aspect": 45, "rim": 18, "category": "performance", "label": "sport_sedan"},
    {"section": 235, "aspect": 40, "rim": 18, "category": "performance", "label": "track_day_front"},
    {"section": 235, "aspect": 45, "rim": 18, "category": "performance", "label": "sport_sedan_plus"},
    {"section": 245, "aspect": 40, "rim": 18, "category": "performance", "label": "track_day_rear"},
    {"section": 245, "aspect": 45, "rim": 18, "category": "touring", "label": "gt_coupe"},
    {"section": 255, "aspect": 35, "rim": 19, "category": "performance", "label": "ultra_performance"},
    {"section": 255, "aspect": 40, "rim": 19, "category": "performance", "label": "staggered_rear"},
    {"section": 265, "aspect": 35, "rim": 19, "category": "performance", "label": "performance_coupe"},
    {"section": 265, "aspect": 40, "rim": 19, "category": "performance", "label": "gt_rear"},
    {"section": 275, "aspect": 35, "rim": 19, "category": "performance", "label": "muscle_rear"},
    {"section": 275, "aspect": 40, "rim": 20, "category": "suv", "label": "sport_suv"},
    {"section": 285, "aspect": 35, "rim": 20, "category": "performance", "label": "supercar_rear"},
    {"section": 225, "aspect": 65, "rim": 17, "category": "suv", "label": "crossover_touring"},
    {"section": 235, "aspect": 60, "rim": 18, "category": "suv", "label": "mid_size_cuv"},
    {"section": 245, "aspect": 60, "rim": 18, "category": "suv", "label": "two_row_suv"},
    {"section": 255, "aspect": 55, "rim": 18, "category": "suv", "label": "touring_suv"},
    {"section": 255, "aspect": 50, "rim": 19, "category": "suv", "label": "sport_cuv"},
    {"section": 265, "aspect": 50, "rim": 20, "category": "suv", "label": "full_size_suv"},
    {"section": 275, "aspect": 45, "rim": 20, "category": "suv", "label": "three_row_suv"},
    {"section": 265, "aspect": 70, "rim": 17, "category": "truck", "label": "all_terrain"},
    {"section": 275, "aspect": 65, "rim": 18, "category": "truck", "label": "light_truck"},
    {"section": 285, "aspect": 70, "rim": 17, "category": "truck", "label": "off_road"},
]


CATEGORY_TUNING = {
    "touring": {
        "wheel_delta": 10,
        "tread_ratio": 0.84,
        "bulge_bias": 0.0,
        "shoulder_radius": 30,
        "crown_base": 390,
        "crown_per_aspect": 0.8,
        "num_spokes": 6,
        "dish_depth": 12,
        "dish_profile": "spherical",
        "hub_bore": 67,
        "wheel_offset": 40,
        "spoke_width_hub": 28,
        "spoke_width_rim": 18,
        "window_inner_gap": 12,
        "window_outer_gap": 15,
        "window_corner_radius": 10,
    },
    "performance": {
        "wheel_delta": 10,
        "tread_ratio": 0.88,
        "bulge_bias": -0.7,
        "shoulder_radius": 22,
        "crown_base": 330,
        "crown_per_aspect": 0.6,
        "num_spokes": 5,
        "dish_depth": 18,
        "dish_profile": "spherical",
        "hub_bore": 72,
        "wheel_offset": 35,
        "spoke_width_hub": 30,
        "spoke_width_rim": 20,
        "window_inner_gap": 12,
        "window_outer_gap": 15,
        "window_corner_radius": 10,
    },
    "suv": {
        "wheel_delta": 15,
        "tread_ratio": 0.83,
        "bulge_bias": 0.4,
        "shoulder_radius": 34,
        "crown_base": 440,
        "crown_per_aspect": 0.9,
        "num_spokes": 6,
        "dish_depth": 16,
        "dish_profile": "spherical",
        "hub_bore": 72,
        "wheel_offset": 30,
        "spoke_width_hub": 30,
        "spoke_width_rim": 18,
        "window_inner_gap": 14,
        "window_outer_gap": 16,
        "window_corner_radius": 10,
    },
    "truck": {
        "wheel_delta": 20,
        "tread_ratio": 0.80,
        "bulge_bias": 1.0,
        "shoulder_radius": 40,
        "crown_base": 520,
        "crown_per_aspect": 1.0,
        "num_spokes": 8,
        "dish_depth": 20,
        "dish_profile": "conical",
        "hub_bore": 78,
        "wheel_offset": 18,
        "spoke_width_hub": 32,
        "spoke_width_rim": 20,
        "window_inner_gap": 16,
        "window_outer_gap": 18,
        "window_corner_radius": 8,
    },
}


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def round_to_5(value):
    return int(round(value / 5.0) * 5)


def slugify(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def merged_spec(overrides=None):
    spec, _ = ta.normalize_parameters(overrides or {})
    return spec


def build_common_spec(entry):
    tuning = CATEGORY_TUNING[entry["category"]]
    section = entry["section"]
    aspect = entry["aspect"]
    rim = entry["rim"]
    tread_width = round_to_5(section * tuning["tread_ratio"])
    wheel_width = round_to_5(max(section - tuning["wheel_delta"], tread_width + 18))
    sidewall_bulge = round(max(4.0, min(12.0, 3.5 + aspect / 10.0 + tuning["bulge_bias"])), 1)
    crown_radius = round_to_5(max(
        tuning["crown_base"] + aspect * tuning["crown_per_aspect"],
        tread_width / 2.0 + 10.0,
    ))

    spec = {
        "section_width_mm": section,
        "aspect_ratio": aspect,
        "rim_diameter_in": rim,
        "sidewall_bulge_mm": sidewall_bulge,
        "sidewall_bulge_pos": 0.45 if entry["category"] == "performance" else 0.5,
        "shoulder_radius_mm": tuning["shoulder_radius"],
        "tread_width_mm": tread_width,
        "crown_radius_mm": crown_radius,
        "wheel_width_mm": wheel_width,
        "rim_flange_mm": 20 if entry["category"] == "truck" else 18,
        "hub_bore_mm": tuning["hub_bore"],
        "wheel_offset_mm": tuning["wheel_offset"],
        "dish_depth_mm": tuning["dish_depth"],
        "dish_profile": tuning["dish_profile"],
        "num_spokes": tuning["num_spokes"],
        "spoke_width_hub_mm": tuning["spoke_width_hub"],
        "spoke_width_rim_mm": tuning["spoke_width_rim"],
        "window_inner_gap_mm": tuning["window_inner_gap"],
        "window_outer_gap_mm": tuning["window_outer_gap"],
        "window_corner_radius_mm": tuning["window_corner_radius"],
    }
    if "overrides" in entry:
        spec.update(entry["overrides"])
    return merged_spec(spec)


def derive(spec):
    rim_radius_mm = spec["rim_diameter_in"] * 25.4 / 2.0
    sidewall_height_mm = spec["section_width_mm"] * spec["aspect_ratio"] / 100.0
    return {
        "rim_radius_mm": rim_radius_mm,
        "sidewall_height_mm": sidewall_height_mm,
        "tire_od_mm": rim_radius_mm * 2.0 + sidewall_height_mm * 2.0,
        "hub_ring_od_mm": spec["hub_bore_mm"] / 2.0 + 30.0,
        "spoke_inner_r_mm": spec["hub_bore_mm"] / 2.0 + 30.0 + spec["window_inner_gap_mm"],
        "spoke_outer_r_mm": rim_radius_mm - spec["window_outer_gap_mm"] - spec["rim_band_width_mm"],
    }


def bezier_point(p0, p1, p2, p3, t):
    return ta._bezier_point(p0, p1, p2, p3, t)


def tire_profile(spec, d, samples=48):
    z_bead = spec["wheel_width_mm"] / 2.0
    z_tread_edge = spec["tread_width_mm"] / 2.0

    r_bead_base = d["rim_radius_mm"]
    r_flange_top = d["rim_radius_mm"] + spec["rim_flange_mm"]
    r_tread_center = d["rim_radius_mm"] + d["sidewall_height_mm"]

    crown_center_r = r_tread_center - spec["crown_radius_mm"]
    r_tread_edge = crown_center_r + math.sqrt(spec["crown_radius_mm"] ** 2 - z_tread_edge ** 2)

    tan_r = z_tread_edge / spec["crown_radius_mm"]
    tan_z = -math.sqrt(spec["crown_radius_mm"] ** 2 - z_tread_edge ** 2) / spec["crown_radius_mm"]

    p0 = (r_flange_top, z_bead)
    p1 = (r_flange_top, z_bead + spec["sidewall_bulge_mm"] * (0.5 + 0.5 * spec["sidewall_bulge_pos"]))
    p3 = (r_tread_edge, z_tread_edge)
    p2 = (
        r_tread_edge - spec["shoulder_radius_mm"] * tan_r,
        z_tread_edge - spec["shoulder_radius_mm"] * tan_z,
    )

    sidewall_pos = [bezier_point(p0, p1, p2, p3, k / samples) for k in range(samples + 1)]
    sidewall_neg = [(r, -z) for r, z in sidewall_pos]
    tread_z = np.linspace(z_tread_edge, -z_tread_edge, 80)
    tread_curve = [
        (crown_center_r + math.sqrt(spec["crown_radius_mm"] ** 2 - z ** 2), z)
        for z in tread_z
    ]

    return {
        "z_bead": z_bead,
        "z_tread_edge": z_tread_edge,
        "r_bead_base": r_bead_base,
        "r_flange_top": r_flange_top,
        "r_tread_center": r_tread_center,
        "r_tread_edge": r_tread_edge,
        "crown_center_r": crown_center_r,
        "sidewall_pos": sidewall_pos,
        "sidewall_neg": sidewall_neg,
        "tread_curve": tread_curve,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }


def wheel_disc_profile(spec, d, samples=28):
    r_hub = spec["hub_bore_mm"] / 2.0
    r_hub_outer = d["hub_ring_od_mm"]
    r_disc_outer = d["rim_radius_mm"] - spec["rim_band_width_mm"] + 0.5

    z_mount = spec["wheel_offset_mm"]
    # Disc OD is flush with the rim's OB face (z = +wheel_width/2) so the
    # cross-section reads as one continuous tangent surface — kept in sync
    # with tire_assembly.build_wheel_disc.
    z_attach_outboard = spec["wheel_width_mm"] / 2.0
    z_attach_inboard = z_attach_outboard - spec["spoke_thickness_mm"]
    z_hub_outboard = z_attach_outboard - spec["dish_depth_mm"]
    if z_hub_outboard <= z_mount + 1.0:
        z_hub_outboard = z_mount + max(spec["spoke_thickness_mm"], 5.0)

    d_r_in = (r_disc_outer - r_hub_outer) * 0.55
    d_r_cap_blend = min(
        max(spec["spoke_outer_crown_mm"] * 2.5, 8.0),
        max((r_disc_outer - r_hub_outer) * 0.35, 8.0),
    )
    p0_in = (r_hub_outer, z_mount)
    p1_in = (r_hub_outer + d_r_in, z_mount)
    p2_in = (r_disc_outer - d_r_cap_blend, z_attach_inboard)
    p3_in = (r_disc_outer, z_attach_inboard)

    # OD cap: +R tangent at IB end (blends into inboard spoke surface),
    # -R tangent at OB end (matches the rim's OB face plane), and bows
    # outward by spoke_outer_crown_mm at the axial midpoint.
    cap_handle_r = max(spec["spoke_outer_crown_mm"], 0.0) / 0.75
    p0_cap = p3_in
    p1_cap = (r_disc_outer + cap_handle_r, z_attach_inboard)
    p2_cap = (r_disc_outer + cap_handle_r, z_attach_outboard)
    p3_cap = (r_disc_outer, z_attach_outboard)

    d_r_out = (r_disc_outer - r_hub_outer) * 0.55
    d_r_out_start = (r_disc_outer - r_hub_outer) * 0.30
    p0_out = (r_disc_outer, z_attach_outboard)
    p1_out = (r_disc_outer - d_r_out_start, z_attach_outboard)  # tangent -R, matches cap
    p2_out = (r_hub_outer + d_r_out, z_hub_outboard)
    p3_out = (r_hub_outer, z_hub_outboard)

    if spec["dish_profile"] == "spherical":
        inboard_pts = [bezier_point(p0_in, p1_in, p2_in, p3_in, k / samples) for k in range(samples + 1)]
        cap_pts = [bezier_point(p0_cap, p1_cap, p2_cap, p3_cap, k / samples) for k in range(samples + 1)]
        outboard_pts = [bezier_point(p0_out, p1_out, p2_out, p3_out, k / samples) for k in range(samples + 1)]
    else:
        inboard_pts = [
            (p0_in[0] + (p3_in[0] - p0_in[0]) * k / samples, p0_in[1] + (p3_in[1] - p0_in[1]) * k / samples)
            for k in range(samples + 1)
        ]
        cap_pts = [bezier_point(p0_cap, p1_cap, p2_cap, p3_cap, k / samples) for k in range(samples + 1)]
        outboard_pts = [
            (p0_out[0] + (p3_out[0] - p0_out[0]) * k / samples, p0_out[1] + (p3_out[1] - p0_out[1]) * k / samples)
            for k in range(samples + 1)
        ]

    return {
        "r_hub": r_hub,
        "r_hub_outer": r_hub_outer,
        "r_disc_outer": r_disc_outer,
        "z_mount": z_mount,
        "z_attach_outboard": z_attach_outboard,
        "z_attach_inboard": z_attach_inboard,
        "z_hub_outboard": z_hub_outboard,
        "inboard_pts": inboard_pts,
        "cap_pts": cap_pts,
        "outboard_pts": outboard_pts,
    }


def rim_barrel_profile(spec, d):
    r_outer = d["rim_radius_mm"]
    r_flange = d["rim_radius_mm"] + spec["rim_flange_mm"]
    r_inner = d["rim_radius_mm"] - spec["rim_band_width_mm"]
    z_out = spec["wheel_width_mm"] / 2.0
    z_in = -spec["wheel_width_mm"] / 2.0
    flange = spec["flange_axial_thickness_mm"]
    return {
        "r_outer": r_outer,
        "r_flange": r_flange,
        "r_inner": r_inner,
        "z_out": z_out,
        "z_in": z_in,
        "flange": flange,
        "polygon": [
            (z_out, r_inner),
            (z_out, r_flange),
            (z_out - flange, r_flange),
            (z_out - flange, r_outer),
            (z_in + flange, r_outer),
            (z_in + flange, r_flange),
            (z_in, r_flange),
            (z_in, r_inner),
        ],
    }


def draw_dimension(ax, start, end, text, text_xy=None, color="#0f172a"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.4, shrinkA=0, shrinkB=0),
    )
    if text_xy is None:
        text_xy = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    ax.text(
        text_xy[0],
        text_xy[1],
        text,
        fontsize=10,
        ha="center",
        va="center",
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.92),
    )


def callout(ax, point, text_xy, text, color="#1f2937", ha="left"):
    ax.annotate(
        text,
        xy=point,
        xytext=text_xy,
        fontsize=10,
        ha=ha,
        va="center",
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.94),
    )


def generate_tunable_map(example_preview_path, spec):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].imshow(mpimg.imread(example_preview_path))
    axes[0].axis("off")
    axes[0].set_title("Default Example Preview", fontsize=18)

    axes[1].axis("off")
    lines = ["Tunable Dimensions Exposed by `tire_assembly.py`", ""]
    for group_name, names in ta.PARAMETER_GROUPS.items():
        lines.append(group_name.upper())
        for name in names:
            marker = "direct"
            if name in {"section_width_mm", "aspect_ratio"}:
                marker = "drives sidewall height"
            elif name == "dish_profile":
                marker = "shape mode"
            lines.append(f"{name:<28} {spec[name]!s:<10}  ({marker})")
        lines.append("")
    lines.append("Note: `section_width_mm` is an input to sidewall height in the")
    lines.append("current generator; axial tire width is mainly controlled by")
    lines.append("`wheel_width_mm` and `tread_width_mm` in this implementation.")
    axes[1].text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("ParamUB Example and Tunable Parameter Map", fontsize=20)
    fig.tight_layout()
    out_path = DOCS_DIR / "example_tunable_map.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_section_cut_diagram(spec):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    d = derive(spec)
    tire = tire_profile(spec, d)
    disc = wheel_disc_profile(spec, d)
    barrel = rim_barrel_profile(spec, d)

    fig, ax = plt.subplots(figsize=(16, 10))

    tire_points = (
        [(tire["z_bead"], tire["r_bead_base"]), (tire["z_bead"], tire["r_flange_top"])]
        + [(z, r) for r, z in tire["sidewall_pos"][1:]]
        + [(z, r) for r, z in tire["tread_curve"][1:]]
        + [(z, r) for r, z in reversed(tire["sidewall_neg"])][1:]
        + [(-tire["z_bead"], tire["r_bead_base"])]
    )
    tire_x, tire_y = zip(*tire_points)
    ax.fill(tire_x, tire_y, color="#d5dbe2", edgecolor="#475569", linewidth=2, alpha=0.9, zorder=1)

    barrel_x, barrel_y = zip(*barrel["polygon"])
    ax.fill(barrel_x, barrel_y, color="#a8b4c0", edgecolor="#334155", linewidth=1.8, alpha=0.95, zorder=3)

    disc_polygon = (
        [(disc["z_mount"], disc["r_hub"]), (disc["z_mount"], disc["r_hub_outer"])]
        + [(z, r) for r, z in disc["inboard_pts"][1:]]
        + [(z, r) for r, z in disc["cap_pts"][1:]]
        + [(z, r) for r, z in disc["outboard_pts"][1:]]
        + [(disc["z_hub_outboard"], disc["r_hub"])]
    )
    disc_x, disc_y = zip(*disc_polygon)
    ax.fill(disc_x, disc_y, color="#6b7280", edgecolor="#111827", linewidth=1.8, alpha=0.9, zorder=4)

    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1)
    ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)

    draw_dimension(
        ax,
        (-spec["wheel_width_mm"] / 2.0, d["rim_radius_mm"] - 25),
        (spec["wheel_width_mm"] / 2.0, d["rim_radius_mm"] - 25),
        f"wheel_width_mm = {spec['wheel_width_mm']}",
        text_xy=(0, d["rim_radius_mm"] - 42),
        color="#0f766e",
    )
    draw_dimension(
        ax,
        (-spec["tread_width_mm"] / 2.0, tire["r_tread_center"] + 32),
        (spec["tread_width_mm"] / 2.0, tire["r_tread_center"] + 32),
        f"tread_width_mm = {spec['tread_width_mm']}",
        text_xy=(0, tire["r_tread_center"] + 48),
        color="#1d4ed8",
    )
    draw_dimension(
        ax,
        (0, d["rim_radius_mm"]),
        (0, tire["r_tread_center"]),
        f"sidewall_height = {d['sidewall_height_mm']:.1f} mm",
        text_xy=(35, (d["rim_radius_mm"] + tire["r_tread_center"]) / 2.0),
        color="#b45309",
    )
    draw_dimension(
        ax,
        (spec["wheel_width_mm"] / 2.0 + 8, d["rim_radius_mm"]),
        (spec["wheel_width_mm"] / 2.0 + 8, d["rim_radius_mm"] + spec["rim_flange_mm"]),
        f"rim_flange_mm = {spec['rim_flange_mm']}",
        text_xy=(spec["wheel_width_mm"] / 2.0 + 52, d["rim_radius_mm"] + spec["rim_flange_mm"] / 2.0),
        color="#7c3aed",
    )
    draw_dimension(
        ax,
        (disc["z_mount"], 0),
        (disc["z_mount"], spec["hub_bore_mm"] / 2.0),
        f"hub_bore_mm = {spec['hub_bore_mm']}",
        text_xy=(disc["z_mount"] - 55, spec["hub_bore_mm"] / 4.0),
        color="#be123c",
    )
    draw_dimension(
        ax,
        (0, disc["r_hub_outer"] + 20),
        (disc["z_mount"], disc["r_hub_outer"] + 20),
        f"wheel_offset_mm = {spec['wheel_offset_mm']}",
        text_xy=(disc["z_mount"] / 2.0, disc["r_hub_outer"] + 38),
        color="#0f766e",
    )
    draw_dimension(
        ax,
        (disc["z_hub_outboard"], disc["r_hub_outer"] + 52),
        (disc["z_attach_outboard"], disc["r_hub_outer"] + 52),
        f"dish_depth_mm = {spec['dish_depth_mm']}",
        text_xy=((disc["z_hub_outboard"] + disc["z_attach_outboard"]) / 2.0, disc["r_hub_outer"] + 68),
        color="#b91c1c",
    )
    draw_dimension(
        ax,
        (disc["z_attach_inboard"], disc["r_disc_outer"] + 14),
        (disc["z_attach_outboard"], disc["r_disc_outer"] + 14),
        f"spoke_thickness_mm = {spec['spoke_thickness_mm']}",
        text_xy=((disc["z_attach_inboard"] + disc["z_attach_outboard"]) / 2.0, disc["r_disc_outer"] + 30),
        color="#1e3a8a",
    )
    draw_dimension(
        ax,
        (barrel["z_out"] - barrel["flange"], barrel["r_flange"] + 18),
        (barrel["z_out"], barrel["r_flange"] + 18),
        f"flange_axial_thickness_mm = {spec['flange_axial_thickness_mm']}",
        text_xy=(barrel["z_out"] - barrel["flange"] / 2.0, barrel["r_flange"] + 34),
        color="#92400e",
    )
    draw_dimension(
        ax,
        (-barrel["z_in"] + 28, barrel["r_inner"]),
        (-barrel["z_in"] + 28, barrel["r_outer"]),
        f"rim_band_width_mm = {spec['rim_band_width_mm']}",
        text_xy=(-barrel["z_in"] + 66, (barrel["r_inner"] + barrel["r_outer"]) / 2.0),
        color="#7c2d12",
    )

    callout(
        ax,
        (tire["p1"][1], tire["p1"][0]),
        (spec["wheel_width_mm"] / 2.0 + 70, tire["p1"][0] + 22),
        "sidewall_bulge_mm\nand sidewall_bulge_pos",
        color="#0369a1",
    )
    callout(
        ax,
        (tire["p3"][1], tire["p3"][0]),
        (spec["wheel_width_mm"] / 2.0 + 65, tire["p3"][0] - 12),
        f"shoulder_radius_mm = {spec['shoulder_radius_mm']}",
        color="#6d28d9",
    )
    callout(
        ax,
        (disc["z_attach_inboard"] + (disc["z_attach_outboard"] - disc["z_attach_inboard"]) / 2.0,
         disc["r_disc_outer"] + spec["spoke_outer_crown_mm"]),
        (spec["wheel_width_mm"] / 2.0 + 78, disc["r_disc_outer"] + 18),
        f"spoke_outer_crown_mm = {spec['spoke_outer_crown_mm']}",
        color="#0f766e",
    )
    callout(
        ax,
        (0, tire["r_tread_center"]),
        (-spec["wheel_width_mm"] / 2.0 - 95, tire["r_tread_center"] + 24),
        f"crown_radius_mm = {spec['crown_radius_mm']}",
        color="#b45309",
        ha="right",
    )
    callout(
        ax,
        (spec["wheel_width_mm"] / 2.0, d["rim_radius_mm"]),
        (spec["wheel_width_mm"] / 2.0 + 70, d["rim_radius_mm"] - 18),
        f"rim_diameter_in = {spec['rim_diameter_in']}",
        color="#0f172a",
    )

    ax.text(
        0.02,
        0.98,
        "Section cut labels track the actual code inputs.\n"
        "`section_width_mm` and `aspect_ratio` combine into sidewall height.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#334155", alpha=0.95),
    )

    ax.set_title("ParamUB Labeled Side Section", fontsize=20)
    ax.set_xlabel("Axial Z (mm)")
    ax.set_ylabel("Radial R (mm)")
    ax.set_aspect("equal")
    ax.set_xlim(-spec["wheel_width_mm"] / 2.0 - 140, spec["wheel_width_mm"] / 2.0 + 160)
    ax.set_ylim(0, tire["r_tread_center"] + 80)
    ax.grid(alpha=0.18)
    fig.tight_layout()

    out_path = DOCS_DIR / "section_cut_labeled.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_spoke_window_diagram(spec):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    d = derive(spec)
    pitch_deg = 360.0 / spec["num_spokes"]
    center_angle_deg = 180.0 / spec["num_spokes"]
    spoke_inner_deg = math.degrees(2 * math.atan(spec["spoke_width_hub_mm"] / 2.0 / d["spoke_inner_r_mm"]))
    spoke_outer_deg = math.degrees(2 * math.atan(spec["spoke_width_rim_mm"] / 2.0 / d["spoke_outer_r_mm"]))
    window_inner_half_deg = (pitch_deg - spoke_inner_deg) / 2.0
    window_outer_half_deg = (pitch_deg - spoke_outer_deg) / 2.0

    inner_lo = math.radians(center_angle_deg - window_inner_half_deg)
    inner_hi = math.radians(center_angle_deg + window_inner_half_deg)
    outer_lo = math.radians(center_angle_deg - window_outer_half_deg)
    outer_hi = math.radians(center_angle_deg + window_outer_half_deg)
    mid = math.radians(center_angle_deg)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    def circle(radius, color, linestyle="-", lw=1.3):
        theta = np.linspace(0, 2 * math.pi, 360)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=color, linestyle=linestyle, linewidth=lw)

    circle(spec["hub_bore_mm"] / 2.0, "#94a3b8")
    circle(d["hub_ring_od_mm"], "#475569", lw=1.5)
    circle(d["spoke_inner_r_mm"], "#0ea5e9", linestyle="--")
    circle(d["spoke_outer_r_mm"], "#0ea5e9", linestyle="--")
    circle(d["rim_radius_mm"] - spec["rim_band_width_mm"], "#334155", linestyle=":")
    circle(d["rim_radius_mm"], "#111827", lw=1.8)
    circle(d["rim_radius_mm"] + spec["rim_flange_mm"], "#64748b")

    for i in range(spec["num_spokes"]):
        spoke_center_deg = i * pitch_deg
        spoke_center = math.radians(spoke_center_deg)
        spoke_pts = [
            (d["hub_ring_od_mm"] * math.cos(spoke_center - math.radians(spoke_inner_deg / 2.0)),
             d["hub_ring_od_mm"] * math.sin(spoke_center - math.radians(spoke_inner_deg / 2.0))),
            ((d["rim_radius_mm"] - spec["rim_band_width_mm"]) * math.cos(spoke_center - math.radians(spoke_outer_deg / 2.0)),
             (d["rim_radius_mm"] - spec["rim_band_width_mm"]) * math.sin(spoke_center - math.radians(spoke_outer_deg / 2.0))),
            ((d["rim_radius_mm"] - spec["rim_band_width_mm"]) * math.cos(spoke_center + math.radians(spoke_outer_deg / 2.0)),
             (d["rim_radius_mm"] - spec["rim_band_width_mm"]) * math.sin(spoke_center + math.radians(spoke_outer_deg / 2.0))),
            (d["hub_ring_od_mm"] * math.cos(spoke_center + math.radians(spoke_inner_deg / 2.0)),
             d["hub_ring_od_mm"] * math.sin(spoke_center + math.radians(spoke_inner_deg / 2.0))),
        ]
        ax.fill(*zip(*spoke_pts), color="#d4d9df", edgecolor="#94a3b8", alpha=0.55, zorder=1)

    inner_arc = np.linspace(inner_lo, inner_hi, 40)
    outer_arc = np.linspace(outer_hi, outer_lo, 40)
    win_x = list(d["spoke_inner_r_mm"] * np.cos(inner_arc))
    win_y = list(d["spoke_inner_r_mm"] * np.sin(inner_arc))
    win_x += [
        d["spoke_outer_r_mm"] * math.cos(outer_hi),
        *list(d["spoke_outer_r_mm"] * np.cos(outer_arc)),
        d["spoke_inner_r_mm"] * math.cos(inner_lo),
    ]
    win_y += [
        d["spoke_outer_r_mm"] * math.sin(outer_hi),
        *list(d["spoke_outer_r_mm"] * np.sin(outer_arc)),
        d["spoke_inner_r_mm"] * math.sin(inner_lo),
    ]
    ax.fill(win_x, win_y, color="#38bdf8", edgecolor="#075985", linewidth=2.0, alpha=0.65, zorder=3)

    gap_point_inner_0 = (d["hub_ring_od_mm"] * math.cos(mid), d["hub_ring_od_mm"] * math.sin(mid))
    gap_point_inner_1 = (d["spoke_inner_r_mm"] * math.cos(mid), d["spoke_inner_r_mm"] * math.sin(mid))
    draw_dimension(
        ax,
        gap_point_inner_0,
        gap_point_inner_1,
        f"window_inner_gap_mm = {spec['window_inner_gap_mm']}",
        text_xy=(55, 45),
        color="#0f766e",
    )

    disc_outer_radius = d["rim_radius_mm"] - spec["rim_band_width_mm"]
    gap_point_outer_0 = (d["spoke_outer_r_mm"] * math.cos(mid), d["spoke_outer_r_mm"] * math.sin(mid))
    gap_point_outer_1 = (disc_outer_radius * math.cos(mid), disc_outer_radius * math.sin(mid))
    draw_dimension(
        ax,
        gap_point_outer_0,
        gap_point_outer_1,
        f"window_outer_gap_mm = {spec['window_outer_gap_mm']}",
        text_xy=(155, 180),
        color="#7c3aed",
    )

    corner_point = (d["spoke_outer_r_mm"] * math.cos(outer_hi), d["spoke_outer_r_mm"] * math.sin(outer_hi))
    callout(
        ax,
        corner_point,
        (190, 95),
        f"window_corner_radius_mm = {spec['window_corner_radius_mm']}",
        color="#be123c",
    )

    callout(
        ax,
        (d["hub_ring_od_mm"], 0),
        (160, -120),
        f"num_spokes = {spec['num_spokes']}\nspoke_width_hub_mm = {spec['spoke_width_hub_mm']}\n"
        f"spoke_width_rim_mm = {spec['spoke_width_rim_mm']}",
        color="#1d4ed8",
    )

    callout(
        ax,
        (0, d["rim_radius_mm"] + spec["rim_flange_mm"]),
        (-205, 220),
        f"rim_radius_mm = {d['rim_radius_mm']:.1f}\nrim_flange_mm = {spec['rim_flange_mm']}\n"
        f"rim_band_width_mm = {spec['rim_band_width_mm']}",
        color="#92400e",
        ha="right",
    )

    ax.set_title("ParamUB Wheel Face and Spoke Window Controls", fontsize=20)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(alpha=0.18)
    limit = d["rim_radius_mm"] + spec["rim_flange_mm"] + 55
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    fig.tight_layout()

    out_path = DOCS_DIR / "spoke_window_labeled.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_parameter_reference(spec):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ParamUB Tunable Parameter Reference",
        "",
        "This reference matches the current code behavior used for the generated examples.",
        "",
    ]
    for group_name, names in ta.PARAMETER_GROUPS.items():
        lines.append(f"## {group_name.title()}")
        lines.append("")
        for name in names:
            note = ""
            if name in {"section_width_mm", "aspect_ratio"}:
                note = " Drives derived `sidewall_height_mm`."
            elif name == "dish_profile":
                note = " Switches between curved and straight spoke-face profiles."
            elif name == "spoke_outer_crown_mm":
                note = " Radial bulge at the midpoint of the OD cap."
            elif name == "spoke_thickness_mm":
                note = " Axial thickness of the disc at the rim attachment."
            elif name == "flange_axial_thickness_mm":
                note = " Axial extent of each rim flange lip."
            lines.append(f"- `{name}`: default `{spec[name]}`.{note}")
        lines.append("")

    lines.extend([
        "## Disc Cross-section Notes",
        "",
        "- The disc OD is built FLUSH with the rim's outboard face (z = +wheel_width/2).",
        "  There is no axial step in front of the OB rim flange — the spoke section's",
        "  outer surface and the rim's OB face form one continuous plane in section.",
        "- The OD cap is a cubic bezier with a +R tangent at its inboard end (so it",
        "  blends into the inboard spoke surface) and a -R tangent at its outboard end",
        "  (so it rolls into the OB face plane). The cap bows outward by",
        "  `spoke_outer_crown_mm` at its axial midpoint.",
        "- The disc's mounting face is built directly at z = +wheel_offset_mm",
        "  (positive ET means the mounting pad sits outboard of the wheel centerline).",
        "  The legacy `axial_shift = wheel_offset_mm - wheel_width_mm / 2` step is now",
        "  a no-op — the disc is already in final coordinates.",
        "",
        "## Notes",
        "",
        "- `section_width_mm` is not a direct axial width control in the current implementation.",
        "- Axial footprint in the generated section is mainly set by `wheel_width_mm` and `tread_width_mm`.",
    ])

    out_path = DOCS_DIR / "tunable_dimensions.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def case_slug(index, entry):
    return f"{index:02d}_{entry['section']}-{entry['aspect']}R{entry['rim']}_{slugify(entry['label'])}"


def run_case(case_name, spec, output_dir, keep_views=False, entry=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    metadata_path = output_dir / "metadata.json"
    summary = {
        "case_name": case_name,
        "status": "failed",
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }
    if entry:
        summary.update({
            "size": f"{entry['section']}/{entry['aspect']}R{entry['rim']}",
            "category": entry["category"],
        })

    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = Tee(sys.stdout, log_handle)
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"=== {case_name} ===")
                result = ta.generate_case(
                    spec,
                    output_dir=str(output_dir),
                    stem=STEM_NAME,
                    preview_name=PREVIEW_NAME,
                    keep_views=keep_views,
                    render=True,
                )
                print(f"=== completed: {case_name} ===")
            payload = {key: value for key, value in result.items() if key != "assembly"}
            write_json(metadata_path, payload)
            summary.update({
                "status": "ok",
                "stl_path": payload["stl_path"],
                "preview_path": payload["preview_path"],
                "watertight": payload["watertight"],
                "repair_applied": payload["repair_applied"],
                "volume_cm3": payload["volume_cm3"],
                "face_count": payload["face_count"],
                "metadata_path": str(metadata_path),
                "normalization_notes": " | ".join(payload["normalization_notes"]),
            })
        except Exception as exc:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"=== FAILED: {case_name} ===")
                traceback.print_exc()
            summary["error"] = str(exc)

    return summary


def write_summary(rows):
    csv_path = CATALOG_DIR / "summary.csv"
    json_path = CATALOG_DIR / "summary.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, rows)
    return csv_path, json_path


def write_outputs_readme(example_row, doc_paths, summary_rows):
    ok_rows = [row for row in summary_rows if row["status"] == "ok"]
    failed_rows = [row for row in summary_rows if row["status"] != "ok"]
    lines = [
        "# ParamUB Generated Outputs",
        "",
        "## Folder Layout",
        "",
        "- `docs/`: labeled technical images and parameter reference.",
        "- `examples/default_225-45R18/`: baseline example STL, preview, views, and run log.",
        "- `catalog/cases/`: one folder per generated tire shape with STL, preview, metadata, and log.",
        "- `catalog/summary.csv`: manifest for the 30 generated cases.",
        "",
        "## Example",
        "",
        f"- Preview: `{example_row.get('preview_path', 'n/a')}`",
        f"- STL: `{example_row.get('stl_path', 'n/a')}`",
        "",
        "## Documentation Files",
        "",
    ]
    for path in doc_paths:
        lines.append(f"- `{path.relative_to(OUTPUT_ROOT)}`")
    lines.extend([
        "",
        "## Catalog Status",
        "",
        f"- Successful cases: {len(ok_rows)} / {len(summary_rows)}",
        f"- Failed cases: {len(failed_rows)}",
    ])
    if failed_rows:
        lines.append("")
        lines.append("### Failed Cases")
        lines.append("")
        for row in failed_rows:
            lines.append(f"- `{row['case_name']}`: {row.get('error', 'unknown error')}")

    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_structure():
    for path in [OUTPUT_ROOT, DOCS_DIR, EXAMPLE_DIR, CATALOG_DIR, CASES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def main():
    ensure_structure()

    default_spec = merged_spec()
    example_row = run_case("default_225-45R18", default_spec, EXAMPLE_DIR, keep_views=True)

    doc_paths = [
        generate_tunable_map(EXAMPLE_DIR / PREVIEW_NAME, default_spec),
        generate_section_cut_diagram(default_spec),
        generate_spoke_window_diagram(default_spec),
        generate_parameter_reference(default_spec),
    ]

    summary_rows = []
    for index, entry in enumerate(COMMON_TIRE_CATALOG, start=1):
        slug = case_slug(index, entry)
        spec = build_common_spec(entry)
        case_dir = CASES_DIR / slug
        row = run_case(slug, spec, case_dir, keep_views=False, entry=entry)
        summary_rows.append(row)

    write_summary(summary_rows)
    write_outputs_readme(example_row, doc_paths, summary_rows)

    failures = [row for row in summary_rows if row["status"] != "ok"]
    if failures:
        print(f"completed with {len(failures)} failure(s)")
        return 1

    print("completed catalog generation with 30 successful cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
