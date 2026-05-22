"""Floor + rear diffuser builder.

Produces a zero-thickness underbody underside (CFD surface mode). Two
diffuser construction modes:

**Legacy (angle + radius)** — XZ contour at Y=0 swept across floor width:

    flat floor (z = ride_height)  ->  R fillet  ->  diffuser ramp (angle)

  - ``diffuser_start_x_mm``: where the floor stops being flat.
  - ``diffuser_radius_mm``: tangent fillet between flat and ramp.
  - ``diffuser_angle_deg``: ramp angle.

**Multisection (Bezier loft)** — set ``FloorSpec.diffuser_sections`` to
a list of :class:`DiffuserSection`, each defining a cubic Bezier
cross-section at a given Y. Cross-sections are auto-mirrored across
Y=0 and lofted (smooth for N>=3, ruled for N==2). The flat front floor
and the diffuser are part of a single lofted surface — each section's
profile spans floor_x_max -> kick (flat) -> Bezier -> rear endpoint.
Each section is tangent to the flat floor at its kickline by
construction. When set, the legacy ``diffuser_angle_deg`` /
``diffuser_radius_mm`` are ignored.

The companion ``below_cropper`` Solid (see :func:`build_below_cropper`)
is a half-space below the floor contour, used elsewhere to clip
wheelhouse faces so they follow the diffuser slope instead of poking out.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from typing import Optional

import cadquery as cq


# =====================================================================
# Cubic Bezier primitives (layered API)
#
#   Layer 1: cubic_bezier_edge — raw 4 control points -> cq.Edge.
#   Layer 2: cubic_bezier_from_tangents — endpoints + outgoing tangent
#            directions + relative handle strengths.
#   Layer 3: DiffuserSection (below) — special case of Layer 2 pinned
#            to the floor and planar in Y per section.
# =====================================================================

def cubic_bezier_edge(p0: cq.Vector, p1: cq.Vector,
                      p2: cq.Vector, p3: cq.Vector) -> cq.Edge:
    """Cubic Bezier as a ``cq.Edge`` from four explicit control points."""
    return cq.Edge.makeBezier([p0, p1, p2, p3])


def cubic_bezier_from_tangents(
    p0: cq.Vector, p3: cq.Vector,
    t0: cq.Vector, t3: cq.Vector,
    start_strength: float, end_strength: float,
) -> cq.Edge:
    """Cubic Bezier defined by endpoints, outgoing tangent directions,
    and tangent handle strengths normalised to the chord length.

    Parameters
    ----------
    p0, p3 : cq.Vector
        Curve endpoints.
    t0, t3 : cq.Vector
        Tangent direction of *curve travel* at each endpoint, in the
        direction of increasing curve parameter (both outgoing — t0
        at the start, t3 at the end). Need not be unit length; the
        function normalises internally.
    start_strength, end_strength : float
        Relative handle lengths. With ``chord = |p3 - p0|``::

            P1 = P0 + start_strength * chord * normalize(t0)
            P2 = P3 - end_strength   * chord * normalize(t3)

        A strength of 0 puts the handle at the endpoint (curve clamps
        hard to its tangent direction); typical values are 0.2–0.5.
    """
    chord = (p3 - p0).Length
    p1 = p0 + t0.normalized() * (start_strength * chord)
    p2 = p3 - t3.normalized() * (end_strength * chord)
    return cubic_bezier_edge(p0, p1, p2, p3)


@dataclass
class DiffuserSection:
    """One cubic-Bezier cross-section of the diffuser at a given Y.

    A special case of :func:`cubic_bezier_from_tangents` pinned to the
    floor and planar in Y::

        P0 = (kick_x_mm, y_mm, ride_h)            # on the flat floor
        P3 = (rear_x_mm, y_mm, rear_z_mm)         # rear endpoint
        t0 = (-1, 0, 0)                           # tangent to floor, -X
        t3 = (-cos(α), 0, +sin(α))                # in XZ at angle α
        start_strength, end_strength              # normalised handle lengths

    where ``α = radians(rear_angle_deg)``. Because ``t0`` has no Z
    component, the curve is tangent to the flat floor at the kickline
    by construction; the loft preserves this along the whole kickline.

    Sections at ``y_mm > 0`` are auto-mirrored across Y=0 by the floor
    builder, so a symmetric body needs only one half specified.
    """
    y_mm: float
    kick_x_mm: float
    rear_x_mm: float
    rear_z_mm: float
    rear_angle_deg: float
    start_strength: float      # |P0-P1| / |P0-P3|, dimensionless
    end_strength: float        # |P3-P2| / |P0-P3|, dimensionless

    def __post_init__(self) -> None:
        if self.rear_x_mm >= self.kick_x_mm:
            raise ValueError(
                f"DiffuserSection: rear_x_mm ({self.rear_x_mm}) must be "
                f"< kick_x_mm ({self.kick_x_mm}) — diffuser runs rearward.")
        if self.rear_z_mm < 0.0:
            raise ValueError(
                f"DiffuserSection: rear_z_mm ({self.rear_z_mm}) must be >= 0.")
        if self.start_strength < 0.0 or self.end_strength < 0.0:
            raise ValueError(
                "DiffuserSection: start_strength and end_strength "
                "must be >= 0.")


@dataclass
class FloorSpec:
    """Plan + diffuser + ride-height parameters for the underbody surface.

    Plan
        floor_x_min, floor_x_max:  longitudinal extent of the floor.
                                   (Computed by the underbody assembler
                                   from wheelbase + overhangs.)
        floor_width_mm:            full lateral width (±half each side).

    Ride
        ride_height_mm:            ground -> floor underside.

    Diffuser (legacy)
        diffuser_start_x_mm:       X where the floor stops being flat and
                                   the fillet starts. None means "use the
                                   rear axle X" — set by the assembler.
        diffuser_angle_deg:        ramp angle (positive = kicks upward at
                                   the rear).
        diffuser_radius_mm:        tangent fillet between flat and ramp.

    Diffuser (multisection)
        diffuser_sections:         list of DiffuserSection at y_mm >= 0.
                                   If set, the legacy fields are ignored
                                   and the diffuser is built as a Bezier
                                   loft between the (mirrored, padded)
                                   sections.
    """
    floor_x_min: float
    floor_x_max: float
    floor_width_mm: float = 1800.0

    ride_height_mm: float = 100.0

    diffuser_start_x_mm: Optional[float] = None
    diffuser_angle_deg: float = 7.0
    diffuser_radius_mm: float = 500.0

    diffuser_sections: Optional[list[DiffuserSection]] = None

    def validate(self) -> None:
        if self.floor_x_max <= self.floor_x_min:
            raise ValueError("floor_x_max must be > floor_x_min")
        if self.diffuser_sections is None:
            if self.diffuser_start_x_mm is not None:
                if self.diffuser_start_x_mm <= self.floor_x_min:
                    raise ValueError(
                        "diffuser_start_x_mm must be > floor_x_min "
                        "(otherwise the diffuser has zero length)")
            return
        # Multisection validation.
        secs = self.diffuser_sections
        if len(secs) < 1:
            raise ValueError(
                "diffuser_sections must contain at least one section.")
        for s in secs:
            if s.y_mm < 0.0:
                raise ValueError(
                    f"DiffuserSection.y_mm ({s.y_mm}) must be >= 0 — "
                    "sections at y > 0 are auto-mirrored across Y=0.")
        if not any(s.y_mm > 0.0 for s in secs):
            raise ValueError(
                "diffuser_sections must contain at least one section with "
                "y_mm > 0 (so auto-mirroring produces a real second "
                "section).")
        ys = [s.y_mm for s in secs]
        if ys != sorted(ys) or len(set(ys)) != len(ys):
            raise ValueError(
                "diffuser_sections must be sorted by y_mm ascending with "
                "no duplicate y values.")
        for s in secs:
            if s.kick_x_mm >= self.floor_x_max:
                raise ValueError(
                    f"DiffuserSection.kick_x_mm ({s.kick_x_mm}) must be "
                    f"< floor_x_max ({self.floor_x_max}).")


def _diff_geometry(spec: FloorSpec):
    """Return the key contour points + segments along the underside profile."""
    spec.validate()
    diff_start_x = (spec.diffuser_start_x_mm
                    if spec.diffuser_start_x_mm is not None
                    else 0.0)  # fallback; assembler always passes a value.
    ride_h = spec.ride_height_mm
    angle_rad = radians(spec.diffuser_angle_deg)

    diff_length = diff_start_x - spec.floor_x_min
    diff_end_z = ride_h + diff_length * tan(angle_rad)

    d_consume = spec.diffuser_radius_mm * tan(angle_rad / 2.0)
    floor_tangent_x = diff_start_x + d_consume
    diff_tangent_x = diff_start_x - d_consume * cos(angle_rad)
    diff_tangent_z = ride_h + d_consume * sin(angle_rad)

    arc_mid_x = diff_start_x + spec.diffuser_radius_mm * (
        tan(angle_rad / 2.0) - sin(angle_rad / 2.0))
    arc_mid_z = ride_h + spec.diffuser_radius_mm * (1.0 - cos(angle_rad / 2.0))

    return dict(
        ride_h=ride_h,
        diff_end_z=diff_end_z,
        floor_tangent_x=floor_tangent_x,
        diff_tangent_x=diff_tangent_x,
        diff_tangent_z=diff_tangent_z,
        arc_mid_x=arc_mid_x,
        arc_mid_z=arc_mid_z,
    )


def _contour_wire(spec: FloorSpec):
    """Open wire (in XZ plane at Y = 0) tracing the underbody underside."""
    g = _diff_geometry(spec)
    e_floor = cq.Edge.makeLine(
        cq.Vector(spec.floor_x_max, 0, g["ride_h"]),
        cq.Vector(g["floor_tangent_x"], 0, g["ride_h"]),
    )
    e_fillet = cq.Edge.makeThreePointArc(
        cq.Vector(g["floor_tangent_x"], 0, g["ride_h"]),
        cq.Vector(g["arc_mid_x"], 0, g["arc_mid_z"]),
        cq.Vector(g["diff_tangent_x"], 0, g["diff_tangent_z"]),
    )
    e_diffuser = cq.Edge.makeLine(
        cq.Vector(g["diff_tangent_x"], 0, g["diff_tangent_z"]),
        cq.Vector(spec.floor_x_min, 0, g["diff_end_z"]),
    )
    return cq.Wire.assembleEdges([e_floor, e_fillet, e_diffuser])


def build_floor_surface(spec: FloorSpec):
    """Zero-thickness Face: floor + diffuser underside, full floor width."""
    contour = _contour_wire(spec)
    y_half = spec.floor_width_mm / 2.0
    contour_neg = contour.translate(cq.Vector(0, -y_half, 0))
    contour_pos = contour.translate(cq.Vector(0, +y_half, 0))
    return cq.Face.makeRuledSurface(contour_neg, contour_pos)


def build_floor(spec: FloorSpec):
    """Return the zero-thickness underbody Face (legacy) or Compound (multisection)."""
    spec.validate()
    if spec.diffuser_sections is None:
        return build_floor_surface(spec)
    return build_floor_multisection(spec)


def build_below_cropper(spec: FloorSpec):
    """Solid (wrapped in a Workplane) representing the half-space BELOW
    the underbody underside.

    Used to crop wheelhouse faces so the rear arches follow the diffuser
    slope. Returned wrapped in a Workplane so the caller can ``.val()`` it
    just like the legacy form.
    """
    spec.validate()
    if spec.diffuser_sections is None:
        return _build_below_cropper_legacy(spec)
    return _build_below_cropper_multisection(spec)


def _build_below_cropper_legacy(spec: FloorSpec):
    g = _diff_geometry(spec)
    z_below = -2000.0
    pad_x = 200.0
    cropper_x_min = spec.floor_x_min - pad_x
    cropper_x_max = spec.floor_x_max + pad_x

    profile = (
        cq.Workplane("XZ")
        .moveTo(cropper_x_min, z_below)
        .lineTo(cropper_x_min, g["diff_end_z"])
        .lineTo(spec.floor_x_min, g["diff_end_z"])
        .lineTo(g["diff_tangent_x"], g["diff_tangent_z"])
        .threePointArc((g["arc_mid_x"], g["arc_mid_z"]),
                       (g["floor_tangent_x"], g["ride_h"]))
        .lineTo(spec.floor_x_max, g["ride_h"])
        .lineTo(cropper_x_max, g["ride_h"])
        .lineTo(cropper_x_max, z_below)
        .close()
    )
    return profile.extrude(spec.floor_width_mm * 1.5, both=True)


# =====================================================================
# Multisection (Bezier loft) diffuser
# =====================================================================


def _section_bezier_edge(sec: DiffuserSection, ride_h: float) -> cq.Edge:
    """Build one section's Bezier edge by delegating to the Layer-2 wrapper.

    Encodes the diffuser's special-case constraints: P0 on the floor,
    start tangent along -X, end tangent in the XZ plane at
    ``rear_angle_deg`` above -X.
    """
    a = radians(sec.rear_angle_deg)
    p0 = cq.Vector(sec.kick_x_mm, sec.y_mm, ride_h)
    p3 = cq.Vector(sec.rear_x_mm, sec.y_mm, sec.rear_z_mm)
    t0 = cq.Vector(-1.0, 0.0, 0.0)
    t3 = cq.Vector(-cos(a), 0.0, sin(a))
    return cubic_bezier_from_tangents(
        p0, p3, t0, t3,
        sec.start_strength, sec.end_strength,
    )


def _expand_sections(user_secs: list[DiffuserSection]) -> list[DiffuserSection]:
    """Auto-mirror y>0 sections across Y=0."""
    mirrored = [
        DiffuserSection(
            y_mm=-s.y_mm,
            kick_x_mm=s.kick_x_mm,
            rear_x_mm=s.rear_x_mm,
            rear_z_mm=s.rear_z_mm,
            rear_angle_deg=s.rear_angle_deg,
            start_strength=s.start_strength,
            end_strength=s.end_strength,
        )
        for s in user_secs if s.y_mm > 0.0
    ]
    return sorted([*user_secs, *mirrored], key=lambda s: s.y_mm)


def _section_wire(sec: DiffuserSection, floor_x_max: float, ride_h: float):
    """Open wire: floor_x_max -> kick (flat) -> Bezier -> rear endpoint,
    all at sec.y_mm. This is the full underbody underside profile at this Y."""
    p0 = cq.Vector(sec.kick_x_mm, sec.y_mm, ride_h)
    front_pt = cq.Vector(floor_x_max, sec.y_mm, ride_h)
    flat_edge = cq.Edge.makeLine(front_pt, p0)
    bezier_edge = _section_bezier_edge(sec, ride_h)
    return cq.Wire.assembleEdges([flat_edge, bezier_edge])


def _loft_section_wires(wires, ruled: bool):
    """Loft a list of open wires into a Shell/Face via BRepOffsetAPI_ThruSections."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    builder = BRepOffsetAPI_ThruSections(False, ruled)  # isSolid=False
    for w in wires:
        builder.AddWire(w.wrapped)
    builder.Build()
    return cq.Shape.cast(builder.Shape())


def build_floor_multisection(spec: FloorSpec):
    """Single Face/Shell: each section's full XZ profile (flat + Bezier)
    swept laterally across Y via a loft. No joint between flat and diffuser."""
    secs = _expand_sections(spec.diffuser_sections)
    ride_h = spec.ride_height_mm
    section_wires = [_section_wire(s, spec.floor_x_max, ride_h) for s in secs]
    ruled = len(section_wires) == 2
    return _loft_section_wires(section_wires, ruled=ruled)


def _build_below_cropper_multisection(spec: FloorSpec):
    """Half-space below the multisection floor: extrude each face down 2000mm."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    compound = build_floor_multisection(spec)
    drop = gp_Vec(0.0, 0.0, -2000.0)
    solids = []
    for f in compound.Faces():
        prism = BRepPrimAPI_MakePrism(f.wrapped, drop).Shape()
        solids.append(cq.Shape.cast(prism))
    cropper = solids[0]
    for s in solids[1:]:
        cropper = cropper.fuse(s)
    return cq.Workplane(obj=cropper)
