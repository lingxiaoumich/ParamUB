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
class SplitterSection:
    """One cubic-Bezier cross-section of the front splitter at a given Y.

    Mirror of :class:`DiffuserSection` at the front of the car::

        P0 = (kick_x_mm,  y_mm, ride_h)            # on the flat floor
        P3 = (front_x_mm, y_mm, front_z_mm)        # front endpoint
        t0 = (+1, 0, 0)                            # tangent to floor, +X
        t3 = (+cos(α), 0, +sin(α))                 # in XZ at angle α
        start_strength, end_strength               # normalised handle lengths

    where ``α = radians(front_angle_deg)``. Positive ``front_angle_deg``
    lifts the splitter away from the ground at the leading edge;
    negative dips it toward the ground. ``front_z_mm`` can be anywhere
    (above, equal to, or below ``ride_h``) — pair it with
    ``front_angle_deg`` to shape leading-edge behaviour.

    Sections at ``y_mm > 0`` are auto-mirrored across Y=0.
    """
    y_mm: float
    kick_x_mm: float
    front_x_mm: float
    front_z_mm: float
    front_angle_deg: float
    start_strength: float
    end_strength: float

    def __post_init__(self) -> None:
        if self.front_x_mm <= self.kick_x_mm:
            raise ValueError(
                f"SplitterSection: front_x_mm ({self.front_x_mm}) must be "
                f"> kick_x_mm ({self.kick_x_mm}) — splitter runs forward.")
        if self.front_z_mm < 0.0:
            raise ValueError(
                f"SplitterSection: front_z_mm ({self.front_z_mm}) must be >= 0.")
        if self.start_strength < 0.0 or self.end_strength < 0.0:
            raise ValueError(
                "SplitterSection: start_strength and end_strength "
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
                                   loft between the mirrored sections.

    Splitter (multisection)
        splitter_sections:         list of SplitterSection at y_mm >= 0.
                                   Optional front-of-car analogue of
                                   diffuser_sections.

    When either splitter_sections or diffuser_sections is set, the
    surface becomes a sewn Shell of (splitter face?) + flat floor face +
    (diffuser face?), with kickline edges shared via boundary extraction
    so the three pieces are C1-continuous at the kicklines. Splitter and
    diffuser can have independent N and Y values.
    """
    floor_x_min: float
    floor_x_max: float
    floor_width_mm: float = 1800.0

    ride_height_mm: float = 100.0

    diffuser_start_x_mm: Optional[float] = None
    diffuser_angle_deg: float = 7.0
    diffuser_radius_mm: float = 500.0

    diffuser_sections: Optional[list[DiffuserSection]] = None
    splitter_sections: Optional[list[SplitterSection]] = None

    def validate(self) -> None:
        if self.floor_x_max <= self.floor_x_min:
            raise ValueError("floor_x_max must be > floor_x_min")
        if self.diffuser_sections is None and self.splitter_sections is None:
            if self.diffuser_start_x_mm is not None:
                if self.diffuser_start_x_mm <= self.floor_x_min:
                    raise ValueError(
                        "diffuser_start_x_mm must be > floor_x_min "
                        "(otherwise the diffuser has zero length)")
            return
        # Multisection validation.
        y_half = self.floor_width_mm / 2.0
        if self.diffuser_sections is not None:
            _validate_sections_list(
                self.diffuser_sections, "diffuser_sections", y_half)
            for s in self.diffuser_sections:
                if s.kick_x_mm >= self.floor_x_max:
                    raise ValueError(
                        f"DiffuserSection.kick_x_mm ({s.kick_x_mm}) must be "
                        f"< floor_x_max ({self.floor_x_max}).")
        if self.splitter_sections is not None:
            _validate_sections_list(
                self.splitter_sections, "splitter_sections", y_half)
            for s in self.splitter_sections:
                if s.kick_x_mm <= self.floor_x_min:
                    raise ValueError(
                        f"SplitterSection.kick_x_mm ({s.kick_x_mm}) must be "
                        f"> floor_x_min ({self.floor_x_min}).")
        # If both sets are present, the flat-floor middle region must be
        # non-empty: every splitter kick must be in front of every diffuser
        # kick. (Per-Y check would be stricter; this coarse check catches
        # the obviously-broken case.)
        if (self.splitter_sections is not None
                and self.diffuser_sections is not None):
            min_split_kick = min(s.kick_x_mm for s in self.splitter_sections)
            max_diff_kick = max(s.kick_x_mm for s in self.diffuser_sections)
            if min_split_kick <= max_diff_kick:
                raise ValueError(
                    "Splitter and diffuser kicklines overlap or invert: "
                    f"min(splitter.kick_x)={min_split_kick} must be "
                    f"> max(diffuser.kick_x)={max_diff_kick} so the flat "
                    "floor between them has positive extent.")


def _validate_sections_list(secs: list, name: str, y_half: float) -> None:
    """Common validation for splitter_sections / diffuser_sections."""
    if len(secs) < 1:
        raise ValueError(f"{name} must contain at least one section.")
    for s in secs:
        if s.y_mm < 0.0:
            raise ValueError(
                f"{type(s).__name__}.y_mm ({s.y_mm}) must be >= 0 — "
                "sections at y > 0 are auto-mirrored across Y=0.")
    if not any(s.y_mm > 0.0 for s in secs):
        raise ValueError(
            f"{name} must contain at least one section with y_mm > 0 "
            "(so auto-mirroring produces a real second section).")
    ys = [s.y_mm for s in secs]
    if ys != sorted(ys) or len(set(ys)) != len(ys):
        raise ValueError(
            f"{name} must be sorted by y_mm ascending with no "
            "duplicate y values.")
    outer_y = max(ys)
    if abs(outer_y - y_half) > 0.01:
        raise ValueError(
            f"{name}'s outermost y_mm ({outer_y}) must equal floor_width/2 "
            f"({y_half}) — the splitter/diffuser must reach the floor edge "
            "for the flat-floor boundary to be well-defined.")


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
    """Return the zero-thickness underbody Face (legacy) or sewn Shell
    (multisection — when splitter_sections and/or diffuser_sections are set)."""
    spec.validate()
    if spec.diffuser_sections is None and spec.splitter_sections is None:
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
    if spec.diffuser_sections is None and spec.splitter_sections is None:
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
# Multisection (Bezier loft) splitter + diffuser
# =====================================================================
#
# When splitter_sections and/or diffuser_sections are provided, the
# floor is built as a sewn Shell of up to three faces:
#
#   splitter face  →  flat floor face  →  diffuser face
#
# Each Bezier loft is built independently, then its kickline edge (the
# rail along the loft's u=0 boundary, which lies at z=ride_h) is
# extracted and reused as part of the flat-floor face's boundary. Using
# the extracted edges directly guarantees the shell sews cleanly and
# tangent-continuously at the kicklines (both lofts start with horizontal
# tangent in X, and the flat floor is horizontal everywhere).


from dataclasses import replace


def _diffuser_bezier_edge(sec: DiffuserSection, ride_h: float) -> cq.Edge:
    """One diffuser section's Bezier (kick → rear)."""
    a = radians(sec.rear_angle_deg)
    p0 = cq.Vector(sec.kick_x_mm, sec.y_mm, ride_h)
    p3 = cq.Vector(sec.rear_x_mm, sec.y_mm, sec.rear_z_mm)
    t0 = cq.Vector(-1.0, 0.0, 0.0)
    t3 = cq.Vector(-cos(a), 0.0, sin(a))
    return cubic_bezier_from_tangents(
        p0, p3, t0, t3, sec.start_strength, sec.end_strength)


def _splitter_bezier_edge(sec: SplitterSection, ride_h: float) -> cq.Edge:
    """One splitter section's Bezier (kick → front)."""
    a = radians(sec.front_angle_deg)
    p0 = cq.Vector(sec.kick_x_mm, sec.y_mm, ride_h)
    p3 = cq.Vector(sec.front_x_mm, sec.y_mm, sec.front_z_mm)
    t0 = cq.Vector(+1.0, 0.0, 0.0)
    t3 = cq.Vector(+cos(a), 0.0, sin(a))
    return cubic_bezier_from_tangents(
        p0, p3, t0, t3, sec.start_strength, sec.end_strength)


def _mirror_sections(user_secs: list) -> list:
    """Auto-mirror y>0 sections across Y=0, return sorted by y_mm."""
    mirrored = [replace(s, y_mm=-s.y_mm) for s in user_secs if s.y_mm > 0.0]
    return sorted([*user_secs, *mirrored], key=lambda s: s.y_mm)


def _loft_section_edges(edges: list, ruled: bool):
    """Loft section edges into a Face/Shell via BRepOffsetAPI_ThruSections."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    builder = BRepOffsetAPI_ThruSections(False, ruled)  # isSolid=False
    for e in edges:
        wire = cq.Wire.assembleEdges([e])
        builder.AddWire(wire.wrapped)
    builder.Build()
    return cq.Shape.cast(builder.Shape())


def _extract_kickline_edge(loft_shape, secs, ride_h, tol=0.5) -> cq.Edge:
    """Find the kickline edge of a loft: the rail at u=0 whose endpoints
    match the first and last sections' kick points (P0)."""
    p_first = cq.Vector(secs[0].kick_x_mm, secs[0].y_mm, ride_h)
    p_last = cq.Vector(secs[-1].kick_x_mm, secs[-1].y_mm, ride_h)
    for edge in loft_shape.Edges():
        verts = edge.Vertices()
        if len(verts) != 2:
            continue
        e0, e1 = verts[0].Center(), verts[1].Center()
        m_fwd = max((e0 - p_first).Length, (e1 - p_last).Length)
        m_rev = max((e1 - p_first).Length, (e0 - p_last).Length)
        if min(m_fwd, m_rev) < tol:
            return edge
    raise RuntimeError(
        f"kickline edge not found in loft (expected endpoints "
        f"{p_first.toTuple()} and {p_last.toTuple()})")


def _kickline_vertex(edge: cq.Edge, y_target: float, tol=0.5) -> cq.Vector:
    """Endpoint Vector of edge whose Y coordinate is closest to y_target."""
    best = None
    best_d = float("inf")
    for v in edge.Vertices():
        c = v.Center()
        d = abs(c.y - y_target)
        if d < best_d:
            best_d = d
            best = c
    if best is None or best_d > tol:
        raise RuntimeError(
            f"no kickline-edge endpoint near y={y_target} (closest was {best_d})")
    return best


def _build_splitter_face(spec: FloorSpec):
    """Loft the splitter sections; return (face, kickline_edge)."""
    secs = _mirror_sections(spec.splitter_sections)
    ride_h = spec.ride_height_mm
    edges = [_splitter_bezier_edge(s, ride_h) for s in secs]
    face = _loft_section_edges(edges, ruled=(len(edges) == 2))
    kickline = _extract_kickline_edge(face, secs, ride_h)
    return face, kickline, secs


def _build_diffuser_face(spec: FloorSpec):
    """Loft the diffuser sections; return (face, kickline_edge)."""
    secs = _mirror_sections(spec.diffuser_sections)
    ride_h = spec.ride_height_mm
    edges = [_diffuser_bezier_edge(s, ride_h) for s in secs]
    face = _loft_section_edges(edges, ruled=(len(edges) == 2))
    kickline = _extract_kickline_edge(face, secs, ride_h)
    return face, kickline, secs


def _build_flat_floor_face(splitter_kickline, diffuser_kickline,
                            floor_x_max, floor_x_min, y_half, ride_h):
    """Planar Face at z=ride_h, bounded by the (possibly-extracted)
    splitter and diffuser kickline edges plus side and rear/front lines.

    Reusing the extracted kickline Edge objects ensures the downstream
    sew of splitter+flat+diffuser produces a watertight Shell.
    """
    edges = []
    # Front (splitter kickline, or straight line at floor_x_max).
    if splitter_kickline is not None:
        edges.append(splitter_kickline)
        front_neg = _kickline_vertex(splitter_kickline, -y_half)
        front_pos = _kickline_vertex(splitter_kickline, +y_half)
    else:
        front_neg = cq.Vector(floor_x_max, -y_half, ride_h)
        front_pos = cq.Vector(floor_x_max, +y_half, ride_h)
        edges.append(cq.Edge.makeLine(front_neg, front_pos))
    # Rear (diffuser kickline, or straight line at floor_x_min).
    if diffuser_kickline is not None:
        edges.append(diffuser_kickline)
        rear_neg = _kickline_vertex(diffuser_kickline, -y_half)
        rear_pos = _kickline_vertex(diffuser_kickline, +y_half)
    else:
        rear_neg = cq.Vector(floor_x_min, -y_half, ride_h)
        rear_pos = cq.Vector(floor_x_min, +y_half, ride_h)
        edges.append(cq.Edge.makeLine(rear_neg, rear_pos))
    # Side edges.
    edges.append(cq.Edge.makeLine(front_neg, rear_neg))   # -y side
    edges.append(cq.Edge.makeLine(front_pos, rear_pos))   # +y side
    wire = cq.Wire.assembleEdges(edges)
    return cq.Face.makeFromWires(wire)


def _sew_faces(faces):
    """Sew faces into a single Shell. Falls back to Compound if sewing fails."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    sewing = BRepBuilderAPI_Sewing(0.1)  # 0.1 mm coincidence tolerance
    for f in faces:
        sewing.Add(f.wrapped)
    sewing.Perform()
    return cq.Shape.cast(sewing.SewedShape())


def build_floor_multisection(spec: FloorSpec):
    """Sewn Shell of (optional) splitter face + flat floor + (optional)
    diffuser face. At least one of splitter_sections / diffuser_sections
    must be set on ``spec`` (otherwise the caller would pick the legacy
    builder)."""
    ride_h = spec.ride_height_mm
    y_half = spec.floor_width_mm / 2.0

    faces = []
    splitter_kickline = None
    diffuser_kickline = None

    if spec.splitter_sections is not None:
        splitter_face, splitter_kickline, _ = _build_splitter_face(spec)
        faces.append(splitter_face)

    if spec.diffuser_sections is not None:
        diffuser_face, diffuser_kickline, _ = _build_diffuser_face(spec)
        faces.append(diffuser_face)

    flat_face = _build_flat_floor_face(
        splitter_kickline, diffuser_kickline,
        spec.floor_x_max, spec.floor_x_min, y_half, ride_h)
    faces.append(flat_face)

    return _sew_faces(faces)


def _build_below_cropper_multisection(spec: FloorSpec):
    """Half-space below the multisection floor: extrude each face down 2000mm
    and fuse."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    shape = build_floor_multisection(spec)
    drop = gp_Vec(0.0, 0.0, -2000.0)
    solids = []
    for f in shape.Faces():
        prism = BRepPrimAPI_MakePrism(f.wrapped, drop).Shape()
        solids.append(cq.Shape.cast(prism))
    cropper = solids[0]
    for s in solids[1:]:
        cropper = cropper.fuse(s)
    return cq.Workplane(obj=cropper)
