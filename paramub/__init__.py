"""ParamUB — parametric car wheel + underbody generator.

Public modules (clean, refactored):

  tire_builder       — TireSpec, build_tire()
  spoke_builder      — SpokeSpec, build_spoke_disc(), build_rim_barrel(),
                       cut_spoke_windows()
  wheel_assem        — WheelSpec, assemble_wheel()
  wheelhouse_builder — WheelhouseSpec, build_wheelhouse_solid(),
                       extract_wheelhouse_surfaces()
  floor_builder      — FloorSpec, build_floor(),
                       DiffuserSection, SplitterSection (multisection
                       Bezier splitter + diffuser; see
                       docs/multisection_diffuser.html),
                       cubic_bezier_edge(), cubic_bezier_from_tangents()
                       (standalone Bezier primitives)
  ub_assem           — UnderbodySpec, build_underbody(spec)
  generate           — generate(spec, output_mode={'stl', 'all'},
                                stl_tolerance_mm=0.1,
                                stl_angular_tolerance_rad=0.1,
                                floor_angular_tolerance_rad=0.02)

Old prototypes are preserved under paramub/debug/.

See paramub/docs/index.html (or userguide.html for the quick start)
for full documentation. The repo-root scripts ``makeUB.py`` and
``makeUB_example.py`` are runnable examples; ``integrate_underbody.py``
bridges an extracted upper-body shell with a matching parametric UB.
"""

from .tire_builder import TireSpec, build_tire
from .spoke_builder import (
    SpokeSpec,
    build_spoke_disc,
    cut_spoke_windows,
    build_rim_barrel,
)
from .wheel_assem import WheelSpec, assemble_wheel
from .wheelhouse_builder import (
    WheelhouseSpec,
    build_wheelhouse_solid,
    extract_wheelhouse_surfaces,
)
from .floor_builder import (
    DiffuserSection,
    FloorSpec,
    SplitterSection,
    build_floor,
    cubic_bezier_edge,
    cubic_bezier_from_tangents,
)
from .ub_assem import UnderbodySpec, build_underbody
from .generate import generate

__all__ = [
    "TireSpec", "build_tire",
    "SpokeSpec", "build_spoke_disc", "cut_spoke_windows", "build_rim_barrel",
    "WheelSpec", "assemble_wheel",
    "WheelhouseSpec", "build_wheelhouse_solid", "extract_wheelhouse_surfaces",
    "cubic_bezier_edge", "cubic_bezier_from_tangents",
    "DiffuserSection", "SplitterSection", "FloorSpec", "build_floor",
    "UnderbodySpec", "build_underbody",
    "generate",
]
