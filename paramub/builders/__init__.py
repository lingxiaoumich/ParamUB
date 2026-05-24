"""CAD-side parametric builders for the underbody + wheel assembly.

Modules:

  tire          TireSpec, build_tire()
  spoke         SpokeSpec, build_spoke_disc(), cut_spoke_windows(),
                build_rim_barrel()
  wheel         WheelSpec, assemble_wheel()
  wheelhouse    WheelhouseSpec, build_wheelhouse_solid(),
                extract_wheelhouse_surfaces()
  floor         FloorSpec, build_floor(), build_below_cropper(),
                DiffuserSection, SplitterSection,
                cubic_bezier_edge, cubic_bezier_from_tangents
  ub_assem      UnderbodySpec, build_underbody()

All builders require cadquery. The top-level :mod:`paramub` package
re-exports the public names from here for convenience.
"""
from .tire import TireSpec, build_tire
from .spoke import (
    SpokeSpec,
    build_spoke_disc,
    cut_spoke_windows,
    build_rim_barrel,
)
from .wheel import WheelSpec, assemble_wheel
from .wheelhouse import (
    WheelhouseSpec,
    build_wheelhouse_solid,
    extract_wheelhouse_surfaces,
)
from .floor import (
    DiffuserSection,
    FloorSpec,
    SplitterSection,
    build_floor,
    cubic_bezier_edge,
    cubic_bezier_from_tangents,
)
from .ub_assem import UnderbodySpec, build_underbody
