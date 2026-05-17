"""ParamUB — parametric car wheel + underbody generator.

Public modules (clean, refactored):

  tire_builder       — TireSpec, build_tire()
  spoke_builder      — SpokeSpec, build_spoke_disc(), build_rim_barrel(),
                       cut_spoke_windows()
  wheel_assem        — WheelSpec, assemble_wheel()
  wheelhouse_builder — WheelhouseSpec, build_wheelhouse_solid(),
                       extract_wheelhouse_surfaces()
  floor_builder      — FloorSpec, build_floor()
  ub_assem           — UnderbodySpec, build_underbody()
  generate           — generate(spec, output_mode={'stl', 'all'})

Old prototypes are preserved under paramub/debug/.
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
from .floor_builder import FloorSpec, build_floor
from .ub_assem import UnderbodySpec, build_underbody
from .generate import generate

__all__ = [
    "TireSpec", "build_tire",
    "SpokeSpec", "build_spoke_disc", "cut_spoke_windows", "build_rim_barrel",
    "WheelSpec", "assemble_wheel",
    "WheelhouseSpec", "build_wheelhouse_solid", "extract_wheelhouse_surfaces",
    "FloorSpec", "build_floor",
    "UnderbodySpec", "build_underbody",
    "generate",
]
