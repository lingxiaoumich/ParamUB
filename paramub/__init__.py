"""ParamUB — parametric car wheel + underbody generator.

Sub-packages:

  builders/    CAD-side parametric pieces (tire / spoke / wheel /
               wheelhouse / floor / ub_assem). Requires cadquery.
  shell/       Upper-body shell extraction (extract / recut / render).
               Trimesh / scipy / shapely; no cadquery dependency.
  pipeline/    Stage-4 post-trim finishers
               (watertight, summary).  Trimesh + bpy 4.2.

Top-level:

  generate     :func:`generate` — high-level entry point that runs the
               full assembly + writes the chosen output bundle.

For convenience the public names from builders/ are re-exported at the
top level, so::

    from paramub import UnderbodySpec, generate, TireSpec

still works.

Old monolithic prototypes are preserved under ``paramub/debug/``.

See ``paramub/docs/index.html`` (or ``userguide.html`` for the quick
start) for the full documentation. The standalone demos live in
``examples/`` (``examples/makeUB.py`` and
``examples/makeUB_example.py``). The reconstructed-car pipeline
drivers — ``run_shell.py``, ``integrate_underbody.py``,
``fill_gap.py``, ``make_watertight.py`` and the master orchestrator
``make_pipeline.py`` — live at the repo root.
"""

# The CAD-side builders depend on cadquery. Wrap their re-exports in a
# try/except so non-CAD submodules of paramub (e.g. paramub.shell.extract,
# paramub.shell.recut, paramub.pipeline.watertight) stay importable in
# environments without cadquery installed — needed for stages 1/3/4 of
# the reconstructed-car pipeline, which don't touch cadquery.
try:
    from .builders import (
        TireSpec, build_tire,
        SpokeSpec, build_spoke_disc, cut_spoke_windows, build_rim_barrel,
        WheelSpec, assemble_wheel,
        WheelhouseSpec, build_wheelhouse_solid, extract_wheelhouse_surfaces,
        DiffuserSection, FloorSpec, SplitterSection, build_floor,
        cubic_bezier_edge, cubic_bezier_from_tangents,
        UnderbodySpec, build_underbody,
    )
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
except ImportError as _cadquery_import_error:
    # cadquery (or one of its OCP-bound deps) is missing — leave the
    # CAD-side names unexported, but let from-import statements on
    # cadquery-free submodules succeed.
    import warnings as _warnings
    _warnings.warn(
        f"paramub: CAD-side builders unavailable "
        f"({_cadquery_import_error}). Stages that don't need cadquery "
        f"(shell_extract, shell_recut, watertight, fill_gap) remain usable.",
        ImportWarning, stacklevel=2,
    )
    __all__ = []
