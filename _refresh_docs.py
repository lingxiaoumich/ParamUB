"""Re-run only the auto-generated docs (parameter reference + outputs/README).

Used after a catalog run if the docs need to pick up changes that landed
after the catalog process started. Reads the existing summary.json so we
don't re-run any geometry.
"""

import json
from pathlib import Path

import tire_assembly as ta
import generate_tire_catalog as gtc


def main():
    spec = gtc.merged_spec()
    out_path = gtc.generate_parameter_reference(spec)
    print(f"refreshed {out_path}")

    summary_json = gtc.CATALOG_DIR / "summary.json"
    if not summary_json.exists():
        print(f"no summary at {summary_json} — skipping README refresh")
        return 0
    summary_rows = json.loads(summary_json.read_text())

    example_meta = gtc.EXAMPLE_DIR / "metadata.json"
    if example_meta.exists():
        example_row = json.loads(example_meta.read_text())
        example_row.setdefault("preview_path", str(gtc.EXAMPLE_DIR / "preview.png"))
        example_row.setdefault("stl_path", str(gtc.EXAMPLE_DIR / "tire_assembly.stl"))
    else:
        example_row = {}

    doc_paths = [
        gtc.DOCS_DIR / "example_tunable_map.png",
        gtc.DOCS_DIR / "section_cut_labeled.png",
        gtc.DOCS_DIR / "spoke_window_labeled.png",
        gtc.DOCS_DIR / "tunable_dimensions.md",
    ]
    gtc.write_outputs_readme(example_row, doc_paths, summary_rows)
    print(f"refreshed {gtc.OUTPUT_ROOT / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
