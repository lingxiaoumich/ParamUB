"""Generate a self-contained HTML labeling UI for the batch_v5 renders.

Produces outputs/batch_v5_summary/label_ui.html — open it directly in a
browser (no server needed). It references the same per-view PNGs the
contact sheets use:
    outputs/batch_v5_summary/views/<view_slug>/<car>.png

Features
  * 3 tabs, one per view (bottom_front_iso / bottom / bottom_rear_iso).
  * Per-CAR label (shared across all 3 views): selecting a problem in any
    view marks that car red in every view.
  * Labels: good (default) / bad - reverse in x / bad - dirty underbody /
    bad - other.
  * Auto-saves to the browser's localStorage (survives reload).
  * "Show problematic only" toggle to pick out the bad ones.
  * Export buttons: problematic CSV, and full labels CSV/JSON.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "batch_v5_summary"
VIEWS = ["bottom_front_iso", "bottom", "bottom_rear_iso"]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParamUB v5 — model labeling</title>
<style>
  :root {{ --bad:#d62828; --good-bd:#d0d6dc; --bg:#f3f4f6; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:#1d2530; }}
  header {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid #d0d6dc;
           padding:10px 16px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
  h1 {{ font-size:16px; margin:0 0 8px; }}
  .tabs {{ display:flex; gap:6px; }}
  .tab {{ padding:7px 14px; border:1px solid #c4ccd4; border-radius:6px 6px 0 0;
         background:#e9edf1; cursor:pointer; font-size:13px; user-select:none; }}
  .tab.active {{ background:#3b6fa6; color:#fff; border-color:#3b6fa6; }}
  .toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:8px; font-size:13px; }}
  .toolbar button {{ padding:6px 12px; border:1px solid #c4ccd4; border-radius:6px;
                    background:#fff; cursor:pointer; font-size:13px; }}
  .toolbar button:hover {{ background:#eef2f6; }}
  .toolbar .counts {{ margin-left:auto; font-variant-numeric:tabular-nums; }}
  .counts b {{ color:var(--bad); }}
  .grid {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:12px; padding:16px; }}
  .tile {{ background:#fff; border:2px solid var(--good-bd); border-radius:8px; padding:6px;
          display:flex; flex-direction:column; }}
  .tile.bad {{ border-color:var(--bad); box-shadow:0 0 0 1px var(--bad) inset; }}
  .tile img {{ width:100%; height:auto; border-radius:4px; background:#fafbfc; }}
  .tile .name {{ font-size:12px; font-weight:600; margin:5px 2px 6px; display:flex;
                justify-content:space-between; align-items:center; }}
  .tile .name .wt {{ font-weight:400; color:#7a8694; font-size:10px; }}
  .labels {{ display:grid; grid-template-columns:1fr 1fr; gap:4px; }}
  .labels button {{ font-size:11px; padding:5px 4px; border:1px solid #c4ccd4; border-radius:5px;
                   background:#f7f9fb; cursor:pointer; line-height:1.15; }}
  .labels button:hover {{ background:#eef2f6; }}
  .labels button.sel-good {{ background:#2a9d3f; color:#fff; border-color:#2a9d3f; }}
  .labels button.sel-bad  {{ background:var(--bad); color:#fff; border-color:var(--bad); }}
  .hidden {{ display:none !important; }}
  footer {{ padding:14px 16px 40px; color:#7a8694; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>ParamUB v5 — model labeling &nbsp;<span style="font-weight:400;color:#7a8694">({n_cars} cars)</span></h1>
  <div class="tabs" id="tabs"></div>
  <div class="toolbar">
    <button id="filterBtn">Show problematic only</button>
    <button id="resetBtn">Reset all to good</button>
    <button id="expBad">Export problematic (CSV)</button>
    <button id="expAll">Export all (CSV)</button>
    <button id="expJson">Export JSON</button>
    <span class="counts" id="counts"></span>
  </div>
</header>
<div class="grid" id="grid"></div>
<footer>Labels auto-save in this browser (localStorage key <code>{storage_key}</code>).
  Selecting any "bad" label marks the car red in all three views.</footer>

<script>
const CARS = {cars_json};
const VIEWS = {views_json};
const LABELS = [
  {{id:"good",            txt:"good",                 bad:false}},
  {{id:"bad_reverse_x",   txt:"bad — reverse in x",   bad:true }},
  {{id:"bad_dirty",       txt:"bad — dirty underbody",bad:true }},
  {{id:"bad_other",       txt:"bad — other",          bad:true }},
];
const STORAGE_KEY = "{storage_key}";
const WT = {wt_json};
const IMG = {img_json};   // {{car: {{view: dataURI}}}} when embedded, else {{}}
function imgSrc(car, view) {{
  if (IMG[car] && IMG[car][view]) return IMG[car][view];
  return `views/${{view}}/${{car}}.png`;
}}

let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{}}; }} catch(e) {{ state = {{}}; }}
let curView = VIEWS[0];
let filterBad = false;

function labelOf(car) {{ return state[car] || "good"; }}
function isBad(car) {{ const l = labelOf(car); return l !== "good"; }}
function save() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }}

function setLabel(car, lab) {{
  if (lab === "good") delete state[car]; else state[car] = lab;
  save();
  // update every tile for this car across the (single rendered) view + counts
  const tile = document.querySelector(`.tile[data-car="${{car}}"]`);
  if (tile) refreshTile(tile, car);
  updateCounts();
  if (filterBad) applyFilter();
}}

function refreshTile(tile, car) {{
  const bad = isBad(car);
  tile.classList.toggle("bad", bad);
  const cur = labelOf(car);
  tile.querySelectorAll(".labels button").forEach(b => {{
    const sel = b.dataset.lab === cur;
    b.classList.toggle("sel-good", sel && cur === "good");
    b.classList.toggle("sel-bad",  sel && cur !== "good");
  }});
}}

function buildGrid() {{
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const car of CARS) {{
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.dataset.car = car;
    const wt = WT[car] ? ` <span class="wt">[${{WT[car]}}]</span>` : "";
    let html = `<div class="name"><span>${{car}}</span><span>${{wt}}</span></div>`;
    html += `<img loading="lazy" src="${{imgSrc(car, curView)}}" alt="${{car}}">`;
    html += `<div class="labels">`;
    for (const L of LABELS) html += `<button data-lab="${{L.id}}">${{L.txt}}</button>`;
    html += `</div>`;
    tile.innerHTML = html;
    tile.querySelectorAll(".labels button").forEach(b => {{
      b.addEventListener("click", () => setLabel(car, b.dataset.lab));
    }});
    refreshTile(tile, car);
    grid.appendChild(tile);
  }}
  applyFilter();
}}

function applyFilter() {{
  document.querySelectorAll(".tile").forEach(t => {{
    const hide = filterBad && !isBad(t.dataset.car);
    t.classList.toggle("hidden", hide);
  }});
}}

function updateCounts() {{
  const bad = CARS.filter(isBad).length;
  document.getElementById("counts").innerHTML =
    `good: ${{CARS.length - bad}} &nbsp; | &nbsp; <b>problematic: ${{bad}}</b> &nbsp; / ${{CARS.length}}`;
}}

function buildTabs() {{
  const tabs = document.getElementById("tabs");
  VIEWS.forEach(v => {{
    const t = document.createElement("div");
    t.className = "tab" + (v === curView ? " active" : "");
    t.textContent = v.replace(/_/g, " ");
    t.addEventListener("click", () => {{
      curView = v;
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      // swap image src only (keep state) — cheaper than full rebuild
      document.querySelectorAll(".tile").forEach(tile => {{
        const car = tile.dataset.car;
        tile.querySelector("img").src = imgSrc(car, curView);
      }});
    }});
    tabs.appendChild(t);
  }});
}}

function downloadFile(name, text, type) {{
  const blob = new Blob([text], {{type}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}}

document.getElementById("filterBtn").addEventListener("click", (e) => {{
  filterBad = !filterBad;
  e.target.textContent = filterBad ? "Show all" : "Show problematic only";
  applyFilter();
}});
document.getElementById("resetBtn").addEventListener("click", () => {{
  if (!confirm("Reset all labels to good?")) return;
  state = {{}}; save(); buildGrid(); updateCounts();
}});
document.getElementById("expBad").addEventListener("click", () => {{
  const rows = ["car,label"];
  CARS.filter(isBad).forEach(c => rows.push(`${{c}},${{labelOf(c)}}`));
  downloadFile("problematic_v5.csv", rows.join("\\n"), "text/csv");
}});
document.getElementById("expAll").addEventListener("click", () => {{
  const rows = ["car,label"];
  CARS.forEach(c => rows.push(`${{c}},${{labelOf(c)}}`));
  downloadFile("labels_v5.csv", rows.join("\\n"), "text/csv");
}});
document.getElementById("expJson").addEventListener("click", () => {{
  const obj = {{}};
  CARS.forEach(c => obj[c] = labelOf(c));
  downloadFile("labels_v5.json", JSON.stringify(obj, null, 2), "application/json");
}});

buildTabs();
buildGrid();
updateCounts();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--batch-out", type=Path,
                    default=REPO_ROOT / "outputs" / "batch_v5")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HTML (default <summary-dir>/label_ui.html, "
                         "or label_ui_embedded.html when --embed).")
    ap.add_argument("--embed", action="store_true",
                    help="Inline every image as a base64 JPEG so the single "
                         "HTML file is fully portable (no views/ folder "
                         "needed). Recommended for downloading.")
    ap.add_argument("--embed-width", type=int, default=384,
                    help="Downsample width (px) for embedded images. Default 384.")
    ap.add_argument("--embed-quality", type=int, default=72,
                    help="JPEG quality for embedded images. Default 72.")
    args = ap.parse_args()

    summary = args.summary_dir
    # Car list = intersection of cars present in all 3 view dirs (sorted).
    per_view = {}
    for v in VIEWS:
        d = summary / "views" / v
        per_view[v] = {p.stem for p in d.glob("*.png")} if d.is_dir() else set()
    cars = sorted(set.intersection(*per_view.values())) if all(per_view.values()) else \
        sorted(set.union(*per_view.values()))
    print(f"[label-ui] cars present in all views: {len(cars)}")
    for v in VIEWS:
        print(f"           {v}: {len(per_view[v])} pngs")

    # Watertight flags (best-effort).
    wt = {}
    for c in cars:
        rep = args.batch_out / c / "integrate" / f"{c}_clean.json"
        if rep.is_file():
            try:
                d = json.loads(rep.read_text())
                w = d.get("is_watertight")
                wt[c] = "wt" if w else "open" if w is False else ""
            except (json.JSONDecodeError, OSError):
                pass

    # Optionally inline every image as a base64 JPEG for a portable file.
    img = {}
    if args.embed:
        from PIL import Image
        print(f"[label-ui] embedding images (w={args.embed_width}, "
              f"q={args.embed_quality}) ...")
        for n, c in enumerate(cars, 1):
            img[c] = {}
            for v in VIEWS:
                p = summary / "views" / v / f"{c}.png"
                if not p.is_file():
                    continue
                im = Image.open(p).convert("RGB")
                h = round(im.height * args.embed_width / im.width)
                im = im.resize((args.embed_width, h), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=args.embed_quality, optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                img[c][v] = f"data:image/jpeg;base64,{b64}"
            if n % 100 == 0:
                print(f"           {n}/{len(cars)} cars embedded")

    html = HTML_TEMPLATE.format(
        n_cars=len(cars),
        cars_json=json.dumps(cars),
        views_json=json.dumps(VIEWS),
        wt_json=json.dumps(wt),
        img_json=json.dumps(img),
        storage_key="paramub_v5_labels",
    )
    default_name = "label_ui_embedded.html" if args.embed else "label_ui.html"
    out = args.out or (summary / default_name)
    out.write_text(html)
    mb = out.stat().st_size / (1024 * 1024)
    print(f"[label-ui] wrote {out}  ({mb:.1f} MB, "
          f"{'embedded' if args.embed else 'references views/'})")
    print(f"[label-ui] open in a browser:  file://{out.resolve()}")


if __name__ == "__main__":
    main()
