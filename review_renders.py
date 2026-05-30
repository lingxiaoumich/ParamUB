"""Tiny browser-based reviewer for the v2 10-view renders.

Serves a single-page UI on a local port. Each car's 10-view montage
is shown one at a time with four buttons:

    Good (g)   Bad (b)   Further inspection (f)   Skip (s)   Back (left arrow)

Each click POSTs the verdict to the server, which appends to
``outputs/summary/review_log.csv`` and advances to the next
unclassified car. The log is append-only and resumable: re-running
the tool picks up where you left off and skips already-classified
cars (Skip is recorded too, so a Skip won't keep cycling).

Usage:
    python review_renders.py                    # port 5057, default dirs
    python review_renders.py --port 8080
    python review_renders.py --renders <dir> --log <csv>

If VSCode's remote tunnel is active, the port is auto-forwarded and
the URL is clickable.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import socketserver
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = Path(__file__).resolve().parent

# Legend used in CSV + UI. Keys match keyboard shortcuts.
LABELS = {
    "good": "Good",
    "bad": "Bad",
    "further": "Further inspection",
    "skip": "Skip",
}

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ParamUB render review</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         margin: 0; padding: 12px 18px; background: #1f2937; color: #e5e7eb; }
  header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
           border-bottom: 1px solid #374151; padding-bottom: 8px;
           margin-bottom: 10px; }
  header h1 { font-size: 16px; font-weight: 600; margin: 0; color: #f3f4f6; }
  .car { font-family: ui-monospace, "SF Mono", Menlo, monospace;
         color: #93c5fd; font-size: 15px; }
  .meta { color: #9ca3af; }
  .img-wrap { background: #f3f4f6; border-radius: 6px; padding: 6px;
              display: block; text-align: center; }
  .img-wrap img { max-width: 100%; height: auto; display: block;
                  margin: 0 auto; }
  .btns { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap;
          justify-content: center; }
  button { font: 600 14px/1 system-ui, sans-serif; padding: 12px 22px;
           border-radius: 6px; border: 1px solid transparent; cursor: pointer;
           min-width: 150px; transition: transform 0.05s; }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: 0.5; cursor: default; }
  .good    { background: #16a34a; color: white; }
  .bad     { background: #dc2626; color: white; }
  .further { background: #d97706; color: white; }
  .skip    { background: #6b7280; color: white; }
  .back    { background: #374151; color: #d1d5db; min-width: 80px; }
  kbd { background: #111827; color: #f3f4f6; padding: 1px 5px;
        border-radius: 3px; font: 11px/1 ui-monospace, monospace;
        border: 1px solid #4b5563; }
  .status { color: #9ca3af; margin-top: 10px; min-height: 18px;
            text-align: center; }
  .done { text-align: center; padding: 40px; color: #93c5fd; }
</style>
</head>
<body>
<header>
  <h1>ParamUB render review</h1>
  <span class="meta">Progress: <strong id="progress">--</strong> /
    <strong id="total">--</strong>
    (<span id="remaining">--</span> unclassified)</span>
</header>
<div class="meta" style="margin-bottom:6px;">
  Car: <span class="car" id="car-name">loading...</span>
</div>
<div id="img-wrap" class="img-wrap"><img id="render" alt="render"></div>
<div class="btns">
  <button class="back"    id="btn-back"    onclick="back()">&larr; Back</button>
  <button class="good"    id="btn-good"    onclick="classify('good')">Good <kbd>g</kbd></button>
  <button class="bad"     id="btn-bad"     onclick="classify('bad')">Bad <kbd>b</kbd></button>
  <button class="further" id="btn-further" onclick="classify('further')">Further inspection <kbd>f</kbd></button>
  <button class="skip"    id="btn-skip"    onclick="classify('skip')">Skip <kbd>s</kbd></button>
</div>
<div class="status" id="status">Loading...</div>
<script>
  let current = null;   // {car, idx, total, remaining}
  let history = [];     // car names visited this session (for back)
  function setBusy(b) {
    for (const id of ['btn-good','btn-bad','btn-further','btn-skip','btn-back']) {
      document.getElementById(id).disabled = b;
    }
  }
  async function load(car) {
    setBusy(true);
    const url = car ? '/api/next?car=' + encodeURIComponent(car) : '/api/next';
    const r = await fetch(url);
    const data = await r.json();
    if (data.done) {
      document.querySelector('header').style.display = 'none';
      document.querySelector('.img-wrap').style.display = 'none';
      document.querySelector('.btns').style.display = 'none';
      document.getElementById('status').innerHTML =
        '<div class="done">All ' + data.total + ' cars reviewed. ' +
        'Log at ' + data.log + '</div>';
      return;
    }
    current = data;
    document.getElementById('car-name').textContent = data.car;
    document.getElementById('progress').textContent = (data.idx + 1);
    document.getElementById('total').textContent = data.total;
    document.getElementById('remaining').textContent = data.remaining;
    document.getElementById('render').src = '/img/' + data.car +
                                            '?t=' + Date.now();
    document.getElementById('status').textContent = '';
    setBusy(false);
  }
  async function classify(label) {
    if (!current) return;
    setBusy(true);
    document.getElementById('status').textContent = 'Saving "' + label + '"...';
    history.push(current.car);
    if (history.length > 50) history.shift();
    const r = await fetch('/api/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({car: current.car, label})
    });
    if (!r.ok) {
      document.getElementById('status').textContent = 'Save failed.';
      setBusy(false);
      return;
    }
    load();
  }
  async function back() {
    if (history.length === 0) {
      document.getElementById('status').textContent = 'No previous car this session.';
      return;
    }
    const prev = history.pop();
    await fetch('/api/unclassify?car=' + encodeURIComponent(prev), {method: 'POST'});
    load(prev);
  }
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'g') classify('good');
    else if (e.key === 'b') classify('bad');
    else if (e.key === 'f') classify('further');
    else if (e.key === 's') classify('skip');
    else if (e.key === 'ArrowLeft') back();
  });
  load();
</script>
</body>
</html>
"""


class Reviewer:
    def __init__(self, renders_dir: Path, log_path: Path):
        self.renders_dir = renders_dir
        self.log_path = log_path
        pngs = sorted(renders_dir.glob("*_10view.png"))
        self.cars = [p.stem[: -len("_10view")] for p in pngs]
        self._lock = threading.Lock()
        self._classified: set[str] = set()
        if log_path.is_file():
            with log_path.open() as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and row[0] != "car":
                        self._classified.add(row[0])
        else:
            with log_path.open("w", newline="") as f:
                csv.writer(f).writerow(["car", "label", "timestamp_iso"])
        print(f"[review] {len(self.cars)} cars total, "
              f"{len(self._classified)} already classified, "
              f"{len(self.cars) - len(self._classified)} to go",
              flush=True)

    def next_car(self, prefer: str | None = None) -> str | None:
        with self._lock:
            if prefer and prefer in self.cars:
                return prefer
            for c in self.cars:
                if c not in self._classified:
                    return c
            return None

    def status(self) -> tuple[int, int, int]:
        with self._lock:
            remaining = sum(1 for c in self.cars if c not in self._classified)
            return len(self.cars), len(self._classified), remaining

    def idx_of(self, car: str) -> int:
        try:
            return self.cars.index(car)
        except ValueError:
            return -1

    def classify(self, car: str, label: str) -> bool:
        if label not in LABELS:
            return False
        with self._lock:
            with self.log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [car, label, time.strftime("%Y-%m-%dT%H:%M:%S")])
            self._classified.add(car)
        return True

    def unclassify(self, car: str) -> None:
        # Just drop from the in-memory set; the CSV stays append-only so
        # the audit log keeps both rows. The latest verdict wins on
        # re-load (set-membership) but back/redo still works in-session.
        with self._lock:
            self._classified.discard(car)

    def render_path(self, car: str) -> Path | None:
        p = self.renders_dir / f"{car}_10view.png"
        return p if p.is_file() else None


def make_handler(reviewer: Reviewer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # silence default access log

        def _send_json(self, payload, code=HTTPStatus.OK):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text, content_type="text/html; charset=utf-8",
                       code=HTTPStatus.OK):
            body = text.encode("utf-8") if isinstance(text, str) else text
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, p: Path, content_type="image/png"):
            data = p.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                self._send_text(INDEX_HTML)
                return
            if url.path == "/api/next":
                q = parse_qs(url.query)
                prefer = q.get("car", [None])[0]
                car = reviewer.next_car(prefer)
                total, classified, remaining = reviewer.status()
                if car is None:
                    self._send_json({"done": True, "total": total,
                                     "log": str(reviewer.log_path)})
                    return
                self._send_json({
                    "car": car,
                    "idx": reviewer.idx_of(car),
                    "total": total,
                    "remaining": remaining,
                    "done": False,
                })
                return
            if url.path.startswith("/img/"):
                car = url.path[len("/img/"):]
                p = reviewer.render_path(car)
                if p is None:
                    self.send_error(HTTPStatus.NOT_FOUND, f"no render for {car}")
                    return
                self._send_file(p)
                return
            self.send_error(HTTPStatus.NOT_FOUND, html.escape(url.path))

        def do_POST(self):
            url = urlsplit(self.path)
            if url.path == "/api/classify":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                ok = reviewer.classify(payload.get("car", ""),
                                        payload.get("label", ""))
                self._send_json({"ok": ok},
                                 HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST)
                return
            if url.path == "/api/unclassify":
                q = parse_qs(url.query)
                car = q.get("car", [""])[0]
                reviewer.unclassify(car)
                self._send_json({"ok": True})
                return
            self.send_error(HTTPStatus.NOT_FOUND, html.escape(url.path))

    return H


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", type=Path,
                    default=REPO_ROOT / "outputs" / "summary" / "renders",
                    help="directory of <car>_10view.png files")
    ap.add_argument("--log", type=Path,
                    default=REPO_ROOT / "outputs" / "summary" / "review_log.csv",
                    help="append-only CSV log of verdicts")
    ap.add_argument("--port", type=int, default=5057)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind host (default 127.0.0.1)")
    args = ap.parse_args()

    if not args.renders.is_dir():
        ap.error(f"renders dir not found: {args.renders}")
    reviewer = Reviewer(args.renders, args.log)

    server = _ThreadingHTTPServer((args.host, args.port),
                                   make_handler(reviewer))
    url = f"http://{args.host}:{args.port}/"
    print(f"[review] serving at {url}", flush=True)
    print(f"[review] log: {args.log}", flush=True)
    print(f"[review] Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[review] stopped.", flush=True)


if __name__ == "__main__":
    main()
