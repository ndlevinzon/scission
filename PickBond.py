"""Interactive 2D bond picker that emits ``restrict_to_bond_smarts`` patterns.

This module backs the ``scission pick-bond`` CLI subcommand. It renders a
ligand in 2D, serves a small localhost web page, and lets the user click two
bonded atoms to build a SMARTS pattern suitable for the fragmentation pipeline's
``restrict_to_bond_smarts`` allow-list. The generated SMARTS marks the two
central bond atoms with atom-map numbers ``:1`` and ``:2`` and supports an
adjustable environment radius so the pattern can be dialed from broad (a class
of bonds) to unique (a single bond). The live match count is computed with the
library's own matcher, :func:`scission.Torsions.match_central_bond_smarts`, so
it reflects exactly what the fragmentation step would select.

RDKit is required; it is an optional ``chem`` extra of this package. The CLI
imports this module lazily so the core CLI stays importable without RDKit.
"""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .LigandIo import load_ligand_from_mol2
from .Torsions import _build_rdkit_mol, match_central_bond_smarts

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None
    Draw = None

#: Largest environment radius offered by the picker. Beyond a handful of bonds
#: the pattern simply spans the whole connected component, so this is clamped.
MAX_RADIUS = 6

#: Canvas size for the 2D depiction, in SVG pixels.
_CANVAS_WIDTH = 760
_CANVAS_HEIGHT = 520


def generate_bond_smarts(mol: "Chem.Mol", atom1: int, atom2: int, radius: int) -> str:
    """Build a ``:1``/``:2``-mapped environment SMARTS for one bond.

    Args:
        mol: RDKit molecule (typically from :func:`_build_rdkit_mol`).
        atom1: Zero-based RDKit index of the first central atom.
        atom2: Zero-based RDKit index of the second central atom.
        radius: Environment radius in bonds. Clamped to ``[0, MAX_RADIUS]``.
            Radius ``0`` yields just the two central atoms and their bond.

    Returns:
        A SMARTS string in which the two central atoms carry atom-map numbers
        ``1`` and ``2``, ready for ``restrict_to_bond_smarts``.

    Raises:
        ValueError: If the two atoms are not directly bonded.
        RuntimeError: If RDKit is unavailable.
    """

    if Chem is None:
        raise RuntimeError("RDKit is required to generate bond SMARTS.")
    atom1 = int(atom1)
    atom2 = int(atom2)
    central = mol.GetBondBetweenAtoms(atom1, atom2)
    if central is None:
        raise ValueError("Selected atoms are not directly bonded.")
    radius = max(0, min(int(radius), MAX_RADIUS))
    atoms: set[int] = {atom1, atom2}
    bonds: set[int] = {central.GetIdx()}
    for center in (atom1, atom2):
        for bond_idx in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center):
            bond = mol.GetBondWithIdx(bond_idx)
            bonds.add(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())
    work = Chem.RWMol(mol)
    work.GetAtomWithIdx(atom1).SetAtomMapNum(1)
    work.GetAtomWithIdx(atom2).SetAtomMapNum(2)
    return Chem.MolFragmentToSmarts(
        work,
        atomsToUse=sorted(atoms),
        bondsToUse=sorted(bonds),
    )


def _render_depiction(mol: "Chem.Mol") -> tuple[str, list[dict], int, int]:
    """Render the molecule to SVG and compute per-atom click hotspots.

    Args:
        mol: RDKit molecule with 2D coordinates already computed.

    Returns:
        A tuple of the inline SVG markup, a list of hotspot records (one per
        atom, carrying the zero-based index, element, atom name, and pixel
        coordinates), and the canvas width and height in pixels.
    """

    drawer = Draw.MolDraw2DSVG(_CANVAS_WIDTH, _CANVAS_HEIGHT)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # Strip the XML prolog so the <svg> element embeds cleanly inside HTML.
    start = svg.find("<svg")
    if start > 0:
        svg = svg[start:]

    hotspots: list[dict] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        point = drawer.GetDrawCoords(idx)
        try:
            name = atom.GetProp("_TriposAtomName")
        except KeyError:
            name = atom.GetSymbol()
        hotspots.append(
            {
                "atomIdx": idx,
                "element": atom.GetSymbol(),
                "atomName": name,
                "px": point.x,
                "py": point.y,
                "isHeavy": atom.GetAtomicNum() > 1,
            }
        )
    return svg, hotspots, _CANVAS_WIDTH, _CANVAS_HEIGHT


def _render_page(
    svg: str,
    hotspots: list[dict],
    width: int,
    height: int,
    title: str,
    initial_radius: int,
) -> str:
    """Substitute rendered data into the static HTML page template."""

    return (
        _PAGE_TEMPLATE.replace("__TITLE__", title)
        .replace("__SVG__", svg)
        .replace("__HOTSPOTS__", json.dumps(hotspots))
        .replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(height))
        .replace("__INIT_RADIUS__", str(initial_radius))
        .replace("__MAX_RADIUS__", str(MAX_RADIUS))
    )


def _make_handler(context: dict):
    """Create a request handler bound to the rendered picker ``context``."""

    class _PickBondHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D401 - silence default logging
            """Suppress per-request stderr logging."""

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path in ("/", "/index.html"):
                body = context["page"].encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/smarts":
                self._handle_smarts()
            elif self.path == "/shutdown":
                self._send_json({"ok": True})
                threading.Thread(
                    target=context["server"].shutdown, daemon=True
                ).start()
            else:
                self.send_response(404)
                self.end_headers()

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def _handle_smarts(self) -> None:
            try:
                data = self._read_json()
                atom1 = int(data["atom1"])
                atom2 = int(data["atom2"])
                radius = int(data.get("radius", 1))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self._send_json({"ok": False, "error": "Invalid request."})
                return
            try:
                smarts = generate_bond_smarts(context["mol"], atom1, atom2, radius)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)})
                return
            matched = sorted(match_central_bond_smarts(context["ligand"], (smarts,)))
            context["last_smarts"] = smarts
            self._send_json(
                {
                    "ok": True,
                    "smarts": smarts,
                    "match_count": len(matched),
                    "matched_bonds": [list(pair) for pair in matched],
                    "matched_bonds_rdkit": [[a - 1, b - 1] for a, b in matched],
                    "radius": max(0, min(radius, MAX_RADIUS)),
                    "is_unique": len(matched) == 1,
                }
            )

    return _PickBondHandler


def run_pick_bond(
    mol2_path: Path | str,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    initial_radius: int = 1,
) -> int:
    """Launch the interactive bond-picker web app for a MOL2 ligand.

    Args:
        mol2_path: Path to the MOL2 file to depict.
        host: Interface to bind. Defaults to localhost only.
        port: TCP port, or ``0`` to auto-pick a free port.
        open_browser: Whether to open the page in a browser automatically.
        initial_radius: Starting environment radius for generated SMARTS.

    Returns:
        A process exit code (``0`` on a clean shutdown, ``2`` on a startup
        error such as missing RDKit or an unreadable MOL2).
    """

    if Chem is None or AllChem is None or Draw is None:
        print(
            "pick-bond requires RDKit. Install it with: pip install 'scission[chem]'",
            file=sys.stderr,
        )
        return 2

    mol2_path = Path(mol2_path)
    try:
        ligand = load_ligand_from_mol2(mol2_path)
    except (OSError, ValueError) as exc:
        print(f"Failed to load {mol2_path}: {exc}", file=sys.stderr)
        return 2

    mol = _build_rdkit_mol(ligand)
    AllChem.Compute2DCoords(mol)
    svg, hotspots, width, height = _render_depiction(mol)
    initial_radius = max(0, min(int(initial_radius), MAX_RADIUS))
    page = _render_page(svg, hotspots, width, height, mol2_path.name, initial_radius)

    context: dict = {"ligand": ligand, "mol": mol, "page": page, "last_smarts": None}
    try:
        server = ThreadingHTTPServer((host, port), _make_handler(context))
    except OSError as exc:
        print(f"Could not start server on {host}:{port}: {exc}", file=sys.stderr)
        return 2
    context["server"] = server

    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"scission pick-bond serving at {url}")
    print(
        "Click two bonded atoms, adjust the radius until the match count suits "
        "you, then copy the SMARTS. Press Ctrl-C (or the Done button) to stop."
    )
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - environment dependent
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    if context["last_smarts"]:
        print(f"\nSelected SMARTS: {context['last_smarts']}")
    return 0


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>scission pick-bond &mdash; __TITLE__</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 24px; color: #1b1f24; background: #f6f7f9; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: #57606a; font-size: 13px; margin: 0 0 16px; }
  .layout { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  .card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; }
  #stage { position: relative; width: __WIDTH__px; height: __HEIGHT__px; }
  #mol svg { display: block; }
  #overlay { position: absolute; left: 0; top: 0; pointer-events: none; }
  #spots { position: absolute; left: 0; top: 0; width: __WIDTH__px; height: __HEIGHT__px; }
  .spot { position: absolute; width: 24px; height: 24px; margin: -12px 0 0 -12px;
          border-radius: 50%; cursor: pointer; }
  .spot.light { background: rgba(0,0,0,0.02); }
  .spot.sel { background: rgba(255,140,0,0.35); box-shadow: 0 0 0 2px #ff8c00; }
  .panel { width: 340px; }
  .panel label { font-size: 13px; font-weight: 600; }
  .row { margin-bottom: 16px; }
  code, .smarts { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .smarts { display: block; background: #f6f8fa; border: 1px solid #d0d7de;
            border-radius: 6px; padding: 10px; word-break: break-all; font-size: 14px;
            min-height: 20px; }
  .count { font-size: 14px; margin-top: 8px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
           font-size: 12px; font-weight: 600; }
  .badge.unique { background: #dafbe1; color: #1a7f37; }
  .badge.multi { background: #fff1e5; color: #9a6700; }
  .badge.none { background: #ffebe9; color: #cf222e; }
  .err { color: #cf222e; font-size: 13px; min-height: 16px; }
  input[type=range] { width: 100%; }
  button { font-size: 13px; padding: 7px 12px; border-radius: 6px; cursor: pointer;
           border: 1px solid #d0d7de; background: #f6f8fa; }
  button.primary { background: #1f883d; color: #fff; border-color: #1a7f37; }
  .hint { color: #57606a; font-size: 12px; }
</style>
</head>
<body>
  <h1>Pick a central bond</h1>
  <p class="sub">__TITLE__ &mdash; click two bonded atoms, then tune the radius for
     a broad or unique SMARTS.</p>
  <div class="layout">
    <div class="card">
      <div id="stage">
        <div id="mol">__SVG__</div>
        <svg id="overlay" width="__WIDTH__" height="__HEIGHT__"></svg>
        <div id="spots"></div>
      </div>
    </div>
    <div class="card panel">
      <div class="row">
        <div id="status" class="hint">No bond selected yet.</div>
        <div id="error" class="err"></div>
      </div>
      <div class="row">
        <label for="radius">Environment radius: <span id="radiusVal">__INIT_RADIUS__</span></label>
        <input type="range" id="radius" min="0" max="__MAX_RADIUS__" value="__INIT_RADIUS__">
        <div class="hint">0 = just the two atoms &amp; bond order. Higher = more
            chemical context (more unique).</div>
      </div>
      <div class="row">
        <label>restrict_to_bond_smarts</label>
        <code id="smarts" class="smarts"></code>
        <div class="count"><span id="count"></span> <span id="badge"></span></div>
      </div>
      <div class="row">
        <button id="copy">Copy SMARTS</button>
        <button id="done" class="primary">Done</button>
      </div>
      <div class="hint">Feed the copied pattern to
          <code>scission fragment ... --restrict-bond-smarts '&lt;pattern&gt;'</code>.</div>
    </div>
  </div>
<script>
const HOTSPOTS = __HOTSPOTS__;
const coords = {};
HOTSPOTS.forEach(h => { coords[h.atomIdx] = h; });
let selected = [];

const spots = document.getElementById('spots');
const overlay = document.getElementById('overlay');
const SVGNS = 'http://www.w3.org/2000/svg';

HOTSPOTS.forEach(h => {
  const d = document.createElement('div');
  d.className = 'spot' + (h.isHeavy ? '' : ' light');
  d.style.left = h.px + 'px';
  d.style.top = h.py + 'px';
  d.title = h.atomName + ' (' + h.element + ')';
  d.dataset.idx = h.atomIdx;
  d.addEventListener('click', () => toggle(h.atomIdx));
  spots.appendChild(d);
});

function spotEl(idx) {
  return spots.querySelector('.spot[data-idx="' + idx + '"]');
}

function toggle(idx) {
  if (selected.includes(idx)) {
    selected = selected.filter(i => i !== idx);
  } else if (selected.length >= 2) {
    selected = [idx];
  } else {
    selected.push(idx);
  }
  paintSelection();
  if (selected.length === 2) {
    update();
  } else {
    clearResult(selected.length === 0
      ? 'No bond selected yet.'
      : 'Pick one more atom to define the bond.');
  }
}

function paintSelection() {
  spots.querySelectorAll('.spot').forEach(s => s.classList.remove('sel'));
  selected.forEach(i => { const e = spotEl(i); if (e) e.classList.add('sel'); });
}

function clearResult(msg) {
  document.getElementById('status').textContent = msg;
  document.getElementById('error').textContent = '';
  document.getElementById('smarts').textContent = '';
  document.getElementById('count').textContent = '';
  document.getElementById('badge').textContent = '';
  document.getElementById('badge').className = 'badge';
  drawBonds([]);
}

function drawBonds(pairs) {
  while (overlay.firstChild) overlay.removeChild(overlay.firstChild);
  // The user-selected bond, drawn first (orange).
  if (selected.length === 2) {
    line(coords[selected[0]], coords[selected[1]], '#ff8c00', 4, 1.0);
  }
  // All matched bonds (green), so the user sees how broad the pattern is.
  pairs.forEach(p => {
    const a = coords[p[0]], b = coords[p[1]];
    if (a && b) line(a, b, '#1a7f37', 6, 0.35);
  });
}

function line(a, b, color, w, opacity) {
  const ln = document.createElementNS(SVGNS, 'line');
  ln.setAttribute('x1', a.px); ln.setAttribute('y1', a.py);
  ln.setAttribute('x2', b.px); ln.setAttribute('y2', b.py);
  ln.setAttribute('stroke', color);
  ln.setAttribute('stroke-width', w);
  ln.setAttribute('stroke-linecap', 'round');
  ln.setAttribute('stroke-opacity', opacity);
  overlay.appendChild(ln);
}

function update() {
  const radius = parseInt(document.getElementById('radius').value, 10);
  document.getElementById('status').textContent =
    'Bond: ' + coords[selected[0]].atomName + ' \\u2013 ' + coords[selected[1]].atomName;
  fetch('/smarts', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({atom1: selected[0], atom2: selected[1], radius: radius})
  }).then(r => r.json()).then(render).catch(e => {
    document.getElementById('error').textContent = 'Request failed: ' + e;
  });
}

function render(resp) {
  const err = document.getElementById('error');
  if (!resp.ok) {
    err.textContent = resp.error || 'Could not build a SMARTS for that pair.';
    document.getElementById('smarts').textContent = '';
    document.getElementById('count').textContent = '';
    document.getElementById('badge').textContent = '';
    drawBonds([]);
    return;
  }
  err.textContent = '';
  document.getElementById('smarts').textContent = resp.smarts;
  document.getElementById('count').textContent =
    'Matches ' + resp.match_count + ' bond' + (resp.match_count === 1 ? '' : 's')
    + ' in this molecule.';
  const badge = document.getElementById('badge');
  if (resp.match_count === 1) { badge.textContent = 'unique'; badge.className = 'badge unique'; }
  else if (resp.match_count === 0) { badge.textContent = 'no match'; badge.className = 'badge none'; }
  else { badge.textContent = resp.match_count + ' matches'; badge.className = 'badge multi'; }
  drawBonds(resp.matched_bonds_rdkit || []);
}

document.getElementById('radius').addEventListener('input', e => {
  document.getElementById('radiusVal').textContent = e.target.value;
  if (selected.length === 2) update();
});

document.getElementById('copy').addEventListener('click', () => {
  const text = document.getElementById('smarts').textContent;
  if (text) navigator.clipboard.writeText(text);
});

document.getElementById('done').addEventListener('click', () => {
  fetch('/shutdown', {method: 'POST'}).finally(() => {
    document.body.innerHTML =
      '<h1>Done</h1><p class="sub">The picker has stopped. You can close this tab.</p>';
  });
});

clearResult('No bond selected yet.');
</script>
</body>
</html>
"""
