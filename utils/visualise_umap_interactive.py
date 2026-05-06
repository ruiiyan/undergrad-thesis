"""
visualise_umap_interactive.py
------------------------------
Canvas-based interactive UMAP.  No Plotly — vanilla JS click handling.
- Click a dot  → side panel shows metadata + STAR sections + Bloom scores
- Click section header → Bloom justification toggles
- Scroll to zoom, drag to pan
- Legend checkboxes toggle clusters
"""

import json
import numpy as np
import umap

EMBEDDINGS_FILE = "reflection_embeddings.npy"
CLUSTERED_FILE  = "reflections_clustered_final.json"
OUTPUT_HTML     = "umap_interactive.html"

UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42)

CLUSTER_LABELS = {
    -1: "Unclustered",
     0: "Coding problem-solving",
     1: "Technical fabrication",
     2: "Technical drawing",
     3: "RGM design and ideation",
     4: "Team communication",
     5: "Project planning & time mgmt",
     6: "Group presentation",
     7: "Critical thinking",
     8: "Collaborative design non-RGM",
     9: "Conceptual solution development",
}

PALETTE = {
    -1: "#aaaaaa",
     0: "#e6194b",
     1: "#f58231",
     2: "#b8a800",
     3: "#3cb44b",
     4: "#00b4d8",
     5: "#4363d8",
     6: "#911eb4",
     7: "#f032e6",
     8: "#7b7bff",
     9: "#9a6324",
}

# ---------------------------------------------------------------------------
print("[1] Loading data...")
embeddings = np.load(EMBEDDINGS_FILE)
with open(CLUSTERED_FILE, "r", encoding="utf-8") as f:
    reflections = json.load(f)
assert len(embeddings) == len(reflections)
print(f"  {len(reflections)} reflections loaded")

# ---------------------------------------------------------------------------
print("[2] Running UMAP to 2D...")
reducer   = umap.UMAP(**UMAP_PARAMS)
projected = reducer.fit_transform(embeddings)
print(f"  Done — shape {projected.shape}")

# ---------------------------------------------------------------------------
print("[3] Preparing JS payload...")

points = []
for i, r in enumerate(reflections):
    bloom = r.get("bloom") or {}
    points.append({
        "idx":   i,
        "x":     float(projected[i, 0]),
        "y":     float(projected[i, 1]),
        "fid":   r["final_cluster_id"],
        "id":    r["id"],
        "band":  r.get("band", "?"),
        "score": r.get("score", "?"),
        "final_cluster_theme": r["final_cluster_theme"],
        "src_cluster_id":      r.get("cluster_id", "?"),
        "src_cluster_theme":   r.get("cluster_theme", "?"),
        "bloom_weighted":  round((bloom.get("weighted_score") or 0), 3),
        "situation":   r.get("situation", ""),
        "task_action": r.get("task_action", ""),
        "result":      r.get("result", ""),
        "bloom_situation":   bloom.get("situation", {}),
        "bloom_task_action": bloom.get("task_action", {}),
        "bloom_result":      bloom.get("result", {}),
    })

clusters_meta = [
    {"fid": fid, "label": CLUSTER_LABELS[fid], "color": PALETTE[fid]}
    for fid in sorted(CLUSTER_LABELS.keys())
]

points_json   = json.dumps(points,         ensure_ascii=False)
clusters_json = json.dumps(clusters_meta,  ensure_ascii=False)

# ---------------------------------------------------------------------------
print("[4] Writing HTML...")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UMAP — Student Reflections</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f0f0f0;
  display: flex;
  height: 100vh;
  overflow: hidden;
}}

/* ── Left: canvas + legend ── */
#left {{
  display: flex;
  flex-direction: column;
  flex: 1 1 62%;
  min-width: 0;
  padding: 10px;
  gap: 8px;
}}

#canvas-wrap {{
  position: relative;
  flex: 1 1 auto;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.12);
  overflow: hidden;
}}
canvas {{
  display: block;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}}
#tooltip {{
  position: absolute;
  background: rgba(30,30,30,.85);
  color: #fff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  white-space: nowrap;
  display: none;
}}
#hint {{
  position: absolute;
  bottom: 8px; left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  color: #999;
  pointer-events: none;
}}

#search-bar {{
  display: flex;
  gap: 6px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.12);
  padding: 8px 12px;
  align-items: center;
}}
#search-bar input {{
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 13px;
  outline: none;
}}
#search-bar input:focus {{ border-color: #4363d8; }}
#search-bar button {{
  background: #4363d8;
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 6px 14px;
  font-size: 13px;
  cursor: pointer;
  white-space: nowrap;
}}
#search-bar button:hover {{ background: #2e4fcc; }}
#search-error {{
  font-size: 12px;
  color: #c0392b;
  display: none;
}}

#legend {{
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.12);
  padding: 10px 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 12px;
  color: #333;
  user-select: none;
}}
.legend-item input {{ cursor: pointer; accent-color: var(--c); }}
.legend-dot {{
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--c);
  flex-shrink: 0;
}}

/* ── Right: detail panel ── */
#panel {{
  flex: 0 0 38%;
  background: #fff;
  border-left: 1px solid #ddd;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
#panel-placeholder {{
  color: #bbb;
  font-size: 14px;
  margin: auto;
  text-align: center;
  line-height: 2;
}}

.meta-grid {{
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 5px 12px;
  background: #f7f7f7;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 13px;
}}
.meta-grid .lbl {{ color: #888; white-space: nowrap; }}
.meta-grid .val {{ font-weight: 600; color: #111; }}

.section-card {{
  border: 1px solid #e4e4e4;
  border-radius: 8px;
  overflow: hidden;
}}
.section-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  cursor: pointer;
  background: #fafafa;
  border-bottom: 1px solid #eee;
  user-select: none;
}}
.section-header:hover {{ background: #eef2ff; }}
.section-title {{
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: #555;
}}
.bloom-chip {{
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 700;
}}
.arrow {{ font-size: 10px; color: #aaa; transition: transform .15s; }}
.arrow.open {{ transform: rotate(180deg); }}

.justif {{
  font-size: 12px;
  font-style: italic;
  color: #555;
  padding: 7px 12px;
  background: #eef2ff;
  border-bottom: 1px solid #d8e0ff;
  display: none;
}}
.justif.open {{ display: block; }}

.section-body {{
  padding: 10px 12px;
  font-size: 13px;
  line-height: 1.65;
  color: #333;
  white-space: pre-wrap;
  word-break: break-word;
}}

h2 {{ font-size: 15px; color: #111; font-weight: 700; }}
hr {{ border: none; border-top: 1px solid #eee; }}
</style>
</head>
<body>

<div id="left">
  <div id="canvas-wrap">
    <canvas id="c"></canvas>
    <div id="tooltip"></div>
    <div id="hint">Scroll to zoom · Drag to pan · Click a dot to inspect</div>
  </div>
  <div id="search-bar">
    <input id="search-input" type="text" placeholder="Search by Row ID e.g. Row_2269" />
    <button onclick="searchRow()">Go</button>
    <span id="search-error">Not found</span>
  </div>
  <div id="legend"></div>
</div>

<div id="panel">
  <p id="panel-placeholder">← Click any dot to inspect a reflection</p>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────
const POINTS   = {points_json};
const CLUSTERS = {clusters_json};

const COLOR_MAP  = {{}};
const LABEL_MAP  = {{}};
CLUSTERS.forEach(c => {{ COLOR_MAP[c.fid] = c.color; LABEL_MAP[c.fid] = c.label; }});

const BLOOM_NAME  = {{1:"Remember",2:"Understand",3:"Apply",4:"Analyse",5:"Evaluate",6:"Create"}};
const BLOOM_COLOR = {{1:"#e57373",2:"#ff9800",3:"#c9a800",4:"#4caf50",5:"#42a5f5",6:"#ab47bc"}};

// ── Canvas setup ─────────────────────────────────────────────────────────
const canvas  = document.getElementById("c");
const ctx     = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

let W, H;
function resize() {{
  const wrap = canvas.parentElement;
  W = canvas.width  = wrap.clientWidth  * devicePixelRatio;
  H = canvas.height = wrap.clientHeight * devicePixelRatio;
  canvas.style.width  = wrap.clientWidth  + "px";
  canvas.style.height = wrap.clientHeight + "px";
  draw();
}}

// ── Transform (pan / zoom) ────────────────────────────────────────────────
let tx = 0, ty = 0, scale = 1;

// compute data bounds once
const xs = POINTS.map(p => p.x), ys = POINTS.map(p => p.y);
const xMin = Math.min(...xs), xMax = Math.max(...xs);
const yMin = Math.min(...ys), yMax = Math.max(...ys);

function initTransform() {{
  const pad = 40 * devicePixelRatio;
  scale = Math.min((W - 2*pad) / (xMax - xMin), (H - 2*pad) / (yMax - yMin));
  tx = (W - scale * (xMax + xMin)) / 2;
  ty = (H - scale * (yMax + yMin)) / 2;
}}

function toCanvas(dx, dy) {{ return [dx * scale + tx, dy * scale + ty]; }}
function toData(cx, cy)   {{ return [(cx - tx) / scale, (cy - ty) / scale]; }}

// ── Visibility state ─────────────────────────────────────────────────────
const visible = {{}};
CLUSTERS.forEach(c => {{ visible[c.fid] = c.fid !== -1; }});

// ── Draw ─────────────────────────────────────────────────────────────────
const DOT_R   = 3.5 * devicePixelRatio;
const SEL_R   = 6   * devicePixelRatio;
let selectedIdx = null;

function draw() {{
  ctx.clearRect(0, 0, W, H);

  // draw noise first (bottom layer)
  [-1, ...Array.from({{length: 10}}, (_, i) => i)].forEach(fid => {{
    if (!visible[fid]) return;
    ctx.globalAlpha = fid === -1 ? 0.25 : 0.75;
    ctx.fillStyle   = COLOR_MAP[fid];
    POINTS.forEach((p, i) => {{
      if (p.fid !== fid || i === selectedIdx) return;
      const [cx, cy] = toCanvas(p.x, p.y);
      if (cx < -10 || cx > W+10 || cy < -10 || cy > H+10) return;
      ctx.beginPath();
      ctx.arc(cx, cy, DOT_R, 0, Math.PI*2);
      ctx.fill();
    }});
  }});

  // draw selected on top
  if (selectedIdx !== null) {{
    const p = POINTS[selectedIdx];
    if (visible[p.fid]) {{
      const [cx, cy] = toCanvas(p.x, p.y);
      ctx.globalAlpha = 1;
      ctx.fillStyle   = COLOR_MAP[p.fid];
      ctx.strokeStyle = "#000";
      ctx.lineWidth   = 1.5 * devicePixelRatio;
      ctx.beginPath();
      ctx.arc(cx, cy, SEL_R, 0, Math.PI*2);
      ctx.fill();
      ctx.stroke();
    }}
  }}
  ctx.globalAlpha = 1;
}}

// ── Hit-test ─────────────────────────────────────────────────────────────
const HIT_R = 10 * devicePixelRatio;
function findNearest(cx, cy) {{
  let best = null, bestD = HIT_R * HIT_R;
  POINTS.forEach((p, i) => {{
    if (!visible[p.fid]) return;
    const [px, py] = toCanvas(p.x, p.y);
    const d = (px-cx)**2 + (py-cy)**2;
    if (d < bestD) {{ bestD = d; best = i; }}
  }});
  return best;
}}

// ── Mouse events ─────────────────────────────────────────────────────────
let dragging = false, lastMX, lastMY, dragMoved = false;

canvas.addEventListener("mousedown", e => {{
  dragging  = true;
  dragMoved = false;
  lastMX    = e.clientX;
  lastMY    = e.clientY;
}});

canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const cx   = (e.clientX - rect.left) * devicePixelRatio;
  const cy   = (e.clientY - rect.top)  * devicePixelRatio;

  if (dragging) {{
    const dx = (e.clientX - lastMX) * devicePixelRatio;
    const dy = (e.clientY - lastMY) * devicePixelRatio;
    if (Math.abs(dx) + Math.abs(dy) > 2) dragMoved = true;
    tx += dx; ty += dy;
    lastMX = e.clientX; lastMY = e.clientY;
    draw();
    return;
  }}

  // hover tooltip
  const hit = findNearest(cx, cy);
  if (hit !== null) {{
    const p = POINTS[hit];
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX - rect.left + 12) + "px";
    tooltip.style.top  = (e.clientY - rect.top  - 24) + "px";
    tooltip.textContent = `${{p.id}}  ·  Band: ${{p.band}}  ·  Score: ${{p.score}}  ·  Bloom: ${{p.bloom_weighted}}`;
    canvas.style.cursor = "pointer";
  }} else {{
    tooltip.style.display = "none";
    canvas.style.cursor   = "crosshair";
  }}
}});

canvas.addEventListener("mouseup", e => {{
  if (!dragMoved) {{
    // treat as click
    const rect = canvas.getBoundingClientRect();
    const cx   = (e.clientX - rect.left) * devicePixelRatio;
    const cy   = (e.clientY - rect.top)  * devicePixelRatio;
    const hit  = findNearest(cx, cy);
    if (hit !== null) {{
      selectedIdx = hit;
      draw();
      showPanel(POINTS[hit]);
    }}
  }}
  dragging = false;
}});

canvas.addEventListener("mouseleave", () => {{
  dragging = false;
  tooltip.style.display = "none";
}});

canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const rect   = canvas.getBoundingClientRect();
  const cx     = (e.clientX - rect.left) * devicePixelRatio;
  const cy     = (e.clientY - rect.top)  * devicePixelRatio;
  const factor = e.deltaY < 0 ? 1.12 : 1/1.12;
  tx    = cx + (tx - cx) * factor;
  ty    = cy + (ty - cy) * factor;
  scale *= factor;
  draw();
}}, {{passive: false}});

// ── Legend ────────────────────────────────────────────────────────────────
const legend = document.getElementById("legend");
CLUSTERS.forEach(c => {{
  const item = document.createElement("label");
  item.className = "legend-item";
  item.style.setProperty("--c", c.color);
  item.innerHTML = `
    <input type="checkbox" data-fid="${{c.fid}}" ${{visible[c.fid] ? "checked" : ""}}>
    <span class="legend-dot"></span>
    ${{c.fid === -1 ? "<em>" + c.label + "</em>" : c.label}}
  `;
  item.querySelector("input").addEventListener("change", ev => {{
    visible[c.fid] = ev.target.checked;
    draw();
  }});
  legend.appendChild(item);
}});

// ── Detail panel ─────────────────────────────────────────────────────────
function esc(s) {{
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

function sectionCard(title, text, bdata) {{
  const level  = (bdata || {{}}).level || "?";
  const name   = BLOOM_NAME[level] || "";
  const justif = esc((bdata || {{}}).justification || "No justification available.");
  const color  = BLOOM_COLOR[level] || "#999";
  const uid    = Math.random().toString(36).slice(2);
  return `
<div class="section-card">
  <div class="section-header" onclick="toggleJ('${{uid}}')">
    <span class="section-title">${{title}}</span>
    <span class="bloom-chip">
      <span style="color:${{color}}">Bloom ${{level}}${{name ? " · "+name : ""}}</span>
      <span class="arrow" id="arr_${{uid}}">▼</span>
    </span>
  </div>
  <div class="justif" id="justif_${{uid}}">${{justif}}</div>
  <div class="section-body">${{esc(text)}}</div>
</div>`;
}}

window.toggleJ = function(uid) {{
  document.getElementById("justif_"+uid).classList.toggle("open");
  document.getElementById("arr_"+uid).classList.toggle("open");
}};

function showPanel(p) {{
  const panel = document.getElementById("panel");
  panel.innerHTML = `
<h2>${{esc(p.id)}}</h2>
<div class="meta-grid">
  <span class="lbl">Band</span>          <span class="val">${{esc(p.band)}}</span>
  <span class="lbl">Score</span>         <span class="val">${{esc(p.score)}}</span>
  <span class="lbl">Bloom (weighted)</span><span class="val">${{p.bloom_weighted}}</span>
  <span class="lbl">Final cluster</span> <span class="val">${{p.fid}} · ${{esc(p.final_cluster_theme)}}</span>
  <span class="lbl">Source cluster</span><span class="val">${{p.src_cluster_id}} · ${{esc(p.src_cluster_theme)}}</span>
</div>
<hr>
${{sectionCard("Situation",     p.situation,   p.bloom_situation)}}
${{sectionCard("Task / Action", p.task_action, p.bloom_task_action)}}
${{sectionCard("Result",        p.result,      p.bloom_result)}}
`;
}}

// ── Search ───────────────────────────────────────────────────────────────
const rowIndex = {{}};
POINTS.forEach((p, i) => {{ rowIndex[p.id.trim().toLowerCase()] = i; }});

window.searchRow = function() {{
  const query = document.getElementById("search-input").value.trim().toLowerCase();
  const errEl = document.getElementById("search-error");
  const idx   = rowIndex[query];
  if (idx === undefined) {{
    errEl.style.display = "inline";
    return;
  }}
  errEl.style.display = "none";

  const p = POINTS[idx];

  // make the cluster visible if it was hidden
  if (!visible[p.fid]) {{
    visible[p.fid] = true;
    document.querySelector(`#legend input[data-fid="${{p.fid}}"]`).checked = true;
  }}

  // centre the view on the point
  const [cx, cy] = toCanvas(p.x, p.y);
  tx += W/2 - cx;
  ty += H/2 - cy;

  selectedIdx = idx;
  draw();
  showPanel(p);
}};

document.getElementById("search-input").addEventListener("keydown", e => {{
  if (e.key === "Enter") searchRow();
}});

// ── Init ─────────────────────────────────────────────────────────────────
window.addEventListener("resize", resize);
resize();
initTransform();
draw();
</script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"  Saved → '{OUTPUT_HTML}'")
print("Open umap_interactive.html in a browser.")
print("Done.")
