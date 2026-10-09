"""
Générateur de Visualiseur Interactif HTML et Carte Vectorielle SVG
===================================================================
Génère sans dépendance externe (pur Python standard) :
1. Une carte vectorielle SVG haute définition de toutes les trajectoires.
2. Un visualiseur interactif HTML5 autonome permettant d'explorer et de tracer
   chaque trajectoire individuelle du dataset avec ses profils de vitesses.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import csv
import math
import json
from typing import List, Dict, Any


def load_trajectories(csv_path: str) -> List[List[Dict[str, Any]]]:
    """Charge toutes les trajectoires du CSV."""
    trajectories = []
    current_traj = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            if ts == 0.0 and current_traj:
                trajectories.append(current_traj)
                current_traj = []
            current_traj.append(row)

    if current_traj:
        trajectories.append(current_traj)

    return trajectories


def determine_category(traj: List[Dict[str, Any]]) -> str:
    x0 = float(traj[0]["x"])
    y0 = float(traj[0]["y"])
    if abs(x0) < 0.05 or abs(y0) < 0.05:
        return "Axial"
    elif x0 > 0 and y0 > 0:
        return "Q1"
    elif x0 < 0 and y0 > 0:
        return "Q2"
    elif x0 < 0 and y0 < 0:
        return "Q3"
    else:
        return "Q4"


def export_svg(trajectories: List[List[Dict[str, Any]]], output_path: str) -> None:
    """Génère une carte SVG vectorielle haute résolution (1000x1000 px)."""
    width = 1000
    height = 1000
    # Plage cartésienne : [-5.5, +5.5] m (domaine 5 mètres étendu)
    scale = width / 11.5  # ~87 px/m
    cx = width / 2.0
    cy = height / 2.0

    def world_to_svg(x: float, y: float):
        # Y inversé en SVG (haut = -Y en svg)
        sx = cx + x * scale
        sy = cy - y * scale
        return sx, sy

    color_map = {
        "Q1": "#1f77b4",     # Bleu
        "Q2": "#ff7f0e",     # Orange
        "Q3": "#9467bd",     # Violet
        "Q4": "#2ca02c",     # Vert
        "Axial": "#d62728",  # Rouge
    }

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#ffffff; font-family: sans-serif;">',
        f'  <defs>',
        f'    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>',
        f'    </marker>',
        f'  </defs>',
        f'  <g id="grid">',
    ]

    # Grille cartésienne jusqu'à 5m
    for m in range(-5, 6):
        sx1, sy1 = world_to_svg(m, -5.2)
        sx2, sy2 = world_to_svg(m, 5.2)
        col = "#888888" if m == 0 else "#e0e0e0"
        stroke_w = "2" if m == 0 else "1"
        svg_lines.append(f'    <line x1="{sx1:.1f}" y1="{sy1:.1f}" x2="{sx2:.1f}" y2="{sy2:.1f}" stroke="{col}" stroke-width="{stroke_w}"/>')

        sx1, sy1 = world_to_svg(-5.2, m)
        sx2, sy2 = world_to_svg(5.2, m)
        col = "#888888" if m == 0 else "#e0e0e0"
        stroke_w = "2" if m == 0 else "1"
        svg_lines.append(f'    <line x1="{sx1:.1f}" y1="{sy1:.1f}" x2="{sx2:.1f}" y2="{sy2:.1f}" stroke="{col}" stroke-width="{stroke_w}"/>')

        if m != 0:
            tx, ty = world_to_svg(m, -0.22)
            svg_lines.append(f'    <text x="{tx:.1f}" y="{ty:.1f}" font-size="11" fill="#777" text-anchor="middle">{m}m</text>')
            tx, ty = world_to_svg(0.22, m)
            svg_lines.append(f'    <text x="{tx:.1f}" y="{ty:.1f}" font-size="11" fill="#777">{m}m</text>')

    svg_lines.append('  </g>')

    # Tracé des trajectoires
    svg_lines.append('  <g id="trajectories">')
    for idx, traj in enumerate(trajectories):
        cat = determine_category(traj)
        col = color_map.get(cat, "#555")

        points = []
        for r in traj:
            sx, sy = world_to_svg(float(r["x"]), float(r["y"]))
            points.append(f"{sx:.1f},{sy:.1f}")

        pts_str = " ".join(points)
        svg_lines.append(f'    <polyline points="{pts_str}" fill="none" stroke="{col}" stroke-width="1.8" opacity="0.6"/>')

        # Point de départ
        sx0, sy0 = world_to_svg(float(traj[0]["x"]), float(traj[0]["y"]))
        svg_lines.append(f'    <circle cx="{sx0:.1f}" cy="{sy0:.1f}" r="3.5" fill="{col}"/>')

        # Flèche de cap initial
        th0 = float(traj[0]["theta"])
        hx = float(traj[0]["x"]) + 0.12 * math.cos(th0)
        hy = float(traj[0]["y"]) + 0.12 * math.sin(th0)
        shx, shy = world_to_svg(hx, hy)
        svg_lines.append(f'    <line x1="{sx0:.1f}" y1="{sy0:.1f}" x2="{shx:.1f}" y2="{shy:.1f}" stroke="#222" stroke-width="1.5" marker-end="url(#arrow)"/>')

    svg_lines.append('  </g>')

    # Cible [0, 0, 0]
    tcx, tcy = world_to_svg(0.0, 0.0)
    svg_lines.append(f'  <circle cx="{tcx:.1f}" cy="{tcy:.1f}" r="{0.10*scale:.1f}" fill="none" stroke="#e63946" stroke-width="2" stroke-dasharray="4,4"/>')
    svg_lines.append(f'  <polygon points="{tcx:.1f},{tcy-12:.1f} {tcx+4:.1f},{tcy-3:.1f} {tcx+12:.1f},{tcy-3:.1f} {tcx+6:.1f},{tcy+3:.1f} {tcx+9:.1f},{tcy+12:.1f} {tcx:.1f},{tcy+6:.1f} {tcx-9:.1f},{tcy+12:.1f} {tcx-6:.1f},{tcy+3:.1f} {tcx-12:.1f},{tcy-3:.1f} {tcx-4:.1f},{tcy-3:.1f}" fill="#e63946"/>')
    svg_lines.append(f'  <text x="{tcx+15:.1f}" y="{tcy+5:.1f}" font-size="14" font-weight="bold" fill="#e63946">Cible [0, 0, 0]</text>')

    # Titre et légende
    svg_lines.append('  <g id="legend" transform="translate(30, 40)">')
    svg_lines.append(f'    <rect width="250" height="180" fill="white" stroke="#ccc" rx="8" opacity="0.95"/>')
    svg_lines.append(f'    <text x="15" y="25" font-size="15" font-weight="bold" fill="#111">Trajectoires LIMO ({len(trajectories)})</text>')
    
    y_off = 50
    for cat, col in color_map.items():
        svg_lines.append(f'    <line x1="15" y1="{y_off}" x2="45" y2="{y_off}" stroke="{col}" stroke-width="3"/>')
        svg_lines.append(f'    <text x="55" y="{y_off+4}" font-size="12" fill="#333">{cat}</text>')
        y_off += 24

    svg_lines.append(f'    <circle cx="30" cy="{y_off}" r="4" fill="#e63946"/>')
    svg_lines.append(f'    <text x="55" y="{y_off+4}" font-size="12" fill="#333">Stationnement final (d &lt; 10 cm)</text>')
    svg_lines.append('  </g>')

    svg_lines.append('</svg>')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
    print(f"[Succès] Carte SVG sauvegardée : {output_path}")


def export_interactive_html(trajectories: List[List[Dict[str, Any]]], output_path: str) -> None:
    """Génère le visualiseur interactif HTML5 complet avec inspecteur de trajectoire."""
    # Préparation des données allégées pour JSON
    data = []
    for idx, t in enumerate(trajectories):
        cat = determine_category(t)
        pts = [[round(float(r["x"]), 3), round(float(r["y"]), 3), round(float(r["theta"]), 3),
                round(float(r["v_lin"]), 3), round(float(r["v_ang"]), 3), round(float(r["timestamp"]), 2)]
               for r in t]
        data.append({
            "id": idx + 1,
            "category": cat,
            "start": [round(float(t[0]["x"]), 2), round(float(t[0]["y"]), 2), round(float(t[0]["theta"]), 2)],
            "end": [round(float(t[-1]["x"]), 3), round(float(t[-1]["y"]), 3), round(float(t[-1]["theta"]), 3)],
            "steps": len(t),
            "duration": round(float(t[-1]["timestamp"]), 2),
            "v_lin_max": round(max(float(r["v_lin"]) for r in t), 3),
            "v_lin_min": round(min(float(r["v_lin"]) for r in t), 3),
            "points": pts
        })

    json_str = json.dumps(data)

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>LIMO Offline Learning - Visualisateur de Trajectoires du Dataset</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; height: 100vh; overflow: hidden; }}
    #sidebar {{ width: 380px; background: #1e293b; border-right: 1px solid #334155; display: flex; flex-direction: column; padding: 20px; overflow-y: auto; }}
    #main {{ flex: 1; display: flex; flex-direction: column; }}
    #header {{ background: #1e293b; padding: 14px 24px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
    #canvas-container {{ flex: 1; position: relative; background: #090d16; overflow: hidden; }}
    canvas {{ display: block; width: 100%; height: 100%; cursor: crosshair; }}
    h1 {{ font-size: 1.15rem; color: #38bdf8; font-weight: 700; }}
    .badge {{ background: #0284c7; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; }}
    .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
    .card-title {{ font-size: 0.85rem; font-weight: bold; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; }}
    .control-row {{ margin-bottom: 12px; }}
    label {{ display: block; font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }}
    select, button {{ width: 100%; background: #334155; border: 1px solid #475569; color: white; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; outline: none; }}
    select:focus {{ border-color: #38bdf8; }}
    .btn-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    button:hover {{ background: #475569; cursor: pointer; }}
    .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .stat-box {{ background: #1e293b; padding: 8px; border-radius: 6px; text-align: center; }}
    .stat-val {{ font-size: 1.05rem; font-weight: bold; color: #38bdf8; }}
    .stat-lbl {{ font-size: 0.7rem; color: #64748b; margin-top: 2px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.8rem; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    #telemetry-canvas {{ width: 100%; height: 120px; background: #090d16; border-radius: 4px; margin-top: 6px; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div style="margin-bottom: 16px;">
      <h1>Trajectoires du Dataset LIMO</h1>
      <p style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">{len(trajectories)} Trajectoires Expertes (Domaine 5 Mètres)</p>
    </div>

    <div class="card">
      <div class="card-title">Filtre et Sélection</div>
      <div class="control-row">
        <label>Région / Quadrant :</label>
        <select id="cat-select" onchange="onCategoryChange()">
          <option value="ALL">Toutes les Régions ({len(trajectories)})</option>
          <option value="Q1">Quadrant 1 (+X, +Y)</option>
          <option value="Q2">Quadrant 2 (-X, +Y)</option>
          <option value="Q3">Quadrant 3 (-X, -Y)</option>
          <option value="Q4">Quadrant 4 (+X, -Y)</option>
          <option value="Axial">Approches Axiales (X=0 ou Y=0)</option>
        </select>
      </div>
      <div class="control-row">
        <label>Trajectoire Individuelle :</label>
        <select id="traj-select" onchange="onTrajSelect()">
          <option value="-1">-- Vue d'ensemble générale --</option>
        </select>
      </div>
      <div class="btn-row">
        <button onclick="prevTraj()">◀ Précédente</button>
        <button onclick="nextTraj()">Suivante ▶</button>
      </div>
    </div>

    <div class="card" id="info-card">
      <div class="card-title" id="info-title">Détails de la Trajectoire</div>
      <div class="stat-grid">
        <div class="stat-box"><div class="stat-val" id="st-start">--</div><div class="stat-lbl">Départ (x, y)</div></div>
        <div class="stat-box"><div class="stat-val" id="st-end">--</div><div class="stat-lbl">Arrivée (x, y)</div></div>
        <div class="stat-box"><div class="stat-val" id="st-dur">--</div><div class="stat-lbl">Durée</div></div>
        <div class="stat-box"><div class="stat-val" id="st-speed">--</div><div class="stat-lbl">V_lin Max</div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Profil de Vitesse v_lin(t)</div>
      <canvas id="telemetry-canvas" width="320" height="120"></canvas>
      <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px; display: flex; justify-content: space-between;">
        <span>Vert : v_lin (m/s)</span>
        <span>Violet : v_ang (rad/s)</span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Légende des Couleurs</div>
      <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Q1 (+X, +Y)</div>
      <div class="legend-item"><span class="dot" style="background:#fb923c;"></span> Q2 (-X, +Y)</div>
      <div class="legend-item"><span class="dot" style="background:#c084fc;"></span> Q3 (-X, -Y)</div>
      <div class="legend-item"><span class="dot" style="background:#4ade80;"></span> Q4 (+X, -Y)</div>
      <div class="legend-item"><span class="dot" style="background:#f87171;"></span> Axiales (Axes X et Y)</div>
      <div class="legend-item"><span class="dot" style="background:#e11d48; width:12px; height:12px;"></span> Étoile : Cible [0, 0, 0]</div>
    </div>
  </div>

  <div id="main">
    <div id="header">
      <div>
        <span style="font-weight:600;">Carte Cartésienne Interactive du Dataset</span>
        <span style="color:#64748b; font-size:0.8rem; margin-left: 10px;">(Glisser pour déplacer, Molette pour zoomer)</span>
      </div>
      <div>
        <button style="width: auto; padding: 4px 12px;" onclick="resetView()">Réinitialiser la Vue</button>
      </div>
    </div>
    <div id="canvas-container">
      <canvas id="map-canvas"></canvas>
    </div>
  </div>

  <script>
    const trajectories = {json_str};
    const catColors = {{
      "Q1": "#38bdf8",
      "Q2": "#fb923c",
      "Q3": "#c084fc",
      "Q4": "#4ade80",
      "Axial": "#f87171"
    }};

    let selectedTrajId = -1;
    let selectedCat = "ALL";
    let scale = 140; // px/m
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const mapCanvas = document.getElementById("map-canvas");
    const mapCtx = mapCanvas.getContext("2d");
    const telemCanvas = document.getElementById("telemetry-canvas");
    const telemCtx = telemCanvas.getContext("2d");

    function init() {{
      resizeCanvas();
      window.addEventListener("resize", () => {{ resizeCanvas(); draw(); }});
      populateSelect();
      resetView();
      setupEvents();
    }}

    function resizeCanvas() {{
      const container = document.getElementById("canvas-container");
      mapCanvas.width = container.clientWidth;
      mapCanvas.height = container.clientHeight;
    }}

    function resetView() {{
      scale = Math.min(mapCanvas.width, mapCanvas.height) / 11.5;
      panX = mapCanvas.width / 2.0;
      panY = mapCanvas.height / 2.0;
      draw();
    }}

    function populateSelect() {{
      const sel = document.getElementById("traj-select");
      sel.innerHTML = '<option value="-1">-- Vue d\\'ensemble générale (' + trajectories.length + ') --</option>';
      trajectories.forEach(t => {{
        if (selectedCat === "ALL" || t.category === selectedCat) {{
          const opt = document.createElement("option");
          opt.value = t.id;
          opt.textContent = `#${{String(t.id).padStart(3, '0')}} [${{t.category}}] (${{t.start[0]}}, ${{t.start[1]}}) th=${{t.start[2]}} rad`;
          sel.appendChild(opt);
        }}
      }});
    }}

    function onCategoryChange() {{
      selectedCat = document.getElementById("cat-select").value;
      populateSelect();
      selectedTrajId = -1;
      updateCard();
      draw();
      drawTelemetry();
    }}

    function onTrajSelect() {{
      selectedTrajId = parseInt(document.getElementById("traj-select").value);
      updateCard();
      draw();
      drawTelemetry();
    }}

    function nextTraj() {{
      const sel = document.getElementById("traj-select");
      if (sel.selectedIndex < sel.options.length - 1) {{
        sel.selectedIndex++;
        onTrajSelect();
      }}
    }}

    function prevTraj() {{
      const sel = document.getElementById("traj-select");
      if (sel.selectedIndex > 0) {{
        sel.selectedIndex--;
        onTrajSelect();
      }}
    }}

    function updateCard() {{
      if (selectedTrajId === -1) {{
        document.getElementById("info-title").textContent = "Vue d'ensemble";
        document.getElementById("st-start").textContent = trajectories.length + " Essais";
        document.getElementById("st-end").textContent = "[0.0, 0.0]";
        document.getElementById("st-dur").textContent = "10-35s";
        document.getElementById("st-speed").textContent = "0.32 m/s";
      }} else {{
        const t = trajectories.find(x => x.id === selectedTrajId);
        if (!t) return;
        document.getElementById("info-title").textContent = `Trajectoire #${{t.id}} [${{t.category}}]`;
        document.getElementById("st-start").textContent = `(${{t.start[0]}}, ${{t.start[1]}})`;
        document.getElementById("st-end").textContent = `(${{t.end[0]}}, ${{t.end[1]}})`;
        document.getElementById("st-dur").textContent = `${{t.duration}}s`;
        document.getElementById("st-speed").textContent = `${{t.v_lin_max}} m/s`;
      }}
    }}

    function toScreen(x, y) {{
      return [panX + x * scale, panY - y * scale];
    }}

    function draw() {{
      mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);

      // 1. Grille
      mapCtx.strokeStyle = "#1e293b";
      mapCtx.lineWidth = 1;
      for (let m = -5; m <= 5; m++) {{
        const [x1, y1] = toScreen(m, -5.2);
        const [x2, y2] = toScreen(m, 5.2);
        mapCtx.beginPath();
        mapCtx.moveTo(x1, y1);
        mapCtx.lineTo(x2, y2);
        mapCtx.stroke();

        const [x3, y3] = toScreen(-5.2, m);
        const [x4, y4] = toScreen(5.2, m);
        mapCtx.beginPath();
        mapCtx.moveTo(x3, y3);
        mapCtx.lineTo(x4, y4);
        mapCtx.stroke();
      }}

      // Axes centraux
      mapCtx.strokeStyle = "#475569";
      mapCtx.lineWidth = 1.5;
      const [ax1, ay1] = toScreen(0, -5.2);
      const [ax2, ay2] = toScreen(0, 5.2);
      mapCtx.beginPath(); mapCtx.moveTo(ax1, ay1); mapCtx.lineTo(ax2, ay2); mapCtx.stroke();
      const [ax3, ay3] = toScreen(-5.2, 0);
      const [ax4, ay4] = toScreen(5.2, 0);
      mapCtx.beginPath(); mapCtx.moveTo(ax3, ay3); mapCtx.lineTo(ax4, ay4); mapCtx.stroke();

      // 2. Trajectoires
      trajectories.forEach(t => {{
        if (selectedCat !== "ALL" && t.category !== selectedCat) return;

        const isSelected = (t.id === selectedTrajId);
        const col = catColors[t.category] || "#94a3b8";

        mapCtx.beginPath();
        const [sx, sy] = toScreen(t.points[0][0], t.points[0][1]);
        mapCtx.moveTo(sx, sy);
        for (let i = 1; i < t.points.length; i++) {{
          const [px, py] = toScreen(t.points[i][0], t.points[i][1]);
          mapCtx.lineTo(px, py);
        }}

        if (isSelected) {{
          mapCtx.strokeStyle = "#ffffff";
          mapCtx.lineWidth = 3.5;
          mapCtx.stroke();
          mapCtx.strokeStyle = col;
          mapCtx.lineWidth = 2.5;
          mapCtx.stroke();
        }} else if (selectedTrajId === -1) {{
          mapCtx.strokeStyle = col;
          mapCtx.lineWidth = 1.4;
          mapCtx.globalAlpha = 0.55;
          mapCtx.stroke();
          mapCtx.globalAlpha = 1.0;
        }} else {{
          // Dimmer les autres
          mapCtx.strokeStyle = col;
          mapCtx.lineWidth = 0.8;
          mapCtx.globalAlpha = 0.12;
          mapCtx.stroke();
          mapCtx.globalAlpha = 1.0;
        }}

        // Point de départ
        mapCtx.fillStyle = isSelected ? "#ffffff" : col;
        mapCtx.beginPath();
        mapCtx.arc(sx, sy, isSelected ? 5.5 : 2.5, 0, Math.PI * 2);
        mapCtx.fill();

        // Direction initiale si sélectionnée
        if (isSelected) {{
          const th0 = t.points[0][2];
          const hx = t.points[0][0] + 0.22 * Math.cos(th0);
          const hy = t.points[0][1] + 0.22 * Math.sin(th0);
          const [hsx, hsy] = toScreen(hx, hy);
          mapCtx.strokeStyle = "#facc15";
          mapCtx.lineWidth = 2.0;
          mapCtx.beginPath();
          mapCtx.moveTo(sx, sy);
          mapCtx.lineTo(hsx, hsy);
          mapCtx.stroke();
        }}
      }});

      // 3. Cible d'accostage [0, 0, 0]
      const [tx, ty] = toScreen(0, 0);
      mapCtx.strokeStyle = "#e11d48";
      mapCtx.lineWidth = 2.0;
      mapCtx.setLineDash([4, 4]);
      mapCtx.beginPath();
      mapCtx.arc(tx, ty, 0.10 * scale, 0, Math.PI * 2);
      mapCtx.stroke();
      mapCtx.setLineDash([]);

      // Étoile cible
      mapCtx.fillStyle = "#e11d48";
      drawStar(mapCtx, tx, ty, 5, 9, 4);
    }}

    function drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius) {{
      let rot = Math.PI / 2 * 3;
      let x = cx;
      let y = cy;
      let step = Math.PI / spikes;

      ctx.beginPath();
      ctx.moveTo(cx, cy - outerRadius);
      for (let i = 0; i < spikes; i++) {{
        x = cx + Math.cos(rot) * outerRadius;
        y = cy + Math.sin(rot) * outerRadius;
        ctx.lineTo(x, y);
        rot += step;

        x = cx + Math.cos(rot) * innerRadius;
        y = cy + Math.sin(rot) * innerRadius;
        ctx.lineTo(x, y);
        rot += step;
      }}
      ctx.lineTo(cx, cy - outerRadius);
      ctx.closePath();
      ctx.fill();
    }}

    function drawTelemetry() {{
      telemCtx.clearRect(0, 0, telemCanvas.width, telemCanvas.height);
      if (selectedTrajId === -1) return;

      const t = trajectories.find(x => x.id === selectedTrajId);
      if (!t || t.points.length === 0) return;

      const w = telemCanvas.width;
      const h = telemCanvas.height;
      const n = t.points.length;

      // Axe zéro
      telemCtx.strokeStyle = "#334155";
      telemCtx.beginPath();
      telemCtx.moveTo(0, h * 0.7);
      telemCtx.lineTo(w, h * 0.7);
      telemCtx.stroke();

      // Courbe v_lin (Vert)
      telemCtx.strokeStyle = "#4ade80";
      telemCtx.lineWidth = 1.8;
      telemCtx.beginPath();
      for (let i = 0; i < n; i++) {{
        const x = (i / (n - 1)) * w;
        const v = t.points[i][3]; // v_lin
        const y = h * 0.7 - (v / 0.28) * (h * 0.55);
        if (i === 0) telemCtx.moveTo(x, y); else telemCtx.lineTo(x, y);
      }}
      telemCtx.stroke();

      // Courbe v_ang (Violet)
      telemCtx.strokeStyle = "#c084fc";
      telemCtx.lineWidth = 1.4;
      telemCtx.beginPath();
      for (let i = 0; i < n; i++) {{
        const x = (i / (n - 1)) * w;
        const wa = t.points[i][4]; // v_ang
        const y = h * 0.7 - (wa / 0.50) * (h * 0.35);
        if (i === 0) telemCtx.moveTo(x, y); else telemCtx.lineTo(x, y);
      }}
      telemCtx.stroke();
    }}

    function setupEvents() {{
      mapCanvas.addEventListener("mousedown", (e) => {{
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      }});

      window.addEventListener("mouseup", () => {{ isDragging = false; }});

      window.addEventListener("mousemove", (e) => {{
        if (isDragging) {{
          panX += e.clientX - lastMouseX;
          panY += e.clientY - lastMouseY;
          lastMouseX = e.clientX;
          lastMouseY = e.clientY;
          draw();
        }}
      }});

      mapCanvas.addEventListener("wheel", (e) => {{
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : 0.87;
        scale *= factor;
        draw();
      }});
    }}

    window.onload = init;
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Succès] Visualiseur interactif HTML généré : {output_path}")


def main():
    csv_path = "data/curated/handcrafted_dataset.csv"
    trajectories = load_trajectories(csv_path)
    print(f"[Info] {len(trajectories)} trajectoires chargées.")

    svg_path = "data/curated/dataset_trajectories_map.svg"
    export_svg(trajectories, svg_path)

    html_path = "data/curated/dataset_trajectories_viewer.html"
    export_interactive_html(trajectories, html_path)


if __name__ == "__main__":
    main()
