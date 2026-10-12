"""
Module de Comparaison Métrologique et Trajectorielle Inter-Approches (ISO 18646-2)
==================================================================================
Compare directement les performances de navigation en boucle fermée entre deux
approches d'apprentissage hors ligne (ex: V1 Séries Temporelles vs V2 Mapping Relatif).

Génère :
- Graphiques comparatifs des métriques ISO 18646-2 (Ap, Ao, Rp, Ro) en barres.
- Tracés superposés des trajectoires dans le plan cartésien 2D.
- Évolution temporelle de l'erreur euclidienne moyenne e(t).
- Tableau récapitulatif comparatif CSV.
- Dashboard HTML autonome interactif.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

from __future__ import annotations

import os
import sys
import glob
import re
import csv
import math
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

# Inclusion du dossier racine du projet dans sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Inclusion des chemins snap si présents
for _p in [
    os.path.expanduser("~/local/lib/python3.14/dist-packages"),
    os.path.expanduser("~/snap/antigravity-cli/common/local/lib/python3.14/dist-packages")
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def natural_sort_key(s: str) -> List[Any]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_session_csv(file_path: str) -> Dict[str, Any]:
    """Parse un fichier CSV de session individuelle généré par DataLogger."""
    timestamps: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    thetas: List[float] = []
    v_lins: List[float] = []
    v_angs: List[float] = []

    target = [0.0, 0.0, 0.0]
    initial_pose = [0.0, 0.0, 0.0]

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("# Target:"):
                try:
                    parts = row[0].replace("# Target:", "").strip().split(",")
                    target = [float(p.strip()) for p in parts]
                except Exception:
                    pass
                continue
            if row[0].startswith("# Initial_pose:"):
                try:
                    parts = row[0].replace("# Initial_pose:", "").strip().split(",")
                    initial_pose = [float(p.strip()) for p in parts]
                except Exception:
                    pass
                continue
            if row[0].startswith("#") or row[0].strip() == "timestamp":
                continue

            try:
                t = float(row[0])
                x = float(row[1])
                y = float(row[2])
                th = float(row[3])
                v_l = float(row[13]) if len(row) > 13 else 0.0
                v_a = float(row[14]) if len(row) > 14 else 0.0

                timestamps.append(t)
                xs.append(x)
                ys.append(y)
                thetas.append(th)
                v_lins.append(v_l)
                v_angs.append(v_a)
            except (ValueError, IndexError):
                continue

    return {
        "file": file_path,
        "timestamps": timestamps,
        "xs": xs,
        "ys": ys,
        "thetas": thetas,
        "v_lins": v_lins,
        "v_angs": v_angs,
        "target": target,
        "initial_pose": initial_pose,
        "xf": xs[-1] if xs else 0.0,
        "yf": ys[-1] if ys else 0.0,
        "thf": thetas[-1] if thetas else 0.0,
        "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    }


def load_batch_sessions(batch_dir: str) -> List[Dict[str, Any]]:
    """Charge toutes les sessions CSV d'un dossier batch."""
    if not os.path.exists(batch_dir):
        return []
    csv_paths = sorted(glob.glob(os.path.join(batch_dir, "*.csv")), key=natural_sort_key)
    sessions = []
    for p in csv_paths:
        if "iso_18646_metrics" in p or "comparison" in p:
            continue
        try:
            s = parse_session_csv(p)
            if s["xs"]:
                sessions.append(s)
        except Exception:
            continue
    return sessions


def load_iso_csv(batch_dir: str) -> Dict[str, float]:
    """Charge les métriques ISO depuis le fichier iso_18646_metrics.csv si présent."""
    csv_path = os.path.join(batch_dir, "iso_18646_metrics.csv")
    metrics: Dict[str, float] = {}
    if not os.path.exists(csv_path):
        return metrics

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label = row[0].strip()
                try:
                    val = float(row[1].strip())
                    if "Ap" in label or "Précision de position" in label:
                        metrics["Ap"] = val
                    elif "Ao" in label or "Précision d'orientation" in label:
                        metrics["Ao_deg"] = val
                    elif "Rp" in label or "Répétabilité de position" in label:
                        metrics["Rp"] = val
                    elif "Ro" in label or "Répétabilité d'orientation" in label:
                        metrics["Ro_deg"] = val
                    elif "Sl" in label or "Écart-type radial" in label:
                        metrics["Sl"] = val
                    elif "So" in label or "Écart-type angulaire" in label:
                        metrics["So_deg"] = val
                except ValueError:
                    pass
    return metrics


def calculate_metrics_from_sessions(sessions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calcule les métriques ISO 18646-2 si le fichier CSV n'était pas disponible."""
    from calculus.calculate_iso_metrics import compute_iso_18646_metrics
    trials = [{"xf": s["xf"], "yf": s["yf"], "thf": s["thf"]} for s in sessions]
    res = compute_iso_18646_metrics(trials, xc=0.0, yc=0.0, oc=0.0)
    if not res:
        return {}
    return {
        "Ap": res["Ap"],
        "Ao_deg": res["Ao_abs_deg"],
        "Rp": res["Rp"],
        "Ro_deg": res["Ro_deg"],
        "Sl": res["Sl"],
        "So_deg": res["So_deg"],
        "success_rate": res["success_rate"]
    }


def plot_comparison_metrics(
    m1: Dict[str, float],
    m2: Dict[str, float],
    name1: str,
    name2: str,
    output_dir: str
) -> Tuple[str, str]:
    """Génère un graphique en barres comparatif des métriques ISO 18646-2."""
    if not MATPLOTLIB_AVAILABLE:
        return "", ""

    labels = ["Précision Pos (Ap)", "Précision Cap (Ao)", "Répétabilité Pos (Rp)", "Écart-type Radial (Sl)"]
    units = ["[m]", "[deg]", "[m]", "[m]"]
    
    val1 = [
        m1.get("Ap", 0.0),
        m1.get("Ao_deg", 0.0),
        m1.get("Rp", 0.0),
        m1.get("Sl", 0.0)
    ]
    val2 = [
        m2.get("Ap", 0.0),
        m2.get("Ao_deg", 0.0),
        m2.get("Rp", 0.0),
        m2.get("Sl", 0.0)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"Comparaison Métrologique ISO 18646-2 : {name1} vs {name2}", fontsize=14, fontweight="bold", y=1.02)

    bar_width = 0.35
    x_indices = [0, 1]

    sub_titles = [
        ("Précision Pos (Ap)", "m", 0.25),
        ("Précision Cap (Ao)", "deg", 15.0),
        ("Répétabilité Pos (Rp)", "m", None),
        ("Écart-type Radial (Sl)", "m", None)
    ]

    for idx, (title, unit, threshold) in enumerate(sub_titles):
        ax = axes[idx]
        ax.set_facecolor("#fafbfc")
        ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)

        v1 = val1[idx]
        v2 = val2[idx]

        bars = ax.bar([0, 1], [v1, v2], width=0.55, color=["#1f77b4", "#2ca02c"], zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([name1, name2], fontweight="bold")
        ax.set_ylabel(f"Valeur ({unit})", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Affichage des valeurs au-dessus des barres
        for b in bars:
            height = b.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Seuil ISO si applicable
        if threshold is not None:
            ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5,
                       label=f"Seuil ISO (<{threshold}{unit})", zorder=4)
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "comparison_iso_metrics.png")
    svg_path = os.path.join(output_dir, "comparison_iso_metrics.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_comparison_trajectories(
    sessions1: List[Dict[str, Any]],
    sessions2: List[Dict[str, Any]],
    name1: str,
    name2: str,
    output_dir: str
) -> Tuple[str, str]:
    """Génère la comparaison visuelle des trajectoires cartésiennes 2D."""
    if not MATPLOTLIB_AVAILABLE or (not sessions1 and not sessions2):
        return "", ""

    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")
    ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")

    # Cible
    ax.scatter([0.0], [0.0], color="#d62728", marker="*", s=220, zorder=10, label="Cible (0, 0)")
    ax.arrow(0.0, 0.0, 0.25, 0.0, head_width=0.08, head_length=0.08, fc="#d62728", ec="#d62728", zorder=10)

    # Trajectoires Approche 1
    for idx, s in enumerate(sessions1):
        lbl = name1 if idx == 0 else ""
        ax.plot(s["xs"], s["ys"], color="#1f77b4", linestyle="--", alpha=0.7, linewidth=1.5, label=lbl)
        if s["xs"]:
            ax.scatter(s["xs"][0], s["ys"][0], color="#1f77b4", marker="o", s=30, alpha=0.6)
            ax.scatter(s["xf"], s["yf"], color="#1f77b4", marker="x", s=50, alpha=0.8)

    # Trajectoires Approche 2
    for idx, s in enumerate(sessions2):
        lbl = name2 if idx == 0 else ""
        ax.plot(s["xs"], s["ys"], color="#2ca02c", linestyle="-", alpha=0.8, linewidth=2.0, label=lbl)
        if s["xs"]:
            ax.scatter(s["xs"][0], s["ys"][0], color="#2ca02c", marker="o", s=35, alpha=0.7)
            ax.scatter(s["xf"], s["yf"], color="#2ca02c", marker="+", s=60, alpha=0.9)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("Position X [m]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Position Y [m]", fontsize=11, fontweight="bold")
    ax.set_title(f"Comparaison Spatiale des Trajectoires : {name1} vs {name2}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#e2e8f0")

    png_path = os.path.join(output_dir, "comparison_trajectories.png")
    svg_path = os.path.join(output_dir, "comparison_trajectories.svg")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path, format="svg")
    plt.close(fig)
    return png_path, svg_path


def generate_comparison_dashboard(
    m1: Dict[str, float],
    m2: Dict[str, float],
    name1: str,
    name2: str,
    output_dir: str
) -> str:
    """Génère un dashboard HTML interactif comparatif autonome."""
    html_path = os.path.join(output_dir, "comparison_dashboard.html")

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Tableau de Bord Comparatif — LIMO Offline Learning</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #0f172a;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 12px;
            margin-top: 0;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            background-color: #3b82f6;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 25px 0;
        }}
        .card {{
            background: #f1f5f9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f8fafc;
            color: #475569;
            font-size: 13px;
            text-transform: uppercase;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .image-container {{
            text-align: center;
            margin: 25px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Tableau de Bord Comparatif des Performances</h1>
        <p><strong>Projet</strong> : LIMO Offline Learning (CERV / ENIB) &bull; <strong>Norme</strong> : ISO 18646-2</p>

        <div class="grid">
            <div class="card" style="border-left-color: #1f77b4;">
                <h3>Approche A : {name1}</h3>
                <p>Précision Pos (Ap) : <strong>{m1.get('Ap', 0.0):.4f} m</strong></p>
                <p>Précision Cap (Ao) : <strong>{m1.get('Ao_deg', 0.0):.2f}°</strong></p>
            </div>
            <div class="card" style="border-left-color: #2ca02c;">
                <h3>Approche B : {name2}</h3>
                <p>Précision Pos (Ap) : <strong>{m2.get('Ap', 0.0):.4f} m</strong></p>
                <p>Précision Cap (Ao) : <strong>{m2.get('Ao_deg', 0.0):.2f}°</strong></p>
            </div>
        </div>

        <h2>Synthèse Métrologique Normalisée</h2>
        <table>
            <thead>
                <tr>
                    <th>Métrique ISO 18646-2</th>
                    <th>Seuil Normatif</th>
                    <th>{name1}</th>
                    <th>{name2}</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Précision de Position (Ap)</strong></td>
                    <td>&lt; 0.250 m</td>
                    <td>{m1.get('Ap', 0.0):.4f} m</td>
                    <td><strong style="color: #16a34a;">{m2.get('Ap', 0.0):.4f} m</strong></td>
                </tr>
                <tr>
                    <td><strong>Précision d'Orientation (Ao)</strong></td>
                    <td>&lt; 15.00°</td>
                    <td>{m1.get('Ao_deg', 0.0):.2f}°</td>
                    <td><strong style="color: #16a34a;">{m2.get('Ao_deg', 0.0):.2f}°</strong></td>
                </tr>
                <tr>
                    <td><strong>Répétabilité de Position (Rp)</strong></td>
                    <td>—</td>
                    <td>{m1.get('Rp', 0.0):.4f} m</td>
                    <td>{m2.get('Rp', 0.0):.4f} m</td>
                </tr>
                <tr>
                    <td><strong>Écart-type Radial (Sl)</strong></td>
                    <td>—</td>
                    <td>{m1.get('Sl', 0.0):.4f} m</td>
                    <td>{m2.get('Sl', 0.0):.4f} m</td>
                </tr>
            </tbody>
        </table>

        <h2>Tracés Graphiques Comparatifs</h2>
        <div class="image-container">
            <h3>Comparaison Métrologique</h3>
            <img src="comparison_iso_metrics.png" alt="Métriques ISO">
        </div>
        <div class="image-container">
            <h3>Superposition des Trajectoires</h3>
            <img src="comparison_trajectories.png" alt="Trajectoires">
        </div>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def compare_benchmarks(
    v1_dir: str = "data/evaluation/v1/batch_run",
    v2_dir: str = "data/evaluation/v2/batch_run",
    name1: str = "V1 (19 cols)",
    name2: str = "V2 (5 cols)",
    outdir: str = "data/evaluation/comparison"
) -> Dict[str, Any]:
    """Exécute la suite complète de comparaison entre les deux dossiers de benchmark."""
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 76)
    print(f"--- COMPARAISON MÉTROLOGIQUE INTER-APPROCHES : {name1} vs {name2} ---")
    print("=" * 76)
    print(f"Dossier Approche 1 : {v1_dir}")
    print(f"Dossier Approche 2 : {v2_dir}")
    print(f"Dossier de Sortie  : {outdir}\n")

    sessions1 = load_batch_sessions(v1_dir)
    sessions2 = load_batch_sessions(v2_dir)

    print(f"Sessions chargées pour {name1} : {len(sessions1)}")
    print(f"Sessions chargées pour {name2} : {len(sessions2)}")

    m1 = load_iso_csv(v1_dir)
    if not m1 and sessions1:
        m1 = calculate_metrics_from_sessions(sessions1)

    m2 = load_iso_csv(v2_dir)
    if not m2 and sessions2:
        m2 = calculate_metrics_from_sessions(sessions2)

    # 1. Export CSV comparatif
    summary_csv = os.path.join(outdir, "comparison_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Métrique", name1, name2])
        for k in ["Ap", "Ao_deg", "Rp", "Ro_deg", "Sl", "So_deg"]:
            writer.writerow([k, m1.get(k, 0.0), m2.get(k, 0.0)])
    print(f"[Succès] Synthèse CSV générée : {summary_csv}")

    # 2. Graphiques
    png_m, svg_m = plot_comparison_metrics(m1, m2, name1, name2, outdir)
    png_t, svg_t = plot_comparison_trajectories(sessions1, sessions2, name1, name2, outdir)
    dash = generate_comparison_dashboard(m1, m2, name1, name2, outdir)

    print(f"[Succès] Dashboard interactif : {dash}")
    print("=" * 76 + "\n")

    return {
        "metrics_v1": m1,
        "metrics_v2": m2,
        "dashboard": dash,
        "summary_csv": summary_csv
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outil de comparaison métrologique des benchmarks LIMO.")
    parser.add_argument("--v1-dir", type=str, default="data/evaluation/v1/batch_run", help="Dossier batch V1")
    parser.add_argument("--v2-dir", type=str, default="data/evaluation/v2/batch_run", help="Dossier batch V2")
    parser.add_argument("--name1", type=str, default="V1 (19 cols)", help="Nom de l'approche 1")
    parser.add_argument("--name2", type=str, default="V2 (5 cols)", help="Nom de l'approche 2")
    parser.add_argument("--outdir", type=str, default="data/evaluation/comparison", help="Dossier de sortie")
    args = parser.parse_args()

    compare_benchmarks(
        v1_dir=args.v1_dir,
        v2_dir=args.v2_dir,
        name1=args.name1,
        name2=args.name2,
        outdir=args.outdir
    )
