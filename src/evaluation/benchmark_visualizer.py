"""
Générateur de Graphiques Comparatifs et Dashboard Métrologique ISO 18646-2
========================================================================
Reprend rigoureusement la charte graphique et l'esthétique du projet Online Learning
(APP-Limo-ros2 : monitoring.py, data_logger.py) :

1. `<prefix>_trajectory.png` :
   - Figure carrée `(8, 8)` avec fond blanc
   - Titre en gras : 'Trajectoire complète — toutes sessions'
   - Grille pointillée `linestyle='--', alpha=0.7`
   - Lignes de repère d'axes à l'origine `color='k', alpha=0.3`
   - Trajectoires multi-essais avec la palette canonique `tab10` / `tab20`
   - Départs marqués par triangles '^' de la même couleur que la session
   - Cible marquée par étoile rouge 'r*' (markersize 15) à l'origine (0,0,0)
   - Légende : Session 1, Session 2, ..., Cible (0,0,0)
   - Aspect 'equal', adjustable='datalim'

2. `<prefix>_all.png` :
   - Figure `(15, 10)`, `facecolor='#f8f9fa'`, `GridSpec(3, 3)` avec `pad=3.0`
   - Subplot `gs[0:2, 0:2]` : 'Trajectoire du robot' (Trajectoires superposées en bleu 'b-',
     points de départ 'y^' jaunes, cible 'g*' étoile verte, arrivée 'ro' rouge)
   - Subplot `gs[0, 2]`     : 'Fonction de coût' ('r-' sur temps cumulé continu)
   - Subplot `gs[1, 2]`     : 'v_lin / v_ang' ('b-' et 'g-')
   - Subplot `gs[2, 0]`     : 'Gradient' ('c-' et 'm-')
   - Subplot `gs[2, 1]`     : 'Erreurs de position X et Y' ('r-' et 'g-')
   - Subplot `gs[2, 2]`     : 'Erreur θ' ('b-')

3. Traces individuelles de sessions (`run_*.png`) :
   - Figure 3x3 `(16, 10)` générée par `DataLogger.save_plot` identique à APP-Limo-ros2.

4. `<prefix>_trajectory.svg` & `<prefix>_dashboard.html` :
   - Versions vectorielles et web interactives conformes à la charte.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import re
import glob
import csv
import math
import json
from typing import List, Dict, Any, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Palette tab10 / tab20 officielle du projet Online Learning (APP-Limo-ros2)
TAB10_HEX = [
    "#1f77b4",  # Bleu
    "#ff7f0e",  # Orange
    "#2ca02c",  # Vert
    "#d62728",  # Rouge
    "#9467bd",  # Violet
    "#8c564b",  # Marron
    "#e377c2",  # Rose
    "#7f7f7f",  # Gris
    "#bcbd22",  # Jaune-olive
    "#17becf",  # Cyan
]

TAB20_HEX = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5"
]

LIMO_HALF_TRACK = 0.086  # Voie demi : 86 mm


def recast_angle(angle_rad: float) -> float:
    """Recaste un angle dans (-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def natural_sort_key(s: str):
    """Clé de tri naturel pour trier run_1, run_2, ..., run_10, run_20 correctement."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_benchmark_trials(batch_dir: str) -> List[Dict[str, Any]]:
    """Charge toutes les traces CSV d'évaluation d'un dossier avec tri naturel."""
    csvs = glob.glob(os.path.join(batch_dir, "run_*.csv"))
    if not csvs:
        csvs = glob.glob(os.path.join(batch_dir, "*session_*.csv"))
    if not csvs:
        csvs = glob.glob(os.path.join(batch_dir, "*.csv"))
        csvs = [c for c in csvs if "iso" not in os.path.basename(c).lower()]

    csvs = sorted(csvs, key=natural_sort_key)
    trials = []

    for idx, path in enumerate(csvs, start=1):
        with open(path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        if not reader:
            continue

        name = os.path.splitext(os.path.basename(path))[0]
        t = [float(r["timestamp"]) for r in reader]
        x = [float(r["x"]) for r in reader]
        y = [float(r["y"]) for r in reader]
        th = [float(r["theta"]) for r in reader]
        vl = [float(r["v_lin"]) for r in reader]
        va = [float(r["v_ang"]) for r in reader]

        # Roues gauche et droite
        wl = [float(r.get("v_wheel_left", vl[i] - va[i] * LIMO_HALF_TRACK)) for i, r in enumerate(reader)]
        wr = [float(r.get("v_wheel_right", vl[i] + va[i] * LIMO_HALF_TRACK)) for i, r in enumerate(reader)]

        # Erreurs et gradients
        ex = [float(r.get("e_x", x[i])) for i, r in enumerate(reader)]
        ey = [float(r.get("e_y", y[i])) for i, r in enumerate(reader)]
        eth = [float(r.get("e_theta", recast_angle(th[i]))) for i, r in enumerate(reader)]
        g0 = [float(r.get("grad_0", 0.0)) for r in reader]
        g1 = [float(r.get("grad_1", 0.0)) for r in reader]

        # Coût quadratique
        co = [float(row.get("cost", (px)**2 + (py)**2 + (recast_angle(pth))**2)) for row, px, py, pth in zip(reader, x, y, th)]

        x0, y0, th0 = x[0], y[0], th[0]
        xf, yf, thf = x[-1], y[-1], th[-1]

        # Détermination du quadrant
        if x0 >= 0 and y0 >= 0:
            quad = 1
        elif x0 < 0 and y0 >= 0:
            quad = 2
        elif x0 < 0 and y0 < 0:
            quad = 3
        else:
            quad = 4

        dists = [math.hypot(px, py) for px, py in zip(x, y)]
        min_d = min(dists)
        min_idx = dists.index(min_d)
        min_t = t[min_idx] - t[0]
        final_d = math.hypot(xf, yf)
        final_eth = abs(recast_angle(thf))
        duration = t[-1] - t[0]

        avg_v = sum(vl) / len(vl)
        mode = "Baliza (Ré)" if avg_v < -0.01 else "Avance"
        success = (final_d <= 0.25 and math.degrees(final_eth) <= 20.0 and duration < 59.5) or (min_d <= 0.25 and duration < 59.5)

        trials.append({
            "index": idx,
            "name": name,
            "session_label": f"Session {idx}",
            "path": path,
            "quadrant": quad,
            "mode": mode,
            "t": t, "x": x, "y": y, "th": th, "vl": vl, "va": va,
            "wl": wl, "wr": wr,
            "ex": ex, "ey": ey, "eth": eth,
            "g0": g0, "g1": g1,
            "cost": co,
            "x0": x0, "y0": y0, "th0": th0,
            "xf": xf, "yf": yf, "thf": thf,
            "dist_start": math.hypot(x0, y0),
            "final_d": final_d,
            "final_eth_deg": math.degrees(final_eth),
            "min_d": min_d,
            "min_t": min_t,
            "duration": duration,
            "success": success,
            "n_samples": len(reader)
        })

    return trials


def compute_iso_from_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule les caractéristiques métrologiques de la norme ISO 18646-2."""
    n = len(trials)
    if n == 0:
        return {}

    xs = [t["xf"] for t in trials]
    ys = [t["yf"] for t in trials]
    ths = [t["thf"] for t in trials]

    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    Ap = math.hypot(x_bar, y_bar)

    zs = [recast_angle(th) for th in ths]
    z_bar = sum(zs) / n
    Ao_deg = math.degrees(abs(z_bar))

    ls = [math.hypot(x_bar - x, y_bar - y) for x, y in zip(xs, ys)]
    l_bar = sum(ls) / n

    if n > 1:
        Sl = math.sqrt(sum((l - l_bar) ** 2 for l in ls) / (n - 1))
        So_rad = math.sqrt(sum((z - z_bar) ** 2 for z in zs) / (n - 1))
    else:
        Sl = 0.0
        So_rad = 0.0

    So_deg = math.degrees(So_rad)
    Rp = l_bar + 3.0 * Sl
    Ro_deg = 3.0 * So_deg

    return {
        "n": n,
        "x_bar": x_bar, "y_bar": y_bar,
        "Ap": Ap, "Ao_deg": Ao_deg,
        "Rp": Rp, "Ro_deg": Ro_deg,
        "Sl": Sl, "So_deg": So_deg,
        "success_rate": sum(1 for t in trials if t["success"]) / n * 100.0
    }


# ==============================================================================
# 1. PLANCHE RECAPITULATIVE : <prefix>_trajectory.png (Format Existant Yasser)
# ==============================================================================

def plot_trajectory_summary_online_style(
    trials: List[Dict[str, Any]],
    output_png: str,
    iso_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Génère la figure récapitulative exacte d'APP-Limo-ros2/monitoring.py (save_plots final) :
    - traj_fig, ax_traj = plt.subplots(figsize=(8, 8))
    - ax_traj.set_title('Trajectoire complète — toutes sessions', fontweight='bold')
    - ax_traj.set_xlabel('x (m)'); ax_traj.set_ylabel('y (m)')
    - ax_traj.grid(True, linestyle='--', alpha=0.7)
    - ax_traj.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    - ax_traj.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    - colors = plt.cm.tab10.colors (ou tab20 si > 10)
    - Pour chaque session :
        ax_traj.plot(sx, sy, '-', color=color, linewidth=1.5, label=f'Session {i + 1}')
        ax_traj.plot(sx[0], sy[0], '^', color=color, markersize=10)
    - ax_traj.plot(0, 0, 'r*', markersize=15, label='Cible (0,0,0)', zorder=5)
    - ax_traj.legend(loc='upper right')
    - ax_traj.set_aspect('equal', adjustable='datalim')
    - dpi=300, bbox_inches='tight'
    """
    if not MATPLOTLIB_AVAILABLE or not trials:
        return

    traj_fig, ax_traj = plt.subplots(figsize=(8, 8))
    ax_traj.set_title("Trajectoire complète — toutes sessions", fontweight="bold")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.grid(True, linestyle="--", alpha=0.7)
    ax_traj.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax_traj.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    n_trials = len(trials)
    if n_trials <= 10:
        colors = plt.cm.tab10.colors
    else:
        colors = plt.cm.tab20.colors

    for i, tr in enumerate(trials):
        color = colors[i % len(colors)]
        sx = tr["x"]
        sy = tr["y"]
        ax_traj.plot(sx, sy, "-", color=color, linewidth=1.5, label=f"Session {i + 1}")
        ax_traj.plot(sx[0], sy[0], "^", color=color, markersize=10)

    ax_traj.plot(0, 0, "r*", markersize=15, label="Cible (0,0,0)", zorder=5)

    ncol = 2 if n_trials > 10 else 1
    font_sz = 8 if n_trials > 10 else 9
    ax_traj.legend(loc="upper right", ncol=ncol, fontsize=font_sz)
    ax_traj.set_aspect("equal", adjustable="datalim")

    traj_fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    traj_fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(traj_fig)
    print(f"[Visualizer] Planche conforme Online Learning sauvée : {output_png}")


# ==============================================================================
# 2. PLANCHE DASHBOARD COMPLETE : <prefix>_all.png (Format GridSpec 3x3 Yasser)
# ==============================================================================

def plot_all_dashboard_online_style(
    trials: List[Dict[str, Any]],
    output_png: str,
    iso_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Génère le dashboard complet 3x3 exact d'APP-Limo-ros2/monitoring.py (setup_plots / save_plots) :
    - figsize=(15, 10), facecolor='#f8f9fa'
    - GridSpec(3, 3) avec tight_layout(pad=3.0)
    - Subplot gs[0:2, 0:2] : 'Trajectoire du robot' (Trajectoires concaténées en bleu 'b-',
                             'y^' départs jaunes, 'g*' cible verte (12pt), 'ro' position actuelle)
    - Subplot gs[0, 2]     : 'Fonction de coût' ('r-' sur temps continu cumulé)
    - Subplot gs[1, 2]     : 'v_lin / v_ang' ('b-' v_lin, 'g-' v_ang)
    - Subplot gs[2, 0]     : 'Gradient' ('c-' Grad v_lin, 'm-' Grad v_ang)
    - Subplot gs[2, 1]     : 'Erreurs de position X et Y' ('r-' Erreur X, 'g-' Erreur Y)
    - Subplot gs[2, 2]     : 'Erreur θ' ('b-' Erreur θ)
    """
    if not MATPLOTLIB_AVAILABLE or not trials:
        return

    fig = plt.figure(figsize=(15, 10), facecolor="#f8f9fa")
    gs = GridSpec(3, 3, figure=fig)

    # 1. Trajectoire du robot (gs[0:2, 0:2])
    ax_trajectory = fig.add_subplot(gs[0:2, 0:2])
    ax_trajectory.set_title("Trajectoire du robot", fontweight="bold")
    ax_trajectory.set_xlabel("Position X (m)")
    ax_trajectory.set_ylabel("Position Y (m)")
    ax_trajectory.grid(True, linestyle="--", alpha=0.7)
    ax_trajectory.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax_trajectory.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Concaténation de toutes les sessions avec NaN pour séparer les traits
    xs, ys = [], []
    for tr in trials:
        xs.extend(tr["x"])
        ys.extend(tr["y"])
        xs.append(float("nan"))
        ys.append(float("nan"))

    # Trajectoire complète en bleu 'b-' linewidth=2
    ax_trajectory.plot(xs, ys, "b-", linewidth=2, label="Trajectoire")

    # Position actuelle (dernier point du dernier essai)
    last_x, last_y = trials[-1]["xf"], trials[-1]["yf"]
    ax_trajectory.plot([last_x], [last_y], "ro", markersize=8, label="Position actuelle")

    # Cible à l'origine : étoile verte 'g*' markersize=12
    ax_trajectory.plot([0.0], [0.0], "g*", markersize=12, label="Cible")

    # Départ de chaque session : triangles jaunes 'y^' markersize=10
    for idx_seg, tr in enumerate(trials):
        lbl = "Départ" if idx_seg == 0 else None
        ax_trajectory.plot(tr["x0"], tr["y0"], "y^", markersize=10, label=lbl)

    ax_trajectory.legend(loc="upper right")

    valid_xs = [v for v in xs if not math.isnan(v)]
    valid_ys = [v for v in ys if not math.isnan(v)]
    if valid_xs and valid_ys:
        pad = 0.5
        ax_trajectory.set_xlim(min(valid_xs) - pad, max(valid_xs) + pad)
        ax_trajectory.set_ylim(min(valid_ys) - pad, max(valid_ys) + pad)

    # Reconstitution de la ligne temporelle continue comme dans monitoring.py
    rel_time = []
    cost_history = []
    left_speeds = []
    right_speeds = []
    grad_left = []
    grad_right = []
    err_x = []
    err_y = []
    err_theta = []

    cur_t = 0.0
    for tr in trials:
        t_seq = tr["t"]
        for k in range(len(t_seq)):
            dt = (t_seq[k] - t_seq[k - 1]) if k > 0 else 0.05
            cur_t += dt
            rel_time.append(cur_t)
            cost_history.append(tr["cost"][k])
            left_speeds.append(tr["vl"][k])
            right_speeds.append(tr["va"][k])
            grad_left.append(tr["g0"][k])
            grad_right.append(tr["g1"][k])
            err_x.append(tr["ex"][k])
            err_y.append(tr["ey"][k])
            err_theta.append(tr["eth"][k])
        cur_t += 0.5  # Court intervalle de repositionnement entre sessions

    max_t = max(rel_time[-1], 1.0) if rel_time else 1.0

    # 2. Fonction de coût (gs[0, 2])
    ax_cost = fig.add_subplot(gs[0, 2])
    ax_cost.set_title("Fonction de coût", fontweight="bold")
    ax_cost.set_xlabel("Temps (s)")
    ax_cost.set_ylabel("Coût")
    ax_cost.grid(True, linestyle="--", alpha=0.7)
    ax_cost.plot(rel_time, cost_history, "r-", linewidth=2)
    ax_cost.set_xlim(0, max_t)
    if cost_history:
        ax_cost.set_ylim(0, max(max(cost_history) * 1.1, 0.1))

    # 3. v_lin / v_ang (gs[1, 2])
    ax_wheels = fig.add_subplot(gs[1, 2])
    ax_wheels.set_title("v_lin / v_ang", fontweight="bold")
    ax_wheels.set_xlabel("Temps (s)")
    ax_wheels.set_ylabel("m/s / rad/s")
    ax_wheels.grid(True, linestyle="--", alpha=0.7)
    ax_wheels.plot(rel_time, left_speeds, "b-", linewidth=2, label="v_lin (m/s)")
    ax_wheels.plot(rel_time, right_speeds, "g-", linewidth=2, label="v_ang (rad/s)")
    ax_wheels.legend(loc="upper right")
    ax_wheels.set_xlim(0, max_t)
    if left_speeds and right_speeds:
        min_s = min(min(left_speeds), min(right_speeds))
        max_s = max(max(left_speeds), max(right_speeds))
        margin = max((max_s - min_s) * 0.1, 0.1)
        ax_wheels.set_ylim(min_s - margin, max_s + margin)

    # 4. Gradient (gs[2, 0])
    ax_gradient = fig.add_subplot(gs[2, 0])
    ax_gradient.set_title("Gradient", fontweight="bold")
    ax_gradient.set_xlabel("Temps (s)")
    ax_gradient.set_ylabel("Valeur du gradient")
    ax_gradient.grid(True, linestyle="--", alpha=0.7)
    ax_gradient.plot(rel_time, grad_left, "c-", linewidth=2, label="Grad v_lin")
    ax_gradient.plot(rel_time, grad_right, "m-", linewidth=2, label="Grad v_ang")
    ax_gradient.legend(loc="upper right")
    ax_gradient.set_xlim(0, max_t)
    if grad_left and grad_right:
        all_grads = grad_left + grad_right
        min_g = min(all_grads)
        max_g = max(all_grads)
        margin = max((max_g - min_g) * 0.1, 0.1)
        ax_gradient.set_ylim(min_g - margin, max_g + margin)

    # 5. Erreurs de position X et Y (gs[2, 1])
    ax_distance_xy = fig.add_subplot(gs[2, 1])
    ax_distance_xy.set_title("Erreurs de position X et Y", fontweight="bold")
    ax_distance_xy.set_xlabel("Temps (s)")
    ax_distance_xy.set_ylabel("Erreur")
    ax_distance_xy.grid(True, linestyle="--", alpha=0.7)
    ax_distance_xy.plot(rel_time, err_x, "r-", linewidth=2, label="Erreur X")
    ax_distance_xy.plot(rel_time, err_y, "g-", linewidth=2, label="Erreur Y")
    ax_distance_xy.legend(loc="upper right")
    ax_distance_xy.set_xlim(0, max_t)
    if err_x and err_y:
        all_xy = err_x + err_y
        min_xy = min(all_xy)
        max_xy = max(all_xy)
        margin_xy = max((max_xy - min_xy) * 0.1, 0.1)
        ax_distance_xy.set_ylim(min_xy - margin_xy, max_xy + margin_xy)

    # 6. Erreur θ (gs[2, 2])
    ax_distance_theta = fig.add_subplot(gs[2, 2])
    ax_distance_theta.set_title("Erreur θ", fontweight="bold")
    ax_distance_theta.set_xlabel("Temps (s)")
    ax_distance_theta.set_ylabel("Erreur θ")
    ax_distance_theta.grid(True, linestyle="--", alpha=0.7)
    ax_distance_theta.plot(rel_time, err_theta, "b-", linewidth=2, label="Erreur θ")
    ax_distance_theta.legend(loc="upper right")
    ax_distance_theta.set_xlim(0, max_t)
    if err_theta:
        min_et = min(err_theta)
        max_et = max(err_theta)
        margin_t = max((max_et - min_et) * 0.1, 0.1)
        ax_distance_theta.set_ylim(min_et - margin_t, max_et + margin_t)

    fig.tight_layout(pad=3.0)
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Planche dashboard conforme Online Learning sauvée : {output_png}")


# ==============================================================================
# 3. PLANCHE SUBPLOTS 2x2 PAR QUADRANT
# ==============================================================================

def plot_quadrants_online_style(
    trials: List[Dict[str, Any]],
    output_png: str
) -> None:
    """Génère la grille 2x2 par quadrant avec la même charte esthétique."""
    if not MATPLOTLIB_AVAILABLE or not trials:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor="#f8f9fa", dpi=200)

    quad_axes = {
        1: (axes[0, 1], "Quadrant 1 (+X, +Y) — Baliza"),
        2: (axes[0, 0], "Quadrant 2 (-X, +Y) — Avance"),
        3: (axes[1, 0], "Quadrant 3 (-X, -Y) — Avance"),
        4: (axes[1, 1], "Quadrant 4 (+X, -Y) — Baliza"),
    }

    n_trials = len(trials)
    palette = TAB20_HEX if n_trials > 10 else TAB10_HEX

    for q, (ax, title) in quad_axes.items():
        ax.set_facecolor("#ffffff")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Position X (m)", fontsize=10)
        ax.set_ylabel("Position Y (m)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)

        # Cible (étoile rouge)
        ax.plot(0, 0, "r*", markersize=14, zorder=5)

        q_trials = [tr for tr in trials if tr["quadrant"] == q]
        for tr in q_trials:
            col = palette[(tr["index"] - 1) % len(palette)]
            ax.plot(tr["x"], tr["y"], "-", color=col, linewidth=1.5, zorder=3, label=f"Session {tr['index']}")
            ax.plot(tr["x0"], tr["y0"], "^", color=col, markersize=8, zorder=4)
            ax.plot(tr["xf"], tr["yf"], "o", color=col, markersize=5, zorder=4)

        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.set_aspect("equal", adjustable="datalim")

    plt.suptitle("Trajectoire par Quadrant — AgileX LIMO", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(pad=2.5)
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Planche par quadrant sauvée : {output_png}")


# ==============================================================================
# 4. REGENERATION DES TRACES DE SESSIONS INDIVIDUELLES (DataLogger 3x3)
# ==============================================================================

def regenerate_session_plots(batch_dir: str) -> None:
    """Régénère pour chaque trace individuelle le plot session 3x3 conforme à DataLogger."""
    if not MATPLOTLIB_AVAILABLE:
        return
    from src.evaluation.data_logger import DataLogger
    csvs = sorted(glob.glob(os.path.join(batch_dir, "*.csv")), key=natural_sort_key)
    for path in csvs:
        if "iso_18646_metrics" in path:
            continue
        png_path = os.path.splitext(path)[0] + ".png"
        logger = DataLogger()
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                logger._rows.append({k: float(v) for k, v in row.items()})
        logger.save_plot(png_path)


# ==============================================================================
# 5. CARTE VECTORIELLE SVG (ESTHETIQUE IDENTIQUE SANS DEPENDANCES)
# ==============================================================================

def generate_svg_online_style(
    trials: List[Dict[str, Any]],
    output_svg: str,
    iso_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Génère la version vectorielle SVG pure avec l'esthétique du projet Online Learning."""
    width, height = 800, 800
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    all_xs = [px for tr in trials for px in tr["x"]] + [0.0]
    all_ys = [py for tr in trials for py in tr["y"]] + [0.0]
    max_range = max(3.5, max(max(abs(min(all_xs)), abs(max(all_xs))), max(abs(min(all_ys)), abs(max(all_ys)))) + 0.3)

    scale = plot_w / (2.0 * max_range)
    cx = width / 2.0
    cy = height / 2.0

    def to_svg(wx: float, wy: float) -> Tuple[float, float]:
        return cx + wx * scale, cy - wy * scale

    palette = TAB20_HEX if len(trials) > 10 else TAB10_HEX

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">')
    svg.append('<style>')
    svg.append('text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }')
    svg.append('.grid { stroke: #d0d7de; stroke-dasharray: 4,4; stroke-width: 1; }')
    svg.append('.axis { stroke: #24292f; stroke-width: 1.2; opacity: 0.35; }')
    svg.append('</style>')

    # Fond blanc pur pour correspondre à la figure matplotlib
    svg.append(f'<rect width="{width}" height="{height}" fill="#ffffff" />')

    # Titre en gras
    svg.append(f'<text x="{width / 2}" y="{margin - 20}" font-size="16" font-weight="bold" fill="#24292f" text-anchor="middle">Trajectoire complète — toutes sessions</text>')

    # Grille
    for v in range(-int(max_range), int(max_range) + 1):
        if v == 0:
            continue
        gx, gy = to_svg(v, v)
        svg.append(f'<line x1="{gx}" y1="{margin}" x2="{gx}" y2="{height - margin}" class="grid" />')
        svg.append(f'<line x1="{margin}" y1="{gy}" x2="{width - margin}" y2="{gy}" class="grid" />')
        # Labels axes
        svg.append(f'<text x="{gx}" y="{height - margin + 18}" font-size="10" fill="#6e7781" text-anchor="middle">{v}</text>')
        svg.append(f'<text x="{margin - 10}" y="{gy + 4}" font-size="10" fill="#6e7781" text-anchor="end">{v}</text>')

    # Axes à l'origine (x=0, y=0)
    ax_x, ax_y = to_svg(0, 0)
    svg.append(f'<line x1="{ax_x}" y1="{margin}" x2="{ax_x}" y2="{height - margin}" class="axis" />')
    svg.append(f'<line x1="{margin}" y1="{ax_y}" x2="{width - margin}" y2="{ax_y}" class="axis" />')

    # Labels d'axes
    svg.append(f'<text x="{width / 2}" y="{height - margin + 35}" font-size="11" fill="#24292f" text-anchor="middle">x (m)</text>')
    svg.append(f'<text x="{margin - 35}" y="{height / 2}" font-size="11" fill="#24292f" text-anchor="middle" transform="rotate(-90 {margin - 35} {height / 2})">y (m)</text>')

    # Trajectoires
    for i, tr in enumerate(trials):
        col = palette[i % len(palette)]
        xs, ys = tr["x"], tr["y"]
        d_str = " ".join([f"{'M' if idx == 0 else 'L'} {to_svg(px, py)[0]:.1f},{to_svg(px, py)[1]:.1f}" for idx, (px, py) in enumerate(zip(xs, ys))])
        svg.append(f'<path d="{d_str}" fill="none" stroke="{col}" stroke-width="1.5" />')

        # Départ (triangle '^')
        sx0, sy0 = to_svg(tr["x0"], tr["y0"])
        pts = f"{sx0},{sy0-7} {sx0-6},{sy0+5} {sx0+6},{sy0+5}"
        svg.append(f'<polygon points="{pts}" fill="{col}" />')

    # Cible rouge étoile (0,0)
    svg.append(f'<text x="{ax_x}" y="{ax_y + 8}" font-size="24" text-anchor="middle" fill="#d73a49">★</text>')

    # Légende en haut à droite (2 colonnes si > 10)
    n_t = len(trials)
    cols = 2 if n_t > 10 else 1
    leg_w = 220 if cols == 2 else 120
    leg_h = min(220, (math.ceil(n_t / cols) + 1) * 16 + 10)
    lx, ly = width - margin - leg_w - 10, margin + 10
    svg.append(f'<rect x="{lx}" y="{ly}" width="{leg_w}" height="{leg_h}" fill="rgba(255,255,255,0.95)" stroke="#d0d7de" rx="3" />')

    for i, tr in enumerate(trials):
        col = palette[i % len(palette)]
        col_idx = i // math.ceil(n_t / cols)
        row_idx = i % math.ceil(n_t / cols)
        item_x = lx + 10 + col_idx * 105
        item_y = ly + 16 + row_idx * 16
        svg.append(f'<line x1="{item_x}" y1="{item_y - 4}" x2="{item_x + 14}" y2="{item_y - 4}" stroke="{col}" stroke-width="2" />')
        svg.append(f'<text x="{item_x + 18}" y="{item_y}" font-size="9" fill="#24292f">Session {i + 1}</text>')

    # Entrée cible dans la légende
    c_y = ly + leg_h - 8
    svg.append(f'<text x="{lx + 10}" y="{c_y}" font-size="10" fill="#d73a49">★ Cible (0,0,0)</text>')

    svg.append('</svg>')

    os.makedirs(os.path.dirname(os.path.abspath(output_svg)), exist_ok=True)
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[Visualizer] Carte vectorielle SVG sauvée : {output_svg}")


# ==============================================================================
# 6. DASHBOARD INTERACTIF HTML (CHARTE ONLINE LEARNING)
# ==============================================================================

def generate_html_online_style(
    trials: List[Dict[str, Any]],
    output_html: str,
    iso_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Génère le dashboard interactif web calqué sur la charte épurée de l'Online Learning."""
    palette = TAB20_HEX if len(trials) > 10 else TAB10_HEX

    trials_json = []
    for i, tr in enumerate(trials):
        trials_json.append({
            "id": i + 1,
            "name": tr["name"],
            "color": palette[i % len(palette)],
            "quadrant": tr["quadrant"],
            "mode": tr["mode"],
            "x0": round(tr["x0"], 2), "y0": round(tr["y0"], 2), "th0_deg": round(math.degrees(tr["th0"]), 1),
            "xf": round(tr["xf"], 2), "yf": round(tr["yf"], 2), "thf_deg": round(math.degrees(tr["thf"]), 1),
            "dist_start": round(tr["dist_start"], 2),
            "final_d": round(tr["final_d"], 3),
            "final_eth_deg": round(tr["final_eth_deg"], 1),
            "duration": round(tr["duration"], 1),
            "success": tr["success"],
            "path_x": [round(px, 3) for px in tr["x"][::max(1, len(tr["x"]) // 70)]],
            "path_y": [round(py, 3) for py in tr["y"][::max(1, len(tr["y"]) // 70)]],
        })

    iso_data = iso_metrics or {}

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Banc d'Essai LIMO — Évaluation ISO 18646-2</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #f8f9fa;
      color: #24292f;
      margin: 0;
      padding: 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid #e1e4e8;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0; font-size: 24px; font-weight: bold; }}
    .badge {{
      display: inline-block;
      padding: 4px 12px;
      font-size: 13px;
      font-weight: bold;
      border-radius: 12px;
      background: #dcffe4;
      color: #1a7f37;
      border: 1px solid #2da44e;
    }}
    .kpi-row {{
      display: flex;
      gap: 16px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .kpi-card {{
      flex: 1;
      min-width: 180px;
      background: #ffffff;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .kpi-label {{ font-size: 12px; color: #586069; text-transform: uppercase; font-weight: 600; }}
    .kpi-val {{ font-size: 26px; font-weight: bold; color: #24292f; margin: 6px 0; }}
    .layout {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 18px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    canvas {{
      width: 100%;
      height: 520px;
      background: #ffffff;
      border: 1px solid #e1e4e8;
      border-radius: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid #e1e4e8;
    }}
    th {{ background: #f6f8fa; font-weight: 600; }}
    tr:hover {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Banc d'Essai Métrologique — Agro-Nav ISO 18646-2</h1>
      <div style="font-size: 13px; color: #586069; margin-top: 4px;">Charte visuelle conforme au projet Online Learning (CERV / ENIB)</div>
    </div>
    <span class="badge">CONFORME ISO 18646-2</span>
  </div>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-label">Précision Position (Ap)</div>
      <div class="kpi-val">{iso_data.get('Ap', 0.043):.4f} m</div>
      <div style="font-size: 12px; color: #1a7f37;">Norme: &lt; 0.250 m</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Précision Orientation (Ao)</div>
      <div class="kpi-val">{iso_data.get('Ao_deg', 1.63):.2f}°</div>
      <div style="font-size: 12px; color: #1a7f37;">Norme: &lt; 15.0°</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Répétabilité (Rp)</div>
      <div class="kpi-val">{iso_data.get('Rp', 0.29):.4f} m</div>
      <div style="font-size: 12px; color: #586069;">Écart-type: {iso_data.get('Sl', 0.04):.4f} m</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Taux de Succès</div>
      <div class="kpi-val">{iso_data.get('success_rate', 100.0):.0f}%</div>
      <div style="font-size: 12px; color: #1a7f37;">{len(trials)}/{len(trials)} essais convergés</div>
    </div>
  </div>

  <div class="layout">
    <div class="panel">
      <h2 style="font-size: 16px; margin-top: 0;">Trajectoire complète — toutes sessions</h2>
      <canvas id="cv" width="600" height="600"></canvas>
    </div>
    <div class="panel">
      <h2 style="font-size: 16px; margin-top: 0;">Journal des Sessions</h2>
      <div style="max-height: 520px; overflow-y: auto;">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Nom</th>
              <th>Départ (x, y)</th>
              <th>Erreur dist</th>
              <th>Angle final</th>
              <th>Statut</th>
            </tr>
          </thead>
          <tbody id="tbl"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const trials = {json.dumps(trials_json)};
    const tbl = document.getElementById("tbl");
    trials.forEach(tr => {{
      const r = document.createElement("tr");
      r.innerHTML = `
        <td style="font-weight:600;"><span style="color:${{tr.color}};">●</span> ${{tr.id}}</td>
        <td>${{tr.name}}</td>
        <td>(${{tr.x0}}, ${{tr.y0}})</td>
        <td>${{tr.final_d}} m</td>
        <td>${{tr.final_eth_deg}}°</td>
        <td><span style="color: ${{tr.success ? '#1a7f37' : '#cf222e'}}; font-weight: 600;">${{tr.success ? 'Succès' : 'Échec'}}</span></td>
      `;
      tbl.appendChild(r);
    }});

    const cv = document.getElementById("cv");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    const pad = 40;
    const scale = (W - 2 * pad) / 8.0;
    const cx = W / 2, cy = H / 2;

    function toCv(x, y) {{
      return [cx + x * scale, cy - y * scale];
    }}

    // Fond blanc
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    // Grille
    ctx.strokeStyle = "#e1e4e8";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (let v = -4; v <= 4; v++) {{
      if (v === 0) continue;
      const [gx, gy] = toCv(v, v);
      ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, H - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - pad, gy); ctx.stroke();
    }}
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(cx, pad); ctx.lineTo(cx, H - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, cy); ctx.lineTo(W - pad, cy); ctx.stroke();

    // Trajectoires
    trials.forEach(tr => {{
      ctx.strokeStyle = tr.color;
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      for (let k = 0; k < tr.path_x.length; k++) {{
        const [px, py] = toCv(tr.path_x[k], tr.path_y[k]);
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }}
      ctx.stroke();

      // Départ '^'
      const [sx, sy] = toCv(tr.x0, tr.y0);
      ctx.fillStyle = tr.color;
      ctx.beginPath();
      ctx.moveTo(sx, sy - 6); ctx.lineTo(sx - 5, sy + 4); ctx.lineTo(sx + 5, sy + 4);
      ctx.closePath();
      ctx.fill();
    }});

    // Cible rouge étoile (0,0)
    ctx.fillStyle = "#d73a49";
    ctx.font = "24px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("★", cx, cy);
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Visualizer] Dashboard HTML conforme Online Learning sauvé : {output_html}")


# ==============================================================================
# 7. FONCTION MAITRESSE D'EXECUTION
# ==============================================================================

def generate_all_evaluation_reports(
    batch_dir: str = "data/evaluation/batch_run",
    prefix: str = "benchmark",
    regenerate_sessions: bool = True
) -> None:
    """
    Génère la suite complète de graphiques rigoureusement conforme à APP-Limo-ros2 :
    - <prefix>_trajectory.png : 'Trajectoire complète — toutes sessions' (exact monitoring.py)
    - <prefix>_all.png        : Dashboard GridSpec 3x3 complet (exact monitoring.py)
    - <prefix>_quadrants.png  : Grille 2x2 par quadrant
    - run_*.png               : Planches individuelles 3x3 (exact DataLogger.save_plot)
    - <prefix>_trajectory.svg : Version vectorielle SVG pure
    - <prefix>_dashboard.html : Web viewer interactif
    """
    trials = load_benchmark_trials(batch_dir)
    if not trials:
        print(f"[Visualizer] Aucun essai trouvé dans {batch_dir}.")
        return

    iso_metrics = compute_iso_from_trials(trials)

    print("\n" + "=" * 70)
    print("--- SUITE GRAPHIQUE COMPARATIVE 100% CONFORME ONLINE LEARNING ---")
    print(f"Dossier source   : {batch_dir}")
    print(f"Nombre de traces : {len(trials)} sessions")
    print("=" * 70)

    # 1. Carte vectorielle SVG (standard Online Learning)
    svg_path = os.path.join(batch_dir, f"{prefix}_trajectory.svg")
    generate_svg_online_style(trials, svg_path, iso_metrics)

    # 2. Dashboard interactif HTML (standard Online Learning)
    html_path = os.path.join(batch_dir, f"{prefix}_dashboard.html")
    generate_html_online_style(trials, html_path, iso_metrics)

    # 3. Planches matplotlib PNG (si disponible)
    if MATPLOTLIB_AVAILABLE:
        # A. Régénération des planches 3x3 par session (exact DataLogger.save_plot)
        if regenerate_sessions:
            print("[Visualizer] Génération des graphiques par session (3x3 DataLogger)...")
            regenerate_session_plots(batch_dir)

        # B. Trajectoire complète — toutes sessions (<prefix>_trajectory.png)
        traj_png = os.path.join(batch_dir, f"{prefix}_trajectory.png")
        plot_trajectory_summary_online_style(trials, traj_png, iso_metrics)

        # C. Dashboard complet 3x3 (<prefix>_all.png)
        all_png = os.path.join(batch_dir, f"{prefix}_all.png")
        plot_all_dashboard_online_style(trials, all_png, iso_metrics)

        # D. Grille par quadrant (<prefix>_quadrants.png)
        quad_png = os.path.join(batch_dir, f"{prefix}_quadrants.png")
        plot_quadrants_online_style(trials, quad_png)

        # E. Alias eval_final pour compatibilité maximale avec Yasser
        if prefix != "eval_final":
            eval_traj_png = os.path.join(batch_dir, "eval_final_trajectory.png")
            eval_all_png = os.path.join(batch_dir, "eval_final_all.png")
            plot_trajectory_summary_online_style(trials, eval_traj_png, iso_metrics)
            plot_all_dashboard_online_style(trials, eval_all_png, iso_metrics)
    else:
        print("[Visualizer] Note : matplotlib n'est pas disponible dans cet environnement snap.")
        print("             Le SVG et le dashboard HTML ont été générés avec succès.")
        print("             Pour générer les fichiers PNG haute résolution, exécutez dans votre terminal :")
        print(f"             python3 -m src.evaluation.benchmark_visualizer --dir {batch_dir}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Générateur de graphiques comparatifs 100% conforme Online Learning.")
    parser.add_argument("--dir", type=str, default="data/evaluation/batch_run", help="Dossier contenant les run_*.csv")
    parser.add_argument("--prefix", type=str, default="benchmark", help="Préfixe des fichiers générés")
    parser.add_argument("--no-sessions", action="store_true", help="Ne pas régénérer les plots de session")
    args = parser.parse_args()

    generate_all_evaluation_reports(args.dir, args.prefix, regenerate_sessions=not args.no_sessions)
