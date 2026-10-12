"""
Visualiseur et Traceur de Trajectoires du Jeu de Données (Dataset Trajectory Plotter)
=====================================================================================
Génère les graphiques de toutes les trajectoires synthétisées dans le dataset hors ligne :
1. Carte Globale 2D (toutes les 240 trajectoires convergeant vers [0, 0, 0]).
2. Cartes Décomposées par Quadrants (Q1, Q2, Q3, Q4 et Approches Axiales).
3. Graphiques Individuels par Trajectoire (XY, erreurs temporelles, vitesses v_lin et v_ang).

Usage :
    python3 -m src.data.plot_dataset_trajectories
    python3 -m src.data.plot_dataset_trajectories --mode all --max-individual 240

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import csv
import math
import argparse
from typing import List, Dict, Any

try:
    import matplotlib
    matplotlib.use("Agg")  # Backend sans affichage X11 pour génération de fichiers
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_trajectories(csv_path: str) -> List[List[Dict[str, Any]]]:
    """Charge les trajectoires à partir du fichier CSV du dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier de données introuvable : {csv_path}")

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
    """Détermine la catégorie (Q1, Q2, Q3, Q4 ou Axial) d'une trajectoire."""
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


def plot_global_map(trajectories: List[List[Dict[str, Any]]], output_path: str) -> None:
    """Génère la carte 2D globale de toutes les trajectoires."""
    fig, ax = plt.subplots(figsize=(12, 12))

    color_map = {
        "Q1": "#1f77b4",     # Bleu
        "Q2": "#ff7f0e",     # Orange
        "Q3": "#9467bd",     # Violet
        "Q4": "#2ca02c",     # Vert
        "Axial": "#d62728",  # Rouge
    }

    category_counts = {cat: 0 for cat in color_map}

    for traj in trajectories:
        cat = determine_category(traj)
        category_counts[cat] += 1
        color = color_map.get(cat, "#7f7f7f")

        xs = [float(r["x"]) for r in traj]
        ys = [float(r["y"]) for r in traj]

        # Tracé de la courbe continue
        ax.plot(xs, ys, color=color, alpha=0.55, linewidth=1.6)

        # Point de départ
        ax.plot(xs[0], ys[0], "o", color=color, markersize=3.5, alpha=0.8)

        # Flèche d'orientation initiale (direction du cap)
        th0 = float(traj[0]["theta"])
        arrow_len = 0.12
        dx = arrow_len * math.cos(th0)
        dy = arrow_len * math.sin(th0)
        ax.arrow(xs[0], ys[0], dx, dy, color="black", width=0.015, head_width=0.06, head_length=0.05, alpha=0.6)

    # Cible [0, 0, 0]
    ax.plot(0, 0, "r*", markersize=18, label="Cible d'accostage [0, 0, 0]", zorder=10)
    circle = plt.Circle((0, 0), 0.05, color="red", fill=False, linestyle="--", linewidth=1.5, label="Zone d'arrêt (d < 5 cm)")
    ax.add_patch(circle)

    # Légendes par catégorie
    for cat, col in color_map.items():
        count = category_counts[cat]
        ax.plot([], [], color=col, linewidth=2.5, label=f"{cat} ({count} trajectoires)")

    ax.set_title(f"Carte Globale des Trajectoires du Dataset ({len(trajectories)} Trajectoires Flottantes)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Succès] Carte globale sauvegardée dans : {output_path}")


def plot_by_quadrants(trajectories: List[List[Dict[str, Any]]], output_path: str) -> None:
    """Génère une planche décomposée par quadrant."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    categories = ["Q1", "Q2", "Q3", "Q4", "Axial"]
    titles = [
        "Quadrant 1 (+X, +Y)",
        "Quadrant 2 (-X, +Y)",
        "Quadrant 3 (-X, -Y)",
        "Quadrant 4 (+X, -Y)",
        "Approches Axiales (X=0 ou Y=0)"
    ]
    colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]

    for idx, (cat, title, col) in enumerate(zip(categories, titles, colors)):
        ax = axes[idx]
        cat_trajs = [t for t in trajectories if determine_category(t) == cat]

        for t in cat_trajs:
            xs = [float(r["x"]) for r in t]
            ys = [float(r["y"]) for r in t]
            ax.plot(xs, ys, color=col, alpha=0.65, linewidth=1.8)
            ax.plot(xs[0], ys[0], "o", color="darkgreen", markersize=4)

            # Flèche de cap initial
            th0 = float(t[0]["theta"])
            dx = 0.15 * math.cos(th0)
            dy = 0.15 * math.sin(th0)
            ax.arrow(xs[0], ys[0], dx, dy, color="black", width=0.015, head_width=0.06, head_length=0.05, alpha=0.7)

        ax.plot(0, 0, "r*", markersize=14, zorder=5)
        ax.add_patch(plt.Circle((0, 0), 0.05, color="red", fill=False, linestyle="--", linewidth=1.2))

        ax.set_title(f"{title} ({len(cat_trajs)} essais)", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.6)
        ax.set_aspect("equal", adjustable="datalim")

    # Masquer le 6e panneau vide
    axes[5].axis("off")
    fig.suptitle("Décomposition des Trajectoires par Région d'Origine", fontsize=15, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Succès] Planche par quadrants sauvegardée dans : {output_path}")


def plot_single_trajectory(traj: List[Dict[str, Any]], traj_id: int, output_dir: str) -> None:
    """Génère une vue détaillée à 3 sous-graphiques pour une trajectoire individuelle."""
    x0 = float(traj[0]["x"])
    y0 = float(traj[0]["y"])
    th0 = float(traj[0]["theta"])
    cat = determine_category(traj)

    t = [float(r["timestamp"]) for r in traj]
    x = [float(r["x"]) for r in traj]
    y = [float(r["y"]) for r in traj]
    th = [float(r["theta"]) for r in traj]
    dist = [math.hypot(float(r["e_x"]), float(r["e_y"])) for r in traj]
    eth = [float(r["e_theta"]) for r in traj]
    vl = [float(r["v_lin"]) for r in traj]
    va = [float(r["v_ang"]) for r in traj]

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"Trajectoire #{traj_id:03d} [{cat}] : Départ ({x0:.2f} m, {y0:.2f} m, {th0:.2f} rad) -> Cible [0, 0, 0]",
        fontsize=12, fontweight="bold"
    )
    gs = GridSpec(2, 2, figure=fig)

    # 1. Tracé Cartésien XY
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(x, y, "b-", linewidth=2.0, label="Trajectoire LIMO")
    ax1.plot(x0, y0, "go", markersize=9, label=f"Départ ({x0:.2f}, {y0:.2f})")
    ax1.plot(0, 0, "r*", markersize=14, label="Cible (0, 0)")
    ax1.add_patch(plt.Circle((0, 0), 0.05, color="red", fill=False, linestyle="--", linewidth=1.2, label="Vaga (5 cm)"))

    # Flèches de direction le long de la trajectoire (tous les 20 pas)
    for i in range(0, len(x), max(1, len(x) // 8)):
        dx = 0.08 * math.cos(th[i])
        dy = 0.08 * math.sin(th[i])
        ax1.arrow(x[i], y[i], dx, dy, color="navy", width=0.01, head_width=0.04, head_length=0.03, alpha=0.7)

    ax1.set_title("Trajectoire Cartésienne (Plan XY)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_aspect("equal", adjustable="datalim")

    # 2. Erreurs vs Temps
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, dist, "r-", linewidth=1.5, label="Distance d(t) [m]")
    ax2.plot(t, eth, "orange", linewidth=1.5, label="Erreur angulaire [rad]")
    ax2.axhline(0.05, color="red", linestyle=":", label="Seuil accostage (5 cm)")
    ax2.set_title("Évolution des Erreurs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Temps (s)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Consignes de Vitesse vs Temps
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, vl, "g-", linewidth=1.5, label="Vitesse linéaire v_lin [m/s]")
    ax3.plot(t, va, "m-", linewidth=1.5, label="Vitesse angulaire v_ang [rad/s]")
    ax3.axhline(0.14, color="gray", linestyle=":", alpha=0.7, label="V_min croisière (0.14 m/s)")
    ax3.set_title("Profil des Vitesses", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Temps (s)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    slug = f"traj_{traj_id:03d}_{cat.lower()}_{x0:+.1f}_{y0:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
    png_path = os.path.join(output_dir, f"{slug}.png")
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Traceur des trajectoires du dataset LIMO.")
    parser.add_argument("--dataset", type=str, default="data/curated/handcrafted_dataset.csv", help="Chemin du CSV")
    parser.add_argument("--output-dir", type=str, default="data/curated/plots", help="Dossier de sortie des graphiques")
    parser.add_argument("--mode", type=str, choices=["overview", "individual", "all"], default="all",
                        help="Mode de tracé : 'overview' (cartes globales), 'individual' (par trajectoire), 'all' (les deux)")
    parser.add_argument("--max-individual", type=int, default=240, help="Nombre max de graphiques individuels à générer")
    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("[Erreur] matplotlib n'est pas disponible dans l'environnement courant.")
        return

    print("=" * 65)
    print("--- TRACÉ DES TRAJECTOIRES DU JEU DE DONNÉES ---")
    print(f"Dataset      : {args.dataset}")
    print(f"Destination  : {args.output_dir}")
    print(f"Mode         : {args.mode}")
    print("=" * 65)

    trajectories = load_trajectories(args.dataset)
    print(f"[Info] {len(trajectories)} trajectoires chargées avec succès.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ("overview", "all"):
        global_path = os.path.join(args.output_dir, "dataset_all_trajectories.png")
        plot_global_map(trajectories, global_path)

        quadrants_path = os.path.join(args.output_dir, "dataset_by_quadrant.png")
        plot_by_quadrants(trajectories, quadrants_path)

    if args.mode in ("individual", "all"):
        indiv_dir = os.path.join(args.output_dir, "individual")
        os.makedirs(indiv_dir, exist_ok=True)
        limit = min(len(trajectories), args.max_individual)
        print(f"[Info] Génération de {limit} graphiques individuels dans {indiv_dir}...")
        for i in range(limit):
            plot_single_trajectory(trajectories[i], i + 1, indiv_dir)
            if (i + 1) % 40 == 0 or (i + 1) == limit:
                print(f"  -> Trajectoire {i + 1}/{limit} générée.")

    print("\n[Terminé] Tous les graphiques ont été générés avec succès !")


if __name__ == "__main__":
    main()
