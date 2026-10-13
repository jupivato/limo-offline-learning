"""
Enregistreur de Télémétrie et Métriques (DataLogger)
===================================================
Enregistre la trajectoire, les erreurs, les consignes et les métriques
à chaque itération de 20 Hz. Compatible avec le moteur de calcul ISO 18646-2.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import sys
import csv
import time
import math
from typing import List, Dict, Any, Optional

# Inclusion automatique du chemin dist-packages snap si présent
for _p in [
    os.path.expanduser("~/local/lib/python3.14/dist-packages"),
    os.path.expanduser("~/snap/antigravity-cli/common/local/lib/python3.14/dist-packages")
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

LIMO_HALF_TRACK = 0.086  # Demi-voie : 86 mm


class DataLogger:
    """
    Gestionnaire d'enregistrement des données de télémétrie en temps réel.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._rows: List[Dict[str, Any]] = []

    def log(
        self,
        position: List[float],
        target: List[float],
        command: List[float],
        grad: Optional[List[float]] = None,
        cost: float = 0.0
    ) -> None:
        """
        Enregistre un échantillon à l'instant courant.

        Args:
            position (List[float]): Pose courante [x, y, theta].
            target (List[float]): Pose cible [tx, ty, ttheta].
            command (List[float]): Commande appliquée [v_lin, v_ang].
            grad (List[float], optional): Gradient [grad_0, grad_1] (défaut [0.0, 0.0]).
            cost (float): Valeur du critère de coût à l'instant courant.
        """
        if self._start_time is None:
            self._start_time = time.time()

        if grad is None:
            grad = [0.0, 0.0]

        t = time.time() - self._start_time
        x, y, theta = position
        tx, ty, ttheta = target
        v_lin, v_ang = command[0], command[1]

        # Erreur angulaire ramenée dans (-pi, pi]
        e_theta_raw = (theta - ttheta) % (2.0 * math.pi)
        if e_theta_raw > math.pi:
            e_theta_raw -= 2.0 * math.pi

        self._rows.append({
            "timestamp": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "theta": round(theta, 4),
            "e_x": round(x - tx, 4),
            "e_y": round(y - ty, 4),
            "e_theta": round(e_theta_raw, 4),
            "v_lin": round(v_lin, 4),
            "v_ang": round(v_ang, 4),
            "v_wheel_left": round(v_lin - v_ang * LIMO_HALF_TRACK, 4),
            "v_wheel_right": round(v_lin + v_ang * LIMO_HALF_TRACK, 4),
            "grad_0": round(grad[0], 6),
            "grad_1": round(grad[1], 6),
            "cost": round(cost, 6),
        })

    def save(self, filename: str) -> None:
        """
        Exporte les données enregistrées dans un fichier CSV.

        Args:
            filename (str): Chemin du fichier CSV de sortie.
        """
        if not self._rows:
            print("[DataLogger] Aucune donnée à enregistrer.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)

        print(f"[DataLogger] Télémétrie sauvegardée dans : {filename} ({len(self._rows)} itérations)")

    def save_plot(self, png_path: str, show: bool = False) -> None:
        """Génère la planche graphique session 3x3 identique à celle d'APP-Limo-ros2."""
        if not self._rows:
            return
        try:
            import sys
            import matplotlib
            if "matplotlib.pyplot" not in sys.modules:
                matplotlib.use("Qt5Agg" if show else "Agg")
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except Exception:
            print("[DataLogger] matplotlib n'est pas disponible — génération graphique ignorée.")
            return

        t = [r["timestamp"] for r in self._rows]
        x = [r["x"] for r in self._rows]
        y = [r["y"] for r in self._rows]
        th = [r["theta"] for r in self._rows]
        ex = [r["e_x"] for r in self._rows]
        ey = [r["e_y"] for r in self._rows]
        et = [r["e_theta"] for r in self._rows]
        vl = [r["v_lin"] for r in self._rows]
        va = [r["v_ang"] for r in self._rows]
        wl = [r["v_wheel_left"] for r in self._rows]
        wr = [r["v_wheel_right"] for r in self._rows]
        g0 = [r["grad_0"] for r in self._rows]
        g1 = [r["grad_1"] for r in self._rows]
        co = [r["cost"] for r in self._rows]

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(os.path.basename(png_path), fontsize=11)
        gs = GridSpec(3, 3, figure=fig)

        ax = fig.add_subplot(gs[0:2, 0])
        ax.plot(x, y, "b-", linewidth=1.5, label="trajectoire")
        ax.plot(x[0], y[0], "go", markersize=8, label="départ")
        ax.plot(x[-1], y[-1], "s", color="#e74c3c", markersize=7, label="arrivée")
        ax.plot(0, 0, "r*", markersize=14, label="cible (0,0)")

        # Flèche d'orientation initiale (cap départ)
        arr_len = 0.20
        ax.annotate("", xy=(x[0] + arr_len * math.cos(th[0]), y[0] + arr_len * math.sin(th[0])), xytext=(x[0], y[0]),
                    arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2.5, mutation_scale=16))

        # Flèche d'orientation finale (cap arrivée)
        ax.annotate("", xy=(x[-1] + arr_len * math.cos(th[-1]), y[-1] + arr_len * math.sin(th[-1])), xytext=(x[-1], y[-1]),
                    arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.5, mutation_scale=16))

        # Flèche d'orientation désirée à la cible (theta=0)
        ax.annotate("", xy=(0.20, 0.0), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color="#f39c12", lw=2.0, mutation_scale=14))

        ax.set_title("Trajectoire XY & Caps (Départ/Arrivée)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="best", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="datalim")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, co, "r-", linewidth=1)
        ax2.set_title("Coût")
        ax2.set_xlabel("t (s)")
        ax2.grid(True)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t, ex, label="e_x")
        ax3.plot(t, ey, label="e_y")
        ax3.plot(t, et, label="e_θ")
        ax3.set_title("Erreurs")
        ax3.set_xlabel("t (s)")
        ax3.legend()
        ax3.grid(True)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t, vl, label="v_lin")
        ax4.plot(t, va, label="v_ang")
        ax4.set_title("Vitesses robot")
        ax4.set_xlabel("t (s)")
        ax4.legend()
        ax4.grid(True)

        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(t, wl, label="gauche")
        ax5.plot(t, wr, label="droite")
        ax5.set_title("Vitesses roues (m/s)")
        ax5.set_xlabel("t (s)")
        ax5.legend()
        ax5.grid(True)

        ax6 = fig.add_subplot(gs[2, 1])
        dists = [math.hypot(xi, yi) for xi, yi in zip(x, y)]
        ax6.plot(t, dists, color="#8e44ad", linewidth=1.5, label="distance (m)")
        ax6.axhline(y=0.20, color="#e67e22", linestyle="--", label="freinage (0.2m)")
        ax6.axhline(y=0.05, color="#27ae60", linestyle="-.", label="arrêt (0.05m)")
        ax6.set_title("Distance à la Cible (m)")
        ax6.set_xlabel("t (s)")
        ax6.legend(fontsize=8)
        ax6.grid(True)

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(t, x, label="x")
        ax7.plot(t, y, label="y")
        ax7.plot(t, th, label="θ")
        ax7.set_title("Position")
        ax7.set_xlabel("t (s)")
        ax7.legend()
        ax7.grid(True)

        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"[DataLogger] Graphique sauvegardé dans : {png_path}")

    def reset(self) -> None:
        """Réinitialise les accumulateurs de télémétrie."""
        self._start_time = None
        self._rows = []
