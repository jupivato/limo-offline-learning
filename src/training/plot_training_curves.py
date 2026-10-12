"""
Génération des Graphiques de Convergence de l'Entraînement Hors Ligne
=====================================================================
Visualise l'évolution de la fonction de perte (MSE) pour les sous-ensembles
d'entraînement et de validation, ainsi que la dynamique du taux d'apprentissage.

Génère deux formats :
- Image haute résolution PNG (300 DPI) conforme aux publications académiques ENIB / CERV.
- Fichier vectoriel SVG autonome pour l'intégration web et les rapports interactifs.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Inclusion automatique du dossier racine du projet dans sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Inclusion automatique du chemin dist-packages snap si présent
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


def plot_training_history(
    history_file_path: str = "models/training_history.json",
    output_dir: str = "data/evaluation"
) -> Dict[str, str]:
    """
    Trace et sauvegarde les courbes de convergence de l'entraînement hors ligne.

    Args:
        history_file_path (str): Chemin vers le fichier JSON d'historique.
        output_dir (str): Dossier cible pour sauvegarder les graphiques.

    Returns:
        Dict[str, str]: Chemins des fichiers générés (PNG, SVG).
    """
    if not os.path.exists(history_file_path):
        raise FileNotFoundError(f"Fichier d'historique introuvable : {history_file_path}")

    with open(history_file_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    lr_history = history.get("lr", [])

    if not epochs or not train_loss:
        raise ValueError("L'historique d'entraînement ne contient aucune donnée valide.")

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "training_curves.png")
    svg_path = os.path.join(output_dir, "training_curves.svg")

    # 1. Tracé via Matplotlib (si disponible)
    if MATPLOTLIB_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
        fig.patch.set_facecolor("#ffffff")

        # Sous-graphique 1 : Courbe de Perte MSE
        ax1.set_facecolor("#fafbfc")
        ax1.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
        ax1.plot(epochs, train_loss, label="Perte d'Entraînement (Train MSE)", color="#1f77b4", linewidth=2.0)
        ax1.plot(epochs, val_loss, label="Perte de Validation (Val MSE)", color="#d62728", linewidth=2.0, linestyle="--")

        # Marqueur de la meilleure époque de validation
        min_val_idx = val_loss.index(min(val_loss))
        best_epoch = epochs[min_val_idx]
        best_val = val_loss[min_val_idx]
        ax1.scatter([best_epoch], [best_val], color="#2ca02c", s=80, zorder=5,
                    label=f"Meilleure Val: {best_val:.6f} (Époque {best_epoch})")

        ax1.set_ylabel("Erreur Quadratique Moyenne (MSE)", fontsize=11, fontweight="bold")
        ax1.set_title("Convergence de l'Apprentissage Supervisé — PioneerNN (Dataset V2)",
                      fontsize=13, fontweight="bold", pad=12)
        ax1.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10)

        # Sous-graphique 2 : Taux d'apprentissage (Learning Rate)
        ax2.set_facecolor("#fafbfc")
        ax2.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
        ax2.plot(epochs, lr_history, label="Taux d'Apprentissage (AdamW)", color="#9467bd", linewidth=1.8)
        ax2.set_yscale("log")
        ax2.set_xlabel("Époque d'Entraînement", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Taux d'Apprentissage (LR)", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10)

        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.savefig(svg_path, format="svg")
        plt.close(fig)
        print(f"[Succès] Graphiques PNG et SVG générés : {png_path} | {svg_path}")

    return {
        "png": png_path,
        "svg": svg_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de graphiques d'entraînement pour LIMO.")
    parser.add_argument("--history", type=str, default="models/training_history.json",
                        help="Chemin du fichier JSON d'historique")
    parser.add_argument("--outdir", type=str, default="data/evaluation",
                        help="Dossier de sortie des graphiques")
    args = parser.parse_args()

    plot_training_history(history_file_path=args.history, output_dir=args.outdir)
