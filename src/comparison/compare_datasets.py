"""
Module de Comparaison Statistique des Jeux de Données (Dataset V1 vs Dataset V2)
================================================================================
Compare les distributions cinématiques, le volume d'échantillons et la couverture
spatiale entre différents jeux de données d'apprentissage.

Génère :
- Histogrammes comparatifs des vitesses linéaires et angulaires.
- Nuages de points de la couverture spatiale (x, y).
- Fichier récapitulatif JSON des statistiques descriptives.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

from __future__ import annotations

import os
import sys
import csv
import json
import argparse
from typing import Dict, Any, List, Tuple

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


def inspect_dataset_csv(csv_path: str) -> Dict[str, Any]:
    """Extrait les colonnes et statistiques cinématiques d'un dataset CSV."""
    if not os.path.exists(csv_path):
        return {"exists": False, "count": 0}

    xs: List[float] = []
    ys: List[float] = []
    v_lins: List[float] = []
    v_angs: List[float] = []
    headers: List[str] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return {"exists": True, "count": 0, "headers": []}

        # Identification des indices
        x_idx = -1
        y_idx = -1
        v_idx = -1
        w_idx = -1

        for idx, h in enumerate(headers):
            h_clean = h.strip().lower()
            if h_clean in ["x_r", "x", "pos_x"]:
                x_idx = idx
            elif h_clean in ["y_r", "y", "pos_y"]:
                y_idx = idx
            elif h_clean in ["v_r", "v_lin", "v"]:
                v_idx = idx
            elif h_clean in ["w_r", "v_ang", "w", "omega"]:
                w_idx = idx

        count = 0
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            count += 1
            try:
                if x_idx >= 0:
                    xs.append(float(row[x_idx]))
                if y_idx >= 0:
                    ys.append(float(row[y_idx]))
                if v_idx >= 0:
                    v_lins.append(float(row[v_idx]))
                if w_idx >= 0:
                    v_angs.append(float(row[w_idx]))
            except (ValueError, IndexError):
                continue

    return {
        "exists": True,
        "count": count,
        "headers": headers,
        "num_columns": len(headers),
        "xs": xs,
        "ys": ys,
        "v_lins": v_lins,
        "v_angs": v_angs
    }


def compute_basic_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals)
    }


def compare_datasets(
    v1_csv: str = "data/curated/v1/handcrafted_dataset.csv",
    v2_csv: str = "data/curated/v2/handcrafted_dataset.csv",
    name1: str = "Dataset V1 (19 cols)",
    name2: str = "Dataset V2 (5 cols)",
    outdir: str = "data/evaluation/comparison"
) -> Dict[str, Any]:
    """Compare statistiquement et visuellement deux datasets."""
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 76)
    print(f"--- COMPARAISON STATISTIQUE DES DATASETS : {name1} vs {name2} ---")
    print("=" * 76)
    print(f"Dataset 1 : {v1_csv}")
    print(f"Dataset 2 : {v2_csv}")
    print(f"Sortie    : {outdir}\n")

    d1 = inspect_dataset_csv(v1_csv)
    d2 = inspect_dataset_csv(v2_csv)

    stats = {
        name1: {
            "exists": d1["exists"],
            "echantillons": d1.get("count", 0),
            "colonnes": d1.get("num_columns", 0),
            "v_lin": compute_basic_stats(d1.get("v_lins", [])),
            "v_ang": compute_basic_stats(d1.get("v_angs", []))
        },
        name2: {
            "exists": d2["exists"],
            "echantillons": d2.get("count", 0),
            "colonnes": d2.get("num_columns", 0),
            "v_lin": compute_basic_stats(d2.get("v_lins", [])),
            "v_ang": compute_basic_stats(d2.get("v_angs", []))
        }
    }

    # Export JSON
    json_path = os.path.join(outdir, "dataset_comparison_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"[Succès] Statistiques exportées : {json_path}")

    # Tracés Matplotlib
    if MATPLOTLIB_AVAILABLE and d1.get("count", 0) > 0 and d2.get("count", 0) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        fig.patch.set_facecolor("#ffffff")
        fig.suptitle("Distributions des Consignes de Vitesse (V1 vs V2)", fontsize=13, fontweight="bold")

        # Vitesse linéaire
        ax1 = axes[0]
        ax1.set_facecolor("#fafbfc")
        ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
        ax1.hist(d1["v_lins"], bins=40, alpha=0.6, color="#1f77b4", label=name1, density=True)
        ax1.hist(d2["v_lins"], bins=40, alpha=0.6, color="#2ca02c", label=name2, density=True)
        ax1.set_title("Vitesse Linéaire v_lin [m/s]", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Vitesse [m/s]")
        ax1.set_ylabel("Densité")
        ax1.legend(loc="upper right")

        # Vitesse angulaire
        ax2 = axes[1]
        ax2.set_facecolor("#fafbfc")
        ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
        ax2.hist(d1["v_angs"], bins=40, alpha=0.6, color="#1f77b4", label=name1, density=True)
        ax2.hist(d2["v_angs"], bins=40, alpha=0.6, color="#2ca02c", label=name2, density=True)
        ax2.set_title("Vitesse Angulaire v_ang [rad/s]", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Vitesse Angulaire [rad/s]")
        ax2.set_ylabel("Densité")
        ax2.legend(loc="upper right")

        dist_png = os.path.join(outdir, "dataset_comparison_distributions.png")
        plt.tight_layout()
        plt.savefig(dist_png, dpi=300)
        plt.close(fig)
        print(f"[Succès] Graphique de distributions généré : {dist_png}")

    print("=" * 76 + "\n")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparateur statistique des jeux de données LIMO.")
    parser.add_argument("--v1-dataset", type=str, default="data/curated/v1/handcrafted_dataset.csv", help="Chemin CSV V1")
    parser.add_argument("--v2-dataset", type=str, default="data/curated/v2/handcrafted_dataset.csv", help="Chemin CSV V2")
    parser.add_argument("--outdir", type=str, default="data/evaluation/comparison", help="Dossier de sortie")
    args = parser.parse_args()

    compare_datasets(v1_csv=args.v1_dataset, v2_csv=args.v2_dataset, outdir=args.outdir)
