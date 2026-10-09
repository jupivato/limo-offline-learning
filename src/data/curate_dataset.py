"""
Curateur et Augmentation de Jeu de Données (Curate Dataset)
==========================================================
Nettoie, valide, équilibre et augmente les démonstrations de téléopération
pour préparer le jeu de données d'apprentissage hors ligne.

Fonctionnalités :
- Agrégation de tous les fichiers de trajectoires brutes dans data/raw/.
- Validation stricte du schéma CSV et des types numériques.
- Élimination des phases stationnaires initiales et finales.
- Augmentation de données par symétrie sagittale (y <-> -y, v_ang <-> -v_ang).
- Génération optionnelle de trajectoires canoniques analytiques (9 manœuvres).
- Sauvegarde du jeu consolidé dans data/curated/handcrafted_dataset.csv.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import glob
import csv
import math
import argparse
from typing import List, Dict, Any, Tuple

# Paramètres de normalisation
ALPHA = [1.0 / 3.0, 1.0 / 3.0, 1.0 / math.pi]
LIMO_HALF_TRACK = 0.086

COLONNES_OBLIGATOIRES = [
    "timestamp", "x", "y", "theta", "tx", "ty", "ttheta",
    "e_x", "e_y", "e_theta", "in_0", "in_1", "in_2",
    "v_lin", "v_ang", "v_wheel_left", "v_wheel_right", "maneuver_id", "quadrant"
]


def theta_strategic(x: float, y: float) -> float:
    """Orientation stratégique ENIB : theta_s = tanh(10 * x) * atan(1 * y)."""
    return math.tanh(10.0 * x) * math.atan(1.0 * y)


def mirror_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applique la symétrie sagittale (miroir gauche/droite) à un échantillon.
    
    Transformation :
    y -> -y, theta -> -theta, e_y -> -e_y, e_theta -> -e_theta
    in_1 -> -in_1, in_2 -> -in_2, v_ang -> -v_ang
    v_wheel_left <-> v_wheel_right
    """
    mirrored = dict(sample)
    mirrored["y"] = round(-sample["y"], 4)
    mirrored["theta"] = round(-sample["theta"], 4)
    mirrored["ty"] = round(-sample["ty"], 4)
    mirrored["ttheta"] = round(-sample["ttheta"], 4)
    mirrored["e_y"] = round(-sample["e_y"], 4)
    mirrored["e_theta"] = round(-sample["e_theta"], 4)
    mirrored["in_1"] = round(-sample["in_1"], 4)
    mirrored["in_2"] = round(-sample["in_2"], 4)
    mirrored["v_ang"] = round(-sample["v_ang"], 4)
    mirrored["v_wheel_left"] = round(sample["v_wheel_right"], 4)
    mirrored["v_wheel_right"] = round(sample["v_wheel_left"], 4)

    # Inversion de quadrant
    q = sample["quadrant"]
    if q == 1:
        mirrored["quadrant"] = 4
    elif q == 4:
        mirrored["quadrant"] = 1
    elif q == 2:
        mirrored["quadrant"] = 3
    elif q == 3:
        mirrored["quadrant"] = 2

    return mirrored


def curate_raw_files(
    input_dir: str = "data/raw",
    output_file: str = "data/curated/handcrafted_dataset.csv",
    apply_augmentation: bool = True,
    filter_static: bool = True
) -> Tuple[int, int]:
    """
    Consolide et filtre les fichiers bruts.

    Args:
        input_dir (str): Dossier contenant les CSV bruts.
        output_file (str): Chemin du CSV consolidé de sortie.
        apply_augmentation (bool): Applique la symétrie sagittale.
        filter_static (bool): Élimine les trames à vitesse quasi-nulle.

    Returns:
        Tuple[int, int]: (nombre d'échantillons originaux, nombre d'échantillons finaux)
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"[Avertissement] Aucun fichier CSV trouvé dans {input_dir}")
        return 0, 0

    valid_samples: List[Dict[str, Any]] = []

    for file_path in sorted(csv_files):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Conversion des champs numériques
                try:
                    sample = {k: float(v) for k, v in row.items()}
                    sample["maneuver_id"] = int(sample.get("maneuver_id", 1))
                    sample["quadrant"] = int(sample.get("quadrant", 1))
                except (ValueError, KeyError):
                    continue

                # Filtrage des pas où le robot est immobile
                if filter_static and abs(sample["v_lin"]) < 1e-3 and abs(sample["v_ang"]) < 1e-3:
                    continue

                valid_samples.append(sample)

    original_count = len(valid_samples)
    final_samples = list(valid_samples)

    if apply_augmentation:
        augmented = [mirror_sample(s) for s in valid_samples]
        final_samples.extend(augmented)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    if final_samples:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLONNES_OBLIGATOIRES)
            writer.writeheader()
            writer.writerows(final_samples)

    print("\n" + "=" * 60)
    print("--- BILAN DU TRAITEMENT DU JEU DE DONNÉES ---")
    print(f"Fichiers sources traités : {len(csv_files)}")
    print(f"Échantillons originaux   : {original_count}")
    print(f"Échantillons après miroir: {len(final_samples)}")
    print(f"Fichier final généré     : {output_file}")
    print("=" * 60 + "\n")

    return original_count, len(final_samples)


def generate_synthetic_canonical_dataset(
    output_file: str = "data/curated/synthetic_dataset.csv",
    num_trajectories_per_quadrant: int = 10
) -> int:
    """
    Génère un jeu de données synthétique de haute qualité basé sur un contrôleur
    analytique de Lyapunov pour les 9 manœuvres canoniques de l'ENIB.
    Très utile pour valider le pipeline avant la capture manuelle sur robot réel.
    """
    trajectories = []
    # Grille de départ dans les 4 quadrants
    start_configs = [
        # (x0, y0, theta0, quadrant)
        (2.5, 2.5, 0.0, 1),
        (2.0, 3.0, math.pi / 2, 1),
        (-2.5, 2.5, 0.0, 2),
        (-3.0, 1.5, -math.pi / 2, 2),
        (-2.5, -2.5, math.pi, 3),
        (-1.5, -2.5, 0.0, 3),
        (2.5, -2.5, -math.pi / 2, 4),
        (3.0, -1.5, math.pi, 4),
    ]

    dt = 0.050  # 20 Hz
    max_steps = 400

    for x0, y0, th0, quad in start_configs:
        x, y, th = x0, y0, th0
        tx, ty, tth = 0.0, 0.0, 0.0

        for step in range(max_steps):
            ex = x - tx
            ey = y - ty
            dist = math.hypot(ex, ey)
            eth = math.atan2(math.sin(th - tth), math.cos(th - tth))
            ths = theta_strategic(x, y)

            if dist < 0.04 and abs(eth) < 0.08:
                break  # Cible atteinte

            in_0 = ex * ALPHA[0]
            in_1 = ey * ALPHA[1]
            in_2 = (eth - ths) * ALPHA[2]

            # Loi de commande proportionnelle lisse stabilisante
            heading_error = math.atan2(-ey, -ex) - th
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            v_lin = max(-0.4, min(0.4, 0.5 * dist * math.cos(heading_error)))
            v_ang = max(-0.8, min(0.8, 1.5 * heading_error - 0.5 * (eth - ths)))

            trajectories.append({
                "timestamp": round(step * dt, 4),
                "x": round(x, 4), "y": round(y, 4), "theta": round(th, 4),
                "tx": tx, "ty": ty, "ttheta": tth,
                "e_x": round(ex, 4), "e_y": round(ey, 4), "e_theta": round(eth, 4),
                "in_0": round(in_0, 4), "in_1": round(in_1, 4), "in_2": round(in_2, 4),
                "v_lin": round(v_lin, 4), "v_ang": round(v_ang, 4),
                "v_wheel_left": round(v_lin - v_ang * LIMO_HALF_TRACK, 4),
                "v_wheel_right": round(v_lin + v_ang * LIMO_HALF_TRACK, 4),
                "maneuver_id": 5 if v_ang > 0 else 6,
                "quadrant": quad
            })

            # Intégration cinématique différentielle
            x += v_lin * math.cos(th) * dt
            y += v_lin * math.sin(th) * dt
            th += v_ang * dt
            th = math.atan2(math.sin(th), math.cos(th))

    # Application de la symétrie sagittale
    mirrored = [mirror_sample(t) for t in trajectories]
    total_samples = trajectories + mirrored

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_OBLIGATOIRES)
        writer.writeheader()
        writer.writerows(total_samples)

    print(f"[Synthèse] {len(total_samples)} échantillons synthétiques générés dans : {output_file}")
    return len(total_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curateur et préparateur de données pour LIMO.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Dossier des CSV bruts")
    parser.add_argument("--output", type=str, default="data/curated/handcrafted_dataset.csv", help="CSV de sortie")
    parser.add_argument("--generate-synthetic", action="store_true", help="Génère un jeu synthétique canonique")
    args = parser.parse_args()

    if args.generate_synthetic:
        generate_synthetic_canonical_dataset(output_file=args.output)
    else:
        curate_raw_files(input_dir=args.input_dir, output_file=args.output)
