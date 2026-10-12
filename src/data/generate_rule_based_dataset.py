"""
Générateur de Jeu de Données Point-à-Point Expert LIMO (Version 2 - Mappage Statique)
========================================================================================
Génère synthétiquement le jeu de données d'apprentissage hors ligne par échantillonnage
spatial régulier (grille d'états dans un rayon de 5 mètres) et lois de commande analytiques.

Caractéristiques de la Version 2 :
1. MAPPAGE STATIQUE ÉTAT -> ACTION (Sans intégration temporelle continue) :
   - Échantillonnage d'états (x_r, y_r, theta_r) couvrant l'espace de travail jusqu'à 5 mètres.
   - Commande calculée directement pour chaque état selon la loi de guidage stratégique.
2. NOUVEAU SCHÉMA SIMPLIFIÉ À 5 COLONNES :
   - Entrées : x_r, y_r, theta_r
   - Sorties : v_r, w_r
3. RÈGLES DE COMMANDE ET ORIENTATION STRATÉGIQUE :
   - Condition terminale à l'origine : (x_r=0, y_r=0, theta_r=0) -> v_r = 0.0, w_r = 0.0.
   - Vitesse linéaire décroissante proportionnelle à la distance (v_r proportional à d).
   - Orientation stratégique theta_s(x, y) = tanh(10.0 * x) * atan(1.0 * y) convergeant vers theta = 0.
   - Commande angulaire w_r proportionnelle au désalignement angulaire vers theta_s.

Projet : LIMO Offline Learning (CERV / ENIB) - Branch dataset-v2
"""

import os
import csv
import math
import argparse
from typing import List, Dict, Tuple, Set


# Facteurs d'échelle et paramètres cinématiques nominaux du LIMO
ALPHA = [1.0 / 3.0, 1.0 / 3.0, 1.0 / math.pi]
V_MAX_DEFAULT = 0.25      # Vitesse linéaire maximale (m/s)
W_MAX_DEFAULT = 0.50      # Vitesse angulaire maximale (rad/s)
K_W_DEFAULT = 1.0         # Gain proportionnel angulaire (s^-1)
R_WORKSPACE_DEFAULT = 5.0 # Rayon maximal de l'espace de travail (m)

COLONNES_V2 = ["x_r", "y_r", "theta_r", "v_r", "w_r"]


def recast_angle(angle_rad: float) -> float:
    """Normalise un angle dans l'intervalle principal (-pi, +pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def theta_strategic(x: float, y: float) -> float:
    """
    Orientation stratégique ENIB évitant les minima locaux et guidant vers l'origine :
    theta_s(x, y) = tanh(10.0 * x) * atan(1.0 * y)
    """
    return math.tanh(10.0 * x) * math.atan(1.0 * y)


def compute_control_action(
    x_r: float,
    y_r: float,
    theta_r: float,
    v_max: float = V_MAX_DEFAULT,
    w_max: float = W_MAX_DEFAULT,
    r_max: float = R_WORKSPACE_DEFAULT,
    k_w: float = K_W_DEFAULT,
    r_stop: float = 0.015,
    th_stop: float = 0.025
) -> Tuple[float, float]:
    """
    Calcule l'action de commande experte (v_r, w_r) pour un état donné (x_r, y_r, theta_r).

    Règles appliquées :
    1. Condition terminale : à l'origine (x_r=0, y_r=0, theta_r=0), arrêt complet (v_r=0, w_r=0).
    2. Vitesse linéaire v_r proportionnelle à la distance d = hypot(x_r, y_r), tendant vers 0.
    3. Orientation stratégique theta_s(x, y) qui guide l'alignement et converge vers 0 à l'origine.
    4. Commande angulaire w_r proportionnelle au désalignement angulaire vers theta_s.
    5. En cas de fort désalignement (ex: theta_r ≈ pi), rotation pure w_r pour réorienter le robot.

    Args:
        x_r (float): Position X du robot (m).
        y_r (float): Position Y du robot (m).
        theta_r (float): Orientation du robot (rad).
        v_max (float): Vitesse linéaire maximale autorisée (m/s).
        w_max (float): Vitesse angulaire maximale autorisée (rad/s).
        r_max (float): Rayon de référence de décélération (m).
        k_w (float): Gain proportionnel d'alignement angulaire.
        r_stop (float): Seuil de proximité spatiale d'arrêt (m).
        th_stop (float): Tolérance angulaire d'arrêt (rad).

    Returns:
        Tuple[float, float]: Consignes (v_r, w_r) désirées.
    """
    dist = math.hypot(x_r, y_r)
    theta_norm = recast_angle(theta_r)

    # 1. Condition terminale à l'origine [0, 0, 0]
    if dist < r_stop and abs(theta_norm) < th_stop:
        return 0.0, 0.0

    # 2. Orientation stratégique theta_s et écart angulaire
    th_s = theta_strategic(x_r, y_r)
    e_theta = recast_angle(th_s - theta_norm)

    # 3. Commande angulaire w_r proportionnelle au désalignement vers theta_s
    w_r = max(-w_max, min(w_max, k_w * e_theta))

    # 4. Commande linéaire v_r décroissante proportionnelle à la distance
    # Décélération douce continue jusqu'à zéro à l'origine
    v_base = v_max * min(1.0, dist / r_max)

    # Modulation par l'alignement :
    # Si aligné (e_theta ≈ 0) -> cos(e_theta) = 1.0 -> avance à v_base
    # Si désaligné (|e_theta| >= pi/2, ex: theta_r ≈ pi) -> cos(e_theta) <= 0 -> v_r = 0 (rotation pure)
    alignment = max(0.0, math.cos(e_theta))
    v_r = v_base * alignment

    # Si le robot est sur l'origine spatiale mais avec un cap non nul, pivot sur place
    if dist < r_stop:
        v_r = 0.0

    return round(v_r, 4), round(w_r, 4)


def expert_rule_based_controller(
    x: float,
    y: float,
    theta: float,
    tx: float = 0.0,
    ty: float = 0.0,
    ttheta: float = 0.0,
    **kwargs
) -> Tuple[float, float, int]:
    """
    Contrôleur expert de compatibilité directe vers compute_control_action.
    """
    v_r, w_r = compute_control_action(x - tx, y - ty, theta - ttheta)
    maneuver_id = 9 if (abs(v_r) < 1e-4 and abs(w_r) < 1e-4) else (1 if v_r > 0 else 7)
    return v_r, w_r, maneuver_id


def generate_state_grid(
    r_max: float = R_WORKSPACE_DEFAULT,
    coarse_step: float = 0.25,
    fine_step: float = 0.10,
    micro_step: float = 0.04,
    num_angles: int = 24
) -> List[Tuple[float, float, float]]:
    """
    Génère un échantillonnage spatial régulier multi-résolution d'états (x_r, y_r, theta_r)
    dans un rayon de travail allant jusqu'à r_max (défaut 5.0 m).

    Structure de la grille :
    - Grille large (coarse, 25 cm) couvrant tout le disque de rayon r_max.
    - Entonnoir d'approche fin (fine, 10 cm) dans le rayon de 1.2 m.
    - Zone de précision terminale (micro, 4 cm) dans le rayon de 0.3 m.
    - Poses exactes à l'origine et le long des axes principaux.
    - Discrétisation angulaire uniforme de num_angles angles dans [-pi, pi).

    Returns:
        List[Tuple[float, float, float]]: Liste des états échantillonnés (x_r, y_r, theta_r).
    """
    positions: Set[Tuple[float, float]] = set()

    # 1. Grille cartésienne générale dans le disque r <= r_max
    steps_c = int(math.ceil(r_max / coarse_step))
    for ix in range(-steps_c, steps_c + 1):
        for iy in range(-steps_c, steps_c + 1):
            x = round(ix * coarse_step, 3)
            y = round(iy * coarse_step, 3)
            if math.hypot(x, y) <= r_max:
                positions.add((x, y))

    # 2. Entonnoir de raffinement haute densité autour du quai (r <= 1.2 m)
    steps_f = int(math.ceil(1.2 / fine_step))
    for ix in range(-steps_f, steps_f + 1):
        for iy in range(-steps_f, steps_f + 1):
            x = round(ix * fine_step, 3)
            y = round(iy * fine_step, 3)
            if math.hypot(x, y) <= 1.2:
                positions.add((x, y))

    # 3. Micro-grille de précision terminale (r <= 0.30 m)
    steps_m = int(math.ceil(0.30 / micro_step))
    for ix in range(-steps_m, steps_m + 1):
        for iy in range(-steps_m, steps_m + 1):
            x = round(ix * micro_step, 3)
            y = round(iy * micro_step, 3)
            if math.hypot(x, y) <= 0.30:
                positions.add((x, y))

    # Origine exacte garantie
    positions.add((0.0, 0.0))
    sorted_positions = sorted(list(positions))

    # Discrétisation angulaire régulière dans [-pi, +pi)
    angles: List[float] = [
        round(-math.pi + i * (2.0 * math.pi / num_angles), 4)
        for i in range(num_angles)
    ]
    # S'assurer que theta = 0.0 exact est présent
    if 0.0 not in angles:
        angles.append(0.0)
        angles.sort()

    states: List[Tuple[float, float, float]] = []
    for x, y in sorted_positions:
        for th in angles:
            states.append((x, y, th))

    # Renforcement terminal à l'origine exacte (ancre stable pour le stationnement)
    # 50 réplications de (0.0, 0.0, 0.0) pour ancrer v=0, w=0 dans le modèle
    for _ in range(50):
        states.append((0.0, 0.0, 0.0))

    return states


def generate_rule_based_dataset(
    output_path: str = "data/curated/handcrafted_dataset.csv",
    r_max: float = R_WORKSPACE_DEFAULT,
    v_max: float = V_MAX_DEFAULT,
    w_max: float = W_MAX_DEFAULT,
    num_angles: int = 24
) -> int:
    """
    Génère et sauvegarde le jeu de données simplifié à 5 colonnes (x_r, y_r, theta_r, v_r, w_r).

    Args:
        output_path (str): Chemin du fichier CSV de sortie.
        r_max (float): Rayon maximal de l'espace de travail en mètres.
        v_max (float): Vitesse linéaire maximale en m/s.
        w_max (float): Vitesse angulaire maximale en rad/s.
        num_angles (int): Nombre d'angles échantillonnés par position.

    Returns:
        int: Nombre total d'échantillons générés.
    """
    print("\n" + "=" * 75)
    print("--- GÉNÉRATION DU DATASET LIMO V2 (MAPPAGE STATIQUE ÉTAT -> ACTION) ---")
    print(f"Schéma CSV simplifié         : {COLONNES_V2}")
    print(f"Objectif terminal            : [X=0.00, Y=0.00, theta=0.00 rad]")
    print(f"Rayon d'échantillonnage      : {r_max:.1f} mètres")
    print(f"Vitesse linéaire max (v_max) : {v_max:.2f} m/s (décroissance proportionnelle)")
    print(f"Vitesse angulaire max (w_max): {w_max:.2f} rad/s")
    print(f"Angles par position          : {num_angles} (résolution: {360.0/num_angles:.1f}°)")
    print(f"Fichier de destination       : {output_path}")
    print("=" * 75)

    states = generate_state_grid(r_max=r_max, num_angles=num_angles)
    total_states = len(states)

    samples: List[Dict[str, float]] = []
    zero_velocity_count = 0
    pure_rotation_count = 0
    forward_turning_count = 0
    straight_advance_count = 0

    for x_r, y_r, theta_r in states:
        v_r, w_r = compute_control_action(
            x_r=x_r,
            y_r=y_r,
            theta_r=theta_r,
            v_max=v_max,
            w_max=w_max,
            r_max=r_max
        )

        if abs(v_r) < 1e-4 and abs(w_r) < 1e-4:
            zero_velocity_count += 1
        elif abs(v_r) < 1e-4 and abs(w_r) >= 1e-4:
            pure_rotation_count += 1
        elif abs(w_r) < 1e-4 and v_r > 1e-4:
            straight_advance_count += 1
        else:
            forward_turning_count += 1

        samples.append({
            "x_r": round(x_r, 4),
            "y_r": round(y_r, 4),
            "theta_r": round(theta_r, 4),
            "v_r": round(v_r, 4),
            "w_r": round(w_r, 4)
        })

    # Écriture du fichier CSV avec l'en-tête strict à 5 colonnes
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_V2)
        writer.writeheader()
        writer.writerows(samples)

    # Statistiques descriptives
    print("\n--- STATISTIQUES DES ACTIONS GÉNÉRÉES ---")
    print(f"Échantillons totaux          : {total_states}")
    print(f"Arrêt total (v=0, w=0)       : {zero_velocity_count:6d} ({zero_velocity_count/total_states*100:5.2f}%)")
    print(f"Rotation pure (v=0, w!=0)    : {pure_rotation_count:6d} ({pure_rotation_count/total_states*100:5.2f}%) [ex: désalignement theta ≈ pi]")
    print(f"Avance rectiligne (w=0, v>0) : {straight_advance_count:6d} ({straight_advance_count/total_states*100:5.2f}%) [ex: y=0, theta=0]")
    print(f"Avance avec virage (v>0, w!=0): {forward_turning_count:6d} ({forward_turning_count/total_states*100:5.2f}%)")

    v_vals = [s["v_r"] for s in samples]
    w_vals = [s["w_r"] for s in samples]
    print(f"Plage v_r                    : [{min(v_vals):.4f}, {max(v_vals):.4f}] m/s")
    print(f"Plage w_r                    : [{min(w_vals):.4f}, {max(w_vals):.4f}] rad/s")
    print("=" * 75)
    print(f"[Succès] Dataset V2 sauvegardé : {output_path}\n")

    return total_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Générateur de dataset statique état -> action pour LIMO (Version 2)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/curated/handcrafted_dataset.csv",
        help="Chemin du fichier CSV de sortie (défaut: data/curated/handcrafted_dataset.csv)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=R_WORKSPACE_DEFAULT,
        help="Rayon maximal de l'espace de travail en mètres (défaut: 5.0)"
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=24,
        help="Nombre d'angles par position spatiale (défaut: 24)"
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=V_MAX_DEFAULT,
        help="Vitesse linéaire maximale en m/s (défaut: 0.25)"
    )
    parser.add_argument(
        "--w-max",
        type=float,
        default=W_MAX_DEFAULT,
        help="Vitesse angulaire maximale en rad/s (défaut: 0.50)"
    )
    args = parser.parse_args()

    generate_rule_based_dataset(
        output_path=args.output,
        r_max=args.radius,
        v_max=args.v_max,
        w_max=args.w_max,
        num_angles=args.num_angles
    )
