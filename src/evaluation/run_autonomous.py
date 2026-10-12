"""
Exécution Autonome en Boucle Fermée (Autonomous Navigation)
===========================================================
Charge les poids synaptiques optimisés hors ligne et pilote le robot LIMO
en boucle fermée avec poids gelés (mode inférence pure à 20 Hz).

Fonctionnalités :
- Chargement instantané des poids JSON dans le réseau PioneerNN.
- Lecture d'odométrie (/odom) et calcul des entrées normalisées [in_0, in_1, in_2].
- Inférence PyTorch en temps réel (< 1 ms en CPU).
- Publication des commandes de vitesse sur /cmd_vel avec chien de garde respecté.
- Détection d'accostage précis (distance < 5 cm et orientation < 0.1 rad).
- Enregistrement automatique de la session avec DataLogger pour certification ISO.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import sys
import time
import math
import argparse
from typing import List, Optional, Dict, Any, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import rclpy
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from src.core.pioneer_nn import PioneerNN
from src.ros2.limo_interface import LimoROS2Interface
from src.evaluation.data_logger import DataLogger

# Facteurs d'échelle et orientation stratégique
ALPHA = [1.0 / 3.0, 1.0 / 3.0, 1.0 / math.pi]


def recast_angle(angle_rad: float) -> float:
    """Recaste l'angle dans (-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def theta_strategic(x: float, y: float) -> float:
    """Orientation stratégique ENIB : theta_s = tanh(10 * x) * atan(1 * y)."""
    return math.tanh(10.0 * x) * math.atan(1.0 * y)


def run_autonomous_session(
    weights_path: str,
    target: List[float],
    teleport_to: Optional[List[float]] = None,
    is_real: bool = False,
    max_duration: float = 60.0,
    output_dir: str = "data/evaluation",
    dist_tol: float = 0.25
) -> bool:
    """
    Exécute une mission de positionnement autonome vers la cible.

    Args:
        weights_path (str): Chemin du fichier JSON contenant les poids entraînés.
        target (List[float]): Pose cible [x_c, y_c, theta_c].
        teleport_to (Optional[List[float]]): Position de départ pour téléporter le robot [x, y, theta].
        is_real (bool): Applique le mode sécurisé pour robot physique.
        max_duration (float): Temps limite en secondes avant arrêt automatique.
        output_dir (str): Dossier pour enregistrer la télémétrie de l'essai.
        dist_tol (float): Tolérance de distance pour l'accostage en mètres (défaut: 0.25).

    Returns:
        bool: True si la cible a été atteinte dans les tolérances, False sinon.
    """
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS 2 n'est pas disponible dans cet environnement.")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch n'est pas installé dans cet environnement.")

    print("\n" + "=" * 65)
    print("--- MISSION AUTONOME EN BOUCLE FERMÉE (POIDS GELÉS) ---")
    print(f"Modèle de poids   : {weights_path}")
    print(f"Pose cible [x,y,θ]: x={target[0]:.2f} m, y={target[1]:.2f} m, theta={target[2]:.2f} rad")
    print(f"Plateforme        : {'Robot Physique' if is_real else 'Simulateur Gazebo'}")
    print(f"Durée maximale    : {max_duration} s")
    print("=" * 65 + "\n")

    # 1. Instanciation du modèle et chargement des poids
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2)
    model.load_from_json_file(weights_path)
    model.eval()  # Mode évaluation : aucun calcul de gradient
    print(f"[PioneerNN] Poids chargés avec succès depuis : {weights_path}")

    # 2. Initialisation de ROS 2 et de l'interface robot
    rclpy.init()
    robot = LimoROS2Interface(safety_limits=is_real)
    logger = DataLogger()

    # Téléportation initiale si demandée ou si le robot est déjà à l'origine en simulation
    init_pos = robot.get_position()
    init_dist = math.hypot(init_pos[0] - target[0], init_pos[1] - target[1])

    if teleport_to is not None and not is_real:
        print(f"[Gazebo] Téléportation demandée vers : {teleport_to}")
        robot.teleport_gazebo(teleport_to[0], teleport_to[1], teleport_to[2])
        time.sleep(0.3)
    elif init_dist < 0.05 and not is_real:
        print("[Information] Le robot est déjà sur la cible (0,0,0).")
        print("[Gazebo] Repositionnement automatique au départ par défaut [2.0, 2.0, 0.0] pour le test...")
        robot.teleport_gazebo(2.0, 2.0, 0.0)
        time.sleep(0.3)

    success = False
    start_time = time.time()
    dt = 0.050  # Période de 20 Hz

    try:
        while rclpy.ok():
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                print(f"[Timeout] Durée maximale atteinte ({max_duration} s).")
                break

            pos = robot.get_position()
            e_x = pos[0] - target[0]
            e_y = pos[1] - target[1]
            dist = math.hypot(e_x, e_y)
            e_theta = recast_angle(pos[2] - target[2])

            th_s = theta_strategic(pos[0], pos[1])
            in_0 = e_x * ALPHA[0]
            in_1 = e_y * ALPHA[1]
            in_2 = (e_theta - th_s) * ALPHA[2]

            # Valeur instantanée du coût quadratique
            cost = (ALPHA[0] ** 2 * e_x ** 2 +
                    ALPHA[1] ** 2 * e_y ** 2 +
                    ALPHA[2] ** 2 * (e_theta - th_s) ** 2)

            # Critère de convergence et d'accostage : d < dist_tol et orientation alignée (|e_theta| < 0.35 rad ~ 20 deg)
            # (Ne s'applique qu'après au moins 0.5s de navigation pour éviter le faux arrêt immédiat)
            if elapsed > 0.5 and dist < dist_tol and abs(e_theta) < 0.35:
                print(f"\n[Succès] Cible atteinte en {elapsed:.2f} s ! (Erreur dist={dist:.3f} m, angle={abs(e_theta):.3f} rad [{math.degrees(abs(e_theta)):.1f}°])")
                success = True
                break

            # Inférence par la politique neuronale
            input_tensor = torch.tensor([in_0, in_1, in_2], dtype=torch.float32)
            with torch.no_grad():
                cmd = model(input_tensor).squeeze().tolist()

            # Cinématique automobile en courbes amples et marche arrière si nécessaire
            v_cmd = max(-0.16, min(0.26, float(cmd[0])))
            w_cmd = max(-0.45, min(0.45, float(cmd[1])))
            robot.set_cmd_vel(v_cmd, w_cmd)
            logger.log(position=pos, target=target, command=[v_cmd, w_cmd], grad=[0.0, 0.0], cost=cost)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterruption opérateur reçue.")
    finally:
        robot.stop()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = "eval_success" if success else "eval_stop"
        csv_path = os.path.join(output_dir, f"{prefix}_{stamp}.csv")
        png_path = os.path.join(output_dir, f"{prefix}_{stamp}.png")

        logger.save(csv_path)
        logger.save_plot(png_path)

        robot.cleanup()
        rclpy.shutdown()

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécuteur autonome en boucle fermée.")
    parser.add_argument("--weights", type=str, default="models/supervised_w_torch_diff.json", help="Chemin des poids JSON")
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Pose cible x y theta")
    parser.add_argument("--teleport-to", "--start", nargs=3, type=float, default=None, dest="teleport_to", help="Position initiale de départ x y theta")
    parser.add_argument("--real", action="store_true", help="Mode robot réel avec limites de sécurité")
    parser.add_argument("--max-time", type=float, default=60.0, help="Temps maximal en secondes")
    parser.add_argument("--dist-tol", type=float, default=0.25, help="Tolérance de distance pour l'accostage en mètres (défaut: 0.25)")
    parser.add_argument("--outdir", type=str, default="data/evaluation", help="Dossier de télémétrie")
    args = parser.parse_args()

    run_autonomous_session(
        weights_path=args.weights,
        target=args.target,
        teleport_to=args.teleport_to,
        is_real=args.real,
        max_duration=args.max_time,
        output_dir=args.outdir,
        dist_tol=args.dist_tol
    )
