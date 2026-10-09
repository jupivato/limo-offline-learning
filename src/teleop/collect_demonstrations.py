"""
Collecteur de Démonstrations par Téléopération (20 Hz)
======================================================
Enregistre les trajectoires de pilotage manuel pour constituer le jeu de données
d'apprentissage hors ligne (Learning from Demonstration / Behavioral Cloning).

Fonctionnalités :
- Écoute des commandes de téléopération (joystick ou clavier) sur /cmd_vel_teleop.
- Répercussion des commandes sur le robot physique ou simulé via LimoROS2Interface.
- Échantillonnage temporel strict à 20 Hz (T = 50 ms).
- Calcul en temps réel des erreurs cartésiennes, de l'orientation stratégique theta_s
  et des 3 entrées normalisées du réseau [in_0, in_1, in_2].
- Exportation automatique des données au format CSV canonique dans data/raw/.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import sys
import time
import math
import csv
import argparse
from typing import List, Dict, Any

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

from src.ros2.limo_interface import LimoROS2Interface

# Constantes cinématiques et de normalisation
LIMO_HALF_TRACK = 0.086  # Demi-voie : voie de 172 mm -> 86 mm
ALPHA = [1.0 / 3.0, 1.0 / 3.0, 1.0 / math.pi]  # Facteurs d'échelle [alpha_x, alpha_y, alpha_theta]


def recast_angle(angle_rad: float) -> float:
    """Recalcule un angle dans l'intervalle (-pi, +pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def theta_strategic(x: float, y: float) -> float:
    """
    Calcule l'orientation stratégique ENIB évitant les minima locaux.
    theta_s = tanh(10 * x) * atan(1 * y)
    """
    return math.tanh(10.0 * x) * math.atan(1.0 * y)


def determine_quadrant(x: float, y: float) -> int:
    """Identifie le quadrant géométrique cartésien (1 à 4)."""
    if x >= 0 and y >= 0:
        return 1
    elif x < 0 and y >= 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:
        return 4


class TeleopCommandSubscriber(Node):
    """Nœud ROS 2 pour intercepter les commandes issues de la manette de téléopération."""

    def __init__(self, topic_name: str = "/cmd_vel_teleop"):
        super().__init__("teleop_listener")
        self.cmd_vel = [0.0, 0.0]
        self._sub = self.create_subscription(Twist, topic_name, self._callback, 10)

    def _callback(self, msg: Twist) -> None:
        self.cmd_vel = [float(msg.linear.x), float(msg.angular.z)]


def record_demonstration(
    target: List[float],
    output_dir: str = "data/raw",
    teleop_topic: str = "/cmd_vel_teleop",
    is_real: bool = False,
    maneuver_id: int = 1
) -> str:
    """
    Exécute une session d'enregistrement de téléopération.

    Args:
        target (List[float]): Cible de consigne [x_c, y_c, theta_c].
        output_dir (str): Dossier d'enregistrement des CSV.
        teleop_topic (str): Topic de la manette/clavier.
        is_real (bool): Applique les sécurités pour robot physique.
        maneuver_id (int): Identifiant de la classe de manœuvre (1 à 9).

    Returns:
        str: Chemin du fichier CSV généré.
    """
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS 2 n'est pas disponible dans cet environnement.")

    rclpy.init()
    listener = TeleopCommandSubscriber(topic_name=teleop_topic)
    robot = LimoROS2Interface(safety_limits=is_real)

    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"demo_{stamp}.csv")

    rows: List[Dict[str, Any]] = []
    print("\n" + "=" * 60)
    print(f"--- ENREGISTREMENT DE LA DÉMONSTRATION ---")
    print(f"Cible assignée     : x={target[0]:.2f} m, y={target[1]:.2f} m, theta={target[2]:.2f} rad")
    print(f"Topic d'écoute     : {teleop_topic}")
    print(f"Fichier de sortie  : {csv_filename}")
    print("Pilotez le robot maintenant. Appuyez sur Ctrl+C dès que la cible est atteinte.")
    print("=" * 60 + "\n")

    start_time = time.time()
    initial_pos = robot.get_position()
    quadrant = determine_quadrant(initial_pos[0] - target[0], initial_pos[1] - target[1])

    try:
        while rclpy.ok():
            rclpy.spin_once(listener, timeout_sec=0.005)
            pos = robot.get_position()
            cmd = listener.cmd_vel

            # Transmission de la commande au robot physique ou à Gazebo
            robot.set_cmd_vel(cmd[0], cmd[1])

            # Calcul des écarts par rapport à la cible
            e_x = pos[0] - target[0]
            e_y = pos[1] - target[1]
            e_theta = recast_angle(pos[2] - target[2])

            # Orientation stratégique et entrées normalisées du réseau
            th_s = theta_strategic(pos[0], pos[1])
            in_0 = e_x * ALPHA[0]
            in_1 = e_y * ALPHA[1]
            in_2 = (e_theta - th_s) * ALPHA[2]

            # Vitesses individuelles des roues gauche et droite
            v_left = cmd[0] - cmd[1] * LIMO_HALF_TRACK
            v_right = cmd[0] + cmd[1] * LIMO_HALF_TRACK

            current_time = time.time() - start_time
            rows.append({
                "timestamp": round(current_time, 4),
                "x": round(pos[0], 4),
                "y": round(pos[1], 4),
                "theta": round(pos[2], 4),
                "tx": round(target[0], 4),
                "ty": round(target[1], 4),
                "ttheta": round(target[2], 4),
                "e_x": round(e_x, 4),
                "e_y": round(e_y, 4),
                "e_theta": round(e_theta, 4),
                "in_0": round(in_0, 4),
                "in_1": round(in_1, 4),
                "in_2": round(in_2, 4),
                "v_lin": round(cmd[0], 4),
                "v_ang": round(cmd[1], 4),
                "v_wheel_left": round(v_left, 4),
                "v_wheel_right": round(v_right, 4),
                "maneuver_id": maneuver_id,
                "quadrant": quadrant,
            })

            # Fréquence rigoureuse de 20 Hz (période Ts = 50 ms)
            time.sleep(0.050)

    except KeyboardInterrupt:
        print("\nArrêt manuel détecté par l'opérateur.")
    finally:
        robot.stop()
        if rows:
            with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[Succès] {len(rows)} échantillons enregistrés dans : {csv_filename}")
        else:
            print("[Avertissement] Aucun échantillon n'a été enregistré.")

        robot.cleanup()
        listener.destroy_node()
        rclpy.shutdown()

    return csv_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecteur de démonstrations pour LIMO.")
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Pose cible x y theta")
    parser.add_argument("--outdir", type=str, default="data/raw", help="Dossier de sortie CSV")
    parser.add_argument("--teleop-topic", type=str, default="/cmd_vel_teleop", help="Topic d'écoute téléopération")
    parser.add_argument("--real", action="store_true", help="Active le mode robot réel avec sécurités")
    parser.add_argument("--maneuver", type=int, default=1, help="Identifiant manœuvre (1 à 9)")
    args = parser.parse_args()

    record_demonstration(
        target=args.target,
        output_dir=args.outdir,
        teleop_topic=args.teleop_topic,
        is_real=args.real,
        maneuver_id=args.maneuver
    )
