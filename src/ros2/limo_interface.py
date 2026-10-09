"""
Interface Matérielle et Simulation ROS 2 pour AgileX LIMO
=========================================================
Gère la communication asynchrone avec le robot réel ou le simulateur Gazebo
via les topics standard /odom et /cmd_vel.

Fonctionnalités :
- Souscription au topic d'odométrie (/odom) à 20-50 Hz.
- Conversion analytique quaternion -> lacet (yaw / theta).
- Publication des consignes de vitesse (Twist) sur /cmd_vel avec limitation de sécurité.
- Respect du chien de garde matériel (watchdog 500 ms) et arrêt sécurisé.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import math
import time
import threading
from typing import List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Remplacement pour environnement hors-ROS


class LimoROS2Interface(Node):
    """
    Interface ROS 2 isolée pour le robot mobile AgileX LIMO.
    
    Fournit un contrat strict de méthodes :
    - get_position() -> [x, y, theta]
    - set_cmd_vel(v_lin, v_ang)
    - stop()
    - cleanup()
    """

    def __init__(self, safety_limits: bool = True, odom_topic: str = "/odom", cmd_vel_topic: str = "/cmd_vel"):
        """
        Initialise l'interface ROS 2 du LIMO.

        Args:
            safety_limits (bool): Active le plafonnement strict des vitesses pour le robot réel.
            odom_topic (str): Nom du topic d'odométrie (défaut: /odom).
            cmd_vel_topic (str): Nom du topic de commande (défaut: /cmd_vel).
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("Le module rclpy n'est pas disponible dans cet environnement Python.")

        super().__init__("limo_offline_interface")
        self.safety_limits = safety_limits
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic

        # Pose interne du robot [x, y, theta] et décalage d'origine
        self._position = [0.0, 0.0, 0.0]
        self._raw_position = [0.0, 0.0, 0.0]
        self._offset = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()
        self._odom_received = False

        # Souscription et publication ROS 2
        self._odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 10
        )
        self._cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Thread d'écoute asynchrone pour le spin ROS 2
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._spin_thread.start()

        # Attente de la première trame d'odométrie
        print(f"[LimoROS2Interface] En attente du premier message sur {self.odom_topic}...")
        while not self._odom_received and rclpy.ok():
            time.sleep(0.05)
        print(f"[LimoROS2Interface] Première odométrie reçue avec succès.")

    def _odom_callback(self, msg: Odometry) -> None:
        """
        Callback de traitement des messages Odometry.
        Extrait la position cartésienne et convertit le quaternion en orientation lacet (yaw).
        """
        pos_x = float(msg.pose.pose.position.x)
        pos_y = float(msg.pose.pose.position.y)
        quat = msg.pose.pose.orientation

        # Formule de conversion Quaternion -> Angle de lacet (Yaw / theta)
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        with self._lock:
            self._raw_position = [pos_x, pos_y, theta]
            adj_theta = math.atan2(
                math.sin(theta + self._offset[2]),
                math.cos(theta + self._offset[2])
            )
            self._position = [
                pos_x + self._offset[0],
                pos_y + self._offset[1],
                adj_theta
            ]
            self._odom_received = True

    def reset_pose(self, x0: float, y0: float, theta0: float) -> None:
        """
        Calibre l'origine odométrique pour que le robot démarre exactement à (x0, y0, theta0).
        Permet le repositionnement virtuel ou après téléportation dans Gazebo.
        """
        with self._lock:
            raw_x, raw_y, raw_th = self._raw_position
            diff_th = math.atan2(math.sin(theta0 - raw_th), math.cos(theta0 - raw_th))
            self._offset = [x0 - raw_x, y0 - raw_y, diff_th]
            self._position = [x0, y0, theta0]
        print(f"[LimoROS2Interface] Pose réinitialisée à : x={x0:.2f} m, y={y0:.2f} m, theta={theta0:.2f} rad")

    def teleport_gazebo(self, x0: float, y0: float, theta0: float, model_name: str = "limo_description") -> bool:
        """
        Téléporte le modèle 3D du robot dans Gazebo et calibre l'odométrie sur (x0, y0, theta0).
        """
        import subprocess
        # 0. Arrêt préalable strict des moteurs
        self.stop()
        time.sleep(0.10)

        success = False
        # 1. Tentative via l'utilitaire CLI gz model
        try:
            res = subprocess.run(
                ["gz", "model", "-m", model_name, "-x", str(x0), "-y", str(y0), "-z", "0.05", "-Y", str(theta0)],
                capture_output=True, timeout=2.0
            )
            if res.returncode == 0:
                success = True
        except Exception:
            pass

        # 2. Tentative via le service ROS 2 /set_entity_state
        if not success:
            try:
                qz = math.sin(theta0 / 2.0)
                qw = math.cos(theta0 / 2.0)
                res = subprocess.run(
                    [
                        "ros2", "service", "call", "/set_entity_state", "gazebo_msgs/srv/SetEntityState",
                        f"{{state: {{name: '{model_name}', pose: {{position: {{x: {x0}, y: {y0}, z: 0.05}}, orientation: {{z: {qz}, w: {qw}}}}}}}}}"
                    ],
                    capture_output=True, timeout=3.0
                )
                if res.returncode == 0:
                    success = True
            except Exception:
                pass

        # 3. Stabilisation physique et calibration odométrique
        time.sleep(0.35)
        self.stop()
        self.reset_pose(x0, y0, theta0)
        return success

    def get_position(self) -> List[float]:
        """
        Renvoie la pose cartésienne et angulaire courante du robot [x, y, theta].

        Returns:
            List[float]: [x (m), y (m), theta (rad)]
        """
        with self._lock:
            return list(self._position)

    def set_cmd_vel(self, v_lin: float, v_ang: float) -> None:
        """
        Publie une consigne de vitesse linéaire et angulaire sur /cmd_vel.

        Args:
            v_lin (float): Vitesse d'avance longitudinale en m/s.
            v_ang (float): Vitesse de rotation en rad/s.
        """
        if self.safety_limits:
            # Plafond de sécurité strict pour préserver le matériel
            v_lin = max(-0.5, min(0.5, float(v_lin)))
            v_ang = max(-1.0, min(1.0, float(v_ang)))

        msg = Twist()
        msg.linear.x = float(v_lin)
        msg.angular.z = float(v_ang)
        self._cmd_pub.publish(msg)

    def stop(self) -> None:
        """Envoie immédiatement une consigne d'arrêt d'urgence (vitesses nulles)."""
        self.set_cmd_vel(0.0, 0.0)

    def cleanup(self) -> None:
        """Arrête le robot et détruit proprement le nœud ROS 2."""
        self.stop()
        time.sleep(0.05)
        self.destroy_node()
