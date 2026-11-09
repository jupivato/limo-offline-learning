#!/usr/bin/env python3
"""
Real-Time Odometry & Heading Monitor for AgileX LIMO
====================================================
Subscribes to odometry (/wheel/odom or /odom) and displays
live Cartesian position (X, Y in meters) and heading (Theta in degrees)
with human-readable directional orientation.

Usage:
------
    python3 src/ros2/show_pose.py
    python3 src/ros2/show_pose.py --topic /wheel/odom
"""

from __future__ import annotations

import sys
import math
import argparse

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
except ImportError:
    print("[Error] ROS 2 (rclpy, nav_msgs) is not available. Please source your ROS 2 environment first:")
    print("  source /opt/ros/foxy/setup.bash (or humble)")
    sys.exit(1)


def get_compass_label(deg: float) -> str:
    """Returns a directional arrow and orientation description for the given heading angle."""
    deg_norm = (deg + 180.0) % 360.0 - 180.0  # normalize into [-180, +180]
    
    if -15.0 <= deg_norm <= 15.0:
        return "⬆️  Frente (0°)"
    elif 15.0 < deg_norm < 75.0:
        return "↖️  Frente-Esquerda"
    elif 75.0 <= deg_norm <= 105.0:
        return "⬅️  Esquerda (+90°)"
    elif 105.0 < deg_norm < 165.0:
        return "↙️  Trás-Esquerda"
    elif deg_norm >= 165.0 or deg_norm <= -165.0:
        return "⬇️  Para Trás (±180°)"
    elif -165.0 < deg_norm < -105.0:
        return "↘️  Trás-Direita"
    elif -105.0 <= deg_norm <= -75.0:
        return "➡️  Direita (-90°)"
    else:  # -75.0 < deg_norm < -15.0
        return "↗️  Frente-Direita"


class PoseMonitor(Node):
    def __init__(self, topic: str | None = None):
        super().__init__("limo_pose_monitor")
        self.active_topic = topic
        self.received = False

        if topic:
            self.create_subscription(Odometry, topic, lambda msg: self.cb(msg, topic), 10)
            print(f"[PoseMonitor] Escutando tópico: {topic}")
        else:
            print("[PoseMonitor] Buscando fluxo de odometria em /wheel/odom e /odom...")
            for candidate in ["/wheel/odom", "/odom"]:
                self.create_subscription(Odometry, candidate, lambda msg, t=candidate: self.cb(msg, t), 10)

    def cb(self, msg: Odometry, topic_name: str) -> None:
        if self.active_topic is None:
            self.active_topic = topic_name
            print(f"[PoseMonitor] Bloqueado com sucesso no tópico: {topic_name}\n")
            print("-" * 75)
            print(" POSIÇÃO DO ROBÔ EM TEMPO REAL (Atualização contínua)")
            print("-" * 75)
        elif self.active_topic != topic_name:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw_rad)

        compass = get_compass_label(yaw_deg)
        dist = math.hypot(x, y)

        # Print on a single refreshing line
        output = (
            f"\r📍 X: {x:+6.2f} m | Y: {y:+6.2f} m | Dist: {dist:5.2f} m | "
            f"Ângulo: {yaw_deg:+6.1f}° [{compass:20s}]   "
        )
        sys.stdout.write(output)
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Live LIMO Odometry & Heading Viewer")
    parser.add_argument("--topic", type=str, default=None, help="Odometry topic (/wheel/odom or /odom)")
    args = parser.parse_args()

    rclpy.init()
    node = PoseMonitor(topic=args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[PoseMonitor] Monitoramento finalizado pelo usuário.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
