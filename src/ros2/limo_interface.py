"""
ROS 2 Hardware and Simulation Interface for AgileX LIMO
========================================================
Handles asynchronous communication with the physical robot or Gazebo simulator
via standard /odom and /cmd_vel topics.

Features:
- Subscription to odometry topic (/odom, /wheel/odom) at 20-50 Hz.
- Analytical quaternion -> yaw (theta) conversion.
- Velocity command publication (Twist) on /cmd_vel with strict safety bounds.
- Hardware watchdog compliance (500 ms) and emergency stop.

Project: LIMO Offline Learning (CERV / ENIB)
"""

from __future__ import annotations

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
except (ImportError, ModuleNotFoundError):
    ROS2_AVAILABLE = False
    Node = object  # Fallback for non-ROS environments
    Twist = object
    Odometry = object


class LimoROS2Interface(Node):
    """
    Isolated ROS 2 interface for the AgileX LIMO mobile robot.
    
    Provides a strict method contract:
    - get_position() -> [x, y, theta]
    - set_cmd_vel(v_lin, v_ang)
    - stop()
    - cleanup()
    """

    def __init__(self, safety_limits: bool = True, odom_topic: Optional[str] = None, cmd_vel_topic: str = "/cmd_vel"):
        """
        Initializes the LIMO ROS 2 interface.

        Args:
            safety_limits (bool): Enables strict velocity clamping for the physical robot.
            odom_topic (Optional[str]): Odometry topic name (None/'auto' listens to both /odom and /wheel/odom).
            cmd_vel_topic (str): Velocity command topic name (default: /cmd_vel).
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("The rclpy module is not available in this Python environment.")

        super().__init__("limo_offline_interface")
        self.safety_limits = safety_limits
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self._active_odom_topic: Optional[str] = None

        # Internal robot pose [x, y, theta] and origin offsets
        self._position = [0.0, 0.0, 0.0]
        self._raw_position = [0.0, 0.0, 0.0]
        self._raw_origin = [0.0, 0.0, 0.0]
        self._target_origin = [0.0, 0.0, 0.0]
        self._offset_th = 0.0
        self._offset = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()
        self._odom_received = False

        # Odometry subscriptions
        self._odom_subs = []
        if self.odom_topic and self.odom_topic != "auto":
            sub = self.create_subscription(
                Odometry, self.odom_topic, lambda msg, t=self.odom_topic: self._odom_callback(msg, t), 10
            )
            self._odom_subs.append(sub)
            print(f"[LimoROS2Interface] Subscribed to odometry topic: {self.odom_topic}")
        else:
            # Auto-discovery: listen simultaneously to /odom and /wheel/odom
            for candidate in ["/odom", "/wheel/odom"]:
                sub = self.create_subscription(
                    Odometry, candidate, lambda msg, t=candidate: self._odom_callback(msg, t), 10
                )
                self._odom_subs.append(sub)
            print("[LimoROS2Interface] Auto-detecting odometry stream (/odom and /wheel/odom)...")

        self._cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Asynchronous background thread for ROS 2 spinning
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._spin_thread.start()

        # Await first odometry frame
        wait_start = time.time()
        while not self._odom_received and rclpy.ok():
            time.sleep(0.05)
            if time.time() - wait_start > 5.0:
                topics_str = self.odom_topic if (self.odom_topic and self.odom_topic != "auto") else "/odom, /wheel/odom"
                print(f"[LimoROS2Interface] Waiting for odometry stream on {topics_str}...")
                wait_start = time.time()

        print(f"[LimoROS2Interface] First odometry frame successfully received on '{self.odom_topic}'.")

    def _odom_callback(self, msg: Odometry, topic_name: str = "/odom") -> None:
        """
        Odometry message processing callback.
        Extracts Cartesian position and converts quaternion to yaw angle (theta).
        Applies 2D SO(2) rotation matrix relative to initial calibration origin.
        """
        with self._lock:
            if self._active_odom_topic is None:
                self._active_odom_topic = topic_name
                self.odom_topic = topic_name
                print(f"[LimoROS2Interface] Odometry stream locked to topic: {topic_name}")
            elif self._active_odom_topic != topic_name:
                return  # Ignore messages from alternate topic once primary is locked

        pos_x = float(msg.pose.pose.position.x)
        pos_y = float(msg.pose.pose.position.y)
        quat = msg.pose.pose.orientation

        # Quaternion -> Yaw angle conversion formula
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        with self._lock:
            self._raw_position = [pos_x, pos_y, theta]
            dx_raw = pos_x - self._raw_origin[0]
            dy_raw = pos_y - self._raw_origin[1]
            diff_th = self._offset_th
            dx_world = dx_raw * math.cos(diff_th) - dy_raw * math.sin(diff_th)
            dy_world = dx_raw * math.sin(diff_th) + dy_raw * math.cos(diff_th)
            adj_theta = math.atan2(
                math.sin(theta + diff_th),
                math.cos(theta + diff_th)
            )
            self._position = [
                self._target_origin[0] + dx_world,
                self._target_origin[1] + dy_world,
                adj_theta
            ]
            self._odom_received = True

    def reset_pose(self, x0: float, y0: float, theta0: float) -> None:
        """
        Calibrates the odometry origin so the robot starts exactly at (x0, y0, theta0).
        Allows virtual repositioning or after Gazebo model teleportation.
        """
        with self._lock:
            raw_x, raw_y, raw_th = self._raw_position
            diff_th = math.atan2(math.sin(theta0 - raw_th), math.cos(theta0 - raw_th))
            self._raw_origin = [raw_x, raw_y, raw_th]
            self._target_origin = [x0, y0, theta0]
            self._offset_th = diff_th
            self._offset = [x0 - raw_x, y0 - raw_y, diff_th]
            self._position = [x0, y0, theta0]
        print(f"[LimoROS2Interface] Pose reset to: x={x0:.2f} m, y={y0:.2f} m, theta={theta0:.2f} rad")

    def teleport_gazebo(self, x0: float, y0: float, theta0: float, model_name: str = "limo_description") -> bool:
        """
        Teleports the 3D robot model in Gazebo and calibrates odometry to (x0, y0, theta0).
        """
        import subprocess
        # 0. Strict motor stop prior to teleportation
        self.stop()
        time.sleep(0.10)

        success = False
        # 1. Attempt via gz model CLI utility
        try:
            res = subprocess.run(
                ["gz", "model", "-m", model_name, "-x", str(x0), "-y", str(y0), "-z", "0.05", "-Y", str(theta0)],
                capture_output=True, timeout=2.0
            )
            if res.returncode == 0:
                success = True
        except Exception:
            pass

        # 2. Attempt via ROS 2 /set_entity_state service
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

        # 3. Physical settling and odometric calibration
        time.sleep(0.35)
        self.stop()
        self.reset_pose(x0, y0, theta0)
        return success

    def get_position(self) -> List[float]:
        """
        Returns the current Cartesian and angular pose of the robot [x, y, theta].

        Returns:
            List[float]: [x (m), y (m), theta (rad)]
        """
        with self._lock:
            return list(self._position)

    def set_cmd_vel(self, v_lin: float, v_ang: float) -> None:
        """
        Publishes linear and angular velocity commands on /cmd_vel.

        Args:
            v_lin (float): Forward linear velocity in m/s.
            v_ang (float): Rotational yaw rate in rad/s.
        """
        if self.safety_limits:
            # Strict safety bounds to protect mobile hardware
            v_lin = max(-0.5, min(0.5, float(v_lin)))
            v_ang = max(-1.0, min(1.0, float(v_ang)))

        msg = Twist()
        msg.linear.x = float(v_lin)
        msg.angular.z = float(v_ang)
        self._cmd_pub.publish(msg)

    def stop(self) -> None:
        """Immediately sends an emergency stop command (zero velocities)."""
        self.set_cmd_vel(0.0, 0.0)

    def cleanup(self) -> None:
        """Stops the robot and cleanly destroys the ROS 2 node."""
        self.stop()
        time.sleep(0.05)
        self.destroy_node()
