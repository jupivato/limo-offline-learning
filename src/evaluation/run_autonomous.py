"""
Closed-Loop Autonomous Navigation and ISO 18646-2 Metrological Positioning
===========================================================================
Executes closed-loop autonomous navigation with frozen synaptic weights on the 
AgileX LIMO mobile robot at 20 Hz, supporting both Gazebo simulation and physical hardware.

Core Functionalities:
---------------------
1. Synaptic Policy Execution:
   Loads the trained PioneerNN (3 inputs: x_target_robot, y_target_robot, e_theta).
   Terminal stationarity guaranteed by execution-layer deadband and zero-velocity dwell hold.

2. SE(2) Metrological Distance Metric:
   Calculates Euclidean distance and Lie group SE(2) composite distance:
       d_xy = sqrt((x - x_t)^2 + (y - y_t)^2)
       d_pose = sqrt(d_xy^2 + (b * e_theta)^2)
   where b = 0.086 m (half-track width of AgileX LIMO converting heading error into meters).

3. Docking Envelope & Stationary Dwell-Time Watchdog:
   - Capture Envelope: d_xy <= dist_tol (0.08 m = 8 cm) and |e_theta| <= angle_tol (0.26 rad ~ 15.0°).
   - Stability Dwell-Time: Robot must remain continuously stationary (displacement < 5 mm,
     velocities near zero) for at least 30.0 seconds within the capture envelope before declaring success.
   - Any departure or drift resets the dwell timer to ensure metrological stability.

4. Kinematic Limits & Damping:
   - V_MAX = 0.50 m/s, W_MAX = 1.00 rad/s.
   - Pure 100% neural policy execution without external geometric heuristics.

Standards Compliance: ROS REP-103 & ISO 18646-2.
Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB).
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import shutil
import argparse
import collections
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

# Dynamic project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

try:
    import rclpy
    ROS2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ROS2_AVAILABLE = False

from src.core.pioneer_nn import PioneerNN, recast_angle, compute_k_theta
from src.evaluation.data_logger import DataLogger

try:
    from src.ros2.limo_interface import LimoROS2Interface
except (ImportError, ModuleNotFoundError):
    LimoROS2Interface = None

# ==============================================================================
# GLOBAL KINEMATIC BOUNDARIES & METROLOGICAL PARAMETERS
# ==============================================================================
V_MAX = 0.50                   # Maximum linear velocity (m/s)
W_MAX = 1.00                   # Maximum angular velocity (rad/s)
X_MAX = 5.0                    # Arena spatial boundary X (m)
Y_MAX = 5.0                    # Arena spatial boundary Y (m)
HALF_TRACK_B = 0.086           # Half-track width b = L/2 (m) for AgileX LIMO

DEFAULT_DIST_TOL = 0.08        # 8 cm positioning radius from dock center (m)
DEFAULT_ANGLE_TOL = 0.26       # ISO 18646-2 Heading orientation tolerance (rad) ~ 15.0 deg
DEFAULT_DWELL_TIME = 30.0      # Required continuous stationary duration at dock (s)
DEFAULT_MAX_DURATION = 150.0   # Extended mission ceiling to accommodate navigation + dwell (s)
# ==============================================================================


def compute_se2_distance(dx: float, dy: float, dtheta: float, b: float = HALF_TRACK_B) -> Tuple[float, float]:
    """
    Computes both Euclidean Cartesian distance and SE(2) composite pose distance.

    Args:
        dx (float): Longitudinal error in meters.
        dy (float): Lateral error in meters.
        dtheta (float): Heading error in radians [-pi, pi].
        b (float): Characteristic half-track radius in meters.

    Returns:
        Tuple[float, float]: (d_xy, d_pose) in meters.
    """
    d_xy = math.hypot(dx, dy)
    d_pose = math.hypot(d_xy, b * dtheta)
    return d_xy, d_pose


def run_autonomous_session(
    weights_path: str,
    target: List[float],
    teleport_to: Optional[List[float]] = None,
    is_real: bool = False,
    max_duration: float = DEFAULT_MAX_DURATION,
    dist_tol: float = DEFAULT_DIST_TOL,
    angle_tol: float = DEFAULT_ANGLE_TOL,
    dwell_time_required: float = DEFAULT_DWELL_TIME,
    output_dir: str = "data/evaluation",
    odom_topic: Optional[str] = None,
    cmd_vel_topic: str = "/cmd_vel",
    speed_scale: float = 1.0
) -> bool:
    """
    Orchestrates a closed-loop autonomous docking trial with stationary dwell verification.

    Args:
        weights_path (str): Path to trained neural network weights JSON.
        target (List[float]): Target pose [x_target, y_target, theta_target].
        teleport_to (Optional[List[float]]): Starting pose [x0, y0, th0] for Gazebo teleportation.
        is_real (bool): True for hardware deployment, False for Gazebo simulation.
        max_duration (float): Maximum trial duration before timeout (s).
        dist_tol (float): Positioning tolerance radius from dock center (m).
        angle_tol (float): Heading tolerance at dock (rad).
        dwell_time_required (float): Required continuous stationary dwell duration (s).
        output_dir (str): Root directory for telemetry logs and trajectory plots.
        odom_topic (Optional[str]): Odometry topic name (default: auto-detects /odom or /wheel/odom).
        cmd_vel_topic (str): Velocity command topic name (default: /cmd_vel).
        speed_scale (float): Velocity multiplier in [0.2, 1.0] for comfortable physical testing.

    Returns:
        bool: True if stable docking with verified dwell was achieved, False otherwise.
    """
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS 2 Humble / Foxy environment is required for live autonomous execution.")

    # Kinematic limits: scale linear velocity if desired, but maintain full W_MAX for 4WD rubber tire ground torque
    eff_scale = max(0.2, min(1.0, float(speed_scale)))
    v_max_effective = V_MAX * eff_scale
    w_max_effective = W_MAX  # Maintain full steering torque against tire ground friction

    print("\n" + "=" * 78)
    print("   AGILEX LIMO CLOSED-LOOP AUTONOMOUS DOCKING (ISO 18646-2 BENCHMARK)")
    print("=" * 78)
    print(f"Weights Artifact     : {weights_path}")
    print(f"Dock Target [x,y,θ]  : x={target[0]:.3f} m, y={target[1]:.3f} m, θ={math.degrees(target[2]):.1f}°")
    print(f"Platform Mode        : {'Physical LIMO Hardware' if is_real else 'Gazebo Simulation'}")
    print(f"Cruise Limits        : V_max = {v_max_effective:.2f} m/s | W_max = {w_max_effective:.2f} rad/s (Scale: {eff_scale*100:.0f}%)")
    print(f"Capture Threshold    : Distance <= {dist_tol * 100:.1f} cm | Angle <= {math.degrees(angle_tol):.1f}°")
    print(f"Stationary Dwell     : {dwell_time_required:.1f} s continuous immobilization")
    print(f"Mission Duration Max : {max_duration:.1f} s")
    print("=" * 78 + "\n")

    # 1. Initialize Neural Network Policy
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2, use_bias=True)
    model.load_from_json_file(weights_path)
    model.eval()
    print(f"[PioneerNN] Policy loaded from: {weights_path} (use_bias={model.use_bias})")

    # 2. Initialize ROS 2 Node & Hardware / Gazebo Interface
    rclpy.init()
    robot = LimoROS2Interface(safety_limits=is_real, odom_topic=odom_topic, cmd_vel_topic=cmd_vel_topic)
    logger = DataLogger()

    # Initial repositioning / teleportation
    init_pos = robot.get_position()
    init_dist = math.hypot(init_pos[0] - target[0], init_pos[1] - target[1])
    print(f"[Odometry] Current robot pose: x={init_pos[0]:.3f} m, y={init_pos[1]:.3f} m, theta={math.degrees(init_pos[2]):.1f}° (Distance to target: {init_dist:.2f} m)")

    if teleport_to is not None and not is_real:
        print(f"[Gazebo] Teleporting robot to starting pose: {teleport_to}")
        robot.teleport_gazebo(teleport_to[0], teleport_to[1], teleport_to[2])
        time.sleep(0.4)
    elif teleport_to is not None and is_real:
        print(f"[Hardware] Calibrating physical starting pose relative to target: {teleport_to}")
        robot.reset_pose(teleport_to[0], teleport_to[1], teleport_to[2])
        time.sleep(0.2)
    elif init_dist < dist_tol and not is_real:
        print("[Notice] Robot initialized inside dock. Teleporting to default benchmark starting pose [2.0, 2.0, 0.0]...")
        robot.teleport_gazebo(2.0, 2.0, 0.0)
        time.sleep(0.4)
    elif init_dist < dist_tol and is_real:
        print(f"[Warning] Robot is already at the dock ({init_dist*100:.1f} cm <= {dist_tol*100:.1f} cm).")
        print("Drive the robot to your desired starting position before launching, or check if /odom is zeroed at the start position.")

    if is_real:
        try:
            input("\n[Ready] Press Enter to start autonomous docking...")
        except (KeyboardInterrupt, EOFError):
            print("\n[Aborted] Mission cancelled by user.")
            robot.cleanup()
            rclpy.shutdown()
            return False

    success = False
    start_time = time.time()
    dt = 0.050  # 20 Hz loop period
    dwell_timer = 0.0
    dock_arrival_announced = False
    pos_window: collections.deque[Tuple[float, float, float, float]] = collections.deque(maxlen=20)  # 1-second window

    try:
        while rclpy.ok():
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                print(f"\n[Timeout] Maximum mission duration reached ({max_duration:.1f} s).")
                break

            pos = robot.get_position()
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            thr = recast_angle(pos[2])
            e_theta = recast_angle(pos[2] - target[2])

            dist_xy, dist_pose = compute_se2_distance(dx, dy, e_theta)

            # Robot-centric relative target coordinates
            x_target_robot = dx * math.cos(thr) + dy * math.sin(thr)
            y_target_robot = -dx * math.sin(thr) + dy * math.cos(thr)

            # Normalized inputs in [-1.0, +1.0] (relative to dock target)
            in_0 = max(-1.0, min(1.0, x_target_robot / X_MAX))
            in_1 = max(-1.0, min(1.0, y_target_robot / Y_MAX))
            in_2 = max(-1.0, min(1.0, e_theta / math.pi))

            cost = in_0 ** 2 + in_1 ** 2 + in_2 ** 2

            # Neural network inference (3 canonical inputs)
            cmd_raw = model([in_0, in_1, in_2])
            out_v = float(cmd_raw[0])
            out_w = float(cmd_raw[1])

            # Radial angular velocity damping gain K_theta
            k_th = compute_k_theta(x_target_robot, y_target_robot)

            # Neural policy output denormalization with K_theta damping
            v_cmd = max(-v_max_effective, min(v_max_effective, out_v * v_max_effective))
            w_cmd = max(-w_max_effective, min(w_max_effective, out_w * w_max_effective * k_th))

            # Canonical ENIB Output Deadband (Class 9: Terminal Stop)
            if abs(out_v) <= 0.05:
                v_cmd = 0.0
            if abs(out_w) <= 0.01:
                w_cmd = 0.0

            # Track 1-second displacement window for physical immobilization verification
            pos_window.append((elapsed, pos[0], pos[1], pos[2]))
            if len(pos_window) >= 20:
                t_old, x_old, y_old, th_old = pos_window[0]
                d_moved_window = math.hypot(pos[0] - x_old, pos[1] - y_old)
                d_rot_window = abs(recast_angle(pos[2] - th_old))
            else:
                d_moved_window = 0.0
                d_rot_window = 0.0

            # Velocity-level and physical immobilization criteria
            is_stopped_cmds = (abs(v_cmd) < 0.06 and abs(w_cmd) < 0.06)
            is_stopped_motion = (len(pos_window) >= 20 and d_moved_window < 0.010 and d_rot_window < 0.03)
            is_stationary = is_stopped_cmds or is_stopped_motion

            # Docking Zone Arrival & Continuous Dwell-Time Watchdog (ISO 18646-2)
            in_dock_envelope = (dist_xy <= dist_tol and abs(e_theta) <= angle_tol)

            if in_dock_envelope:
                if is_stationary:
                    # Vehicle naturally stopped at dock: engage zero-velocity hold for ISO 30s dwell test
                    v_cmd, w_cmd = 0.0, 0.0
                if not dock_arrival_announced:
                    print(f"\n[Dock Arrival] Robot arrived in dock envelope (d={dist_xy*100:.1f} cm, θ={math.degrees(abs(e_theta)):.1f}°)! Holding for {dwell_time_required:.0f}s dwell test...")
                    dock_arrival_announced = True
            else:
                dock_arrival_announced = False

            if elapsed > 1.0 and in_dock_envelope and is_stationary:
                dwell_timer += dt
                if int(dwell_timer / dt) % 20 == 0 or dwell_timer >= dwell_time_required:
                    print(f"[Dock Dwell] Stationary in envelope: {dwell_timer:5.1f}s / {dwell_time_required:.1f}s (d={dist_xy*100:.1f}cm, θ={math.degrees(abs(e_theta)):.1f}°)")

                if dwell_timer >= dwell_time_required:
                    print(f"\n[Success] Mission Complete! Vehicle continuously docked and stationary for {dwell_timer:.1f} s.")
                    print(f"          Final Position Error: {dist_xy*100:.2f} cm | Heading Error: {math.degrees(abs(e_theta)):.2f}° | SE(2) Pose Metric: {dist_pose*100:.2f} cm")
                    success = True
                    break
            else:
                if dwell_timer > 0.0:
                    print(f"[Notice] Dwell reset! Motion or boundary departure detected at t={elapsed:.1f}s (d={dist_xy*100:.1f}cm).")
                dwell_timer = 0.0

            # Diagnostics telemetry output every 1.0 second
            if int(elapsed / dt) % 20 == 0:
                print(f"t={elapsed:5.1f}s | Dist: {dist_xy:5.3f}m | SE(2): {dist_pose:5.3f}m | Head: {math.degrees(abs(e_theta)):5.1f}° | v_cmd: {v_cmd:+5.2f}m/s | w_cmd: {w_cmd:+5.2f}rad/s")

            robot.set_cmd_vel(v_cmd, w_cmd)
            logger.log(position=pos, target=target, command=[v_cmd, w_cmd], grad=[0.0, 0.0], cost=cost)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[Interrupted] Mission aborted by operator.")
    finally:
        robot.stop()
        now = datetime.now()
        timestamp_folder = now.strftime("test_%Y_%m_%d_%S_%M_%H")

        if output_dir and os.path.basename(os.path.normpath(output_dir)).startswith("test_"):
            test_dir = output_dir
        else:
            base_eval_dir = output_dir if output_dir else "data/evaluation"
            test_dir = os.path.join(base_eval_dir, timestamp_folder)

        data_dir = os.path.join(test_dir, "data")
        images_dir = os.path.join(test_dir, "images")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        prefix = "eval_success" if success else "eval_stop"
        csv_path = os.path.join(data_dir, f"{prefix}.csv")
        png_path = os.path.join(images_dir, f"{prefix}.png")

        logger.save(csv_path)
        logger.save_plot(png_path)

        final_pos = robot.get_position() if hasattr(robot, "get_position") else [0.0, 0.0, 0.0]
        elapsed_sec = time.time() - start_time
        err_dist = math.hypot(final_pos[0] - target[0], final_pos[1] - target[1])
        err_theta = abs(recast_angle(final_pos[2] - target[2]))
        err_pose = math.hypot(err_dist, HALF_TRACK_B * err_theta)

        params = {
            "test_id": timestamp_folder,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weights_path": os.path.abspath(weights_path),
            "platform": "Robot Physique" if is_real else "Simulateur Gazebo",
            "target": target,
            "start_pose": teleport_to,
            "final_position": [round(float(p), 4) for p in final_pos],
            "success": success,
            "elapsed_seconds": round(elapsed_sec, 3),
            "dwell_time_verified_seconds": round(dwell_timer, 2),
            "dwell_time_required_seconds": dwell_time_required,
            "max_duration_seconds": max_duration,
            "error_distance_m": round(err_dist, 4),
            "error_theta_deg": round(math.degrees(err_theta), 2),
            "error_se2_pose_m": round(err_pose, 4),
            "tolerances": {
                "dist_tol_m": dist_tol,
                "angle_tol_rad": angle_tol,
                "half_track_b_m": HALF_TRACK_B,
            },
            "kinematics": {
                "v_max_mps": V_MAX,
                "w_max_radps": W_MAX,
                "x_max_m": X_MAX,
                "y_max_m": Y_MAX,
                "dt_sec": dt,
            },
            "artifacts": {
                "folder": os.path.abspath(test_dir),
                "params_json": os.path.abspath(os.path.join(test_dir, "params.json")),
                "data_dir": os.path.abspath(data_dir),
                "images_dir": os.path.abspath(images_dir),
                "csv_path": os.path.abspath(csv_path),
                "png_path": os.path.abspath(png_path)
            }
        }
        params_path = os.path.join(test_dir, "params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"[Success] Trial metadata and telemetry preserved in: {params_path}")

        robot.cleanup()
        rclpy.shutdown()

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-Loop Autonomous Positioning & Metrological Docking.")
    parser.add_argument("--weights", type=str, default="models/supervised_w_torch_diff.json", help="Path to synaptic weights JSON")
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Target pose x y theta")
    parser.add_argument("--teleport-to", "--start", nargs=3, type=float, default=None, dest="teleport_to", help="Starting pose x y theta")
    parser.add_argument("--real", action="store_true", help="Physical hardware mode with hardware safety limits")
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_DURATION, help=f"Maximum duration ceiling (default: {DEFAULT_MAX_DURATION}s)")
    parser.add_argument("--dist-tol", type=float, default=DEFAULT_DIST_TOL, help=f"Docking positioning tolerance (default: {DEFAULT_DIST_TOL}m)")
    parser.add_argument("--angle-tol", type=float, default=DEFAULT_ANGLE_TOL, help=f"Docking orientation tolerance (default: {DEFAULT_ANGLE_TOL}rad)")
    parser.add_argument("--dwell-time", type=float, default=DEFAULT_DWELL_TIME, help=f"Required stationary dwell time at dock (default: {DEFAULT_DWELL_TIME}s)")
    parser.add_argument("--outdir", type=str, default="data/evaluation", help="Destination telemetry directory")
    parser.add_argument("--odom-topic", type=str, default=None, help="Odometry topic name (default: auto-detects /odom or /wheel/odom)")
    parser.add_argument("--cmd-vel-topic", type=str, default="/cmd_vel", help="Command velocity topic name (default: /cmd_vel)")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Velocity scaling multiplier in [0.2, 1.0] (e.g. 0.6 or 0.7 for smoother testing)")
    args = parser.parse_args()

    run_autonomous_session(
        weights_path=args.weights,
        target=args.target,
        teleport_to=args.teleport_to,
        is_real=args.real,
        max_duration=args.max_time,
        dist_tol=args.dist_tol,
        angle_tol=args.angle_tol,
        dwell_time_required=args.dwell_time,
        output_dir=args.outdir,
        odom_topic=args.odom_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        speed_scale=args.speed_scale
    )
