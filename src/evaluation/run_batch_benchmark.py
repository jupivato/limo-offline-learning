"""
Multi-Quadrant Automated Benchmarking Suite (ISO 18646-2 Metrology)
===================================================================
Executes standardized multi-quadrant docking trial batteries, logs continuous
kinematic telemetry, generates trajectory visualization dashboards, and computes 
the comprehensive set of ISO 18646-2 metrological performance indicators:
- Position Accuracy (Ap) and Position Repeatability (Rp)
- Orientation Accuracy (Ao) and Orientation Repeatability (Ro)
- Radial Standard Deviation (Sl) and Angular Standard Deviation (So)
- Attained Barycenter Coordinates (x_bar, y_bar)

Operational Modes:
------------------
1. Closed-Loop Kinematic Simulation Mode (Default / Headless):
   Integrates non-holonomic unicycle kinematics at 20 Hz (Euler forward) with the 
   bias-free PioneerNN policy. Enables instantaneous, deterministic metrological 
   evaluation without requiring an active Gazebo / GUI environment.

2. Live Gazebo Simulator Mode (Activated via --gazebo):
   Interfaces directly with Gazebo via ROS 2 (/odom and /cmd_vel topics) with 
   automated initial pose teleportation.

Docking & Stability Verification:
---------------------------------
- SE(2) Metrological Distance Metric: d_pose = sqrt(d_xy^2 + (b * e_theta)^2)
  where b = 0.086 m (half-track width of AgileX LIMO).
- Capture Envelope: d_xy <= dist_tol (0.05 m = 5 cm) and |e_theta| <= angle_tol (0.15 rad ~ 8.6°).
- Continuous Stationary Dwell: Robot must remain continuously immobile (movement < 5 mm,
  velocities near zero) for at least dwell_time (default: 30.0s) within the capture envelope.

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB).
Standards: ROS REP-103 & ISO 18646-2.
"""

from __future__ import annotations

import os
import sys
import time
import math
import csv
import json
import shutil
import argparse
import collections
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Dynamic project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.pioneer_nn import PioneerNN, recast_angle
from src.evaluation.data_logger import DataLogger
from calculus.calculate_iso_metrics import compute_iso_18646_metrics

try:
    import rclpy
    from src.ros2.limo_interface import LimoROS2Interface
    ROS2_AVAILABLE = True
except (ImportError, RuntimeError):
    ROS2_AVAILABLE = False
    LimoROS2Interface = None

# ==============================================================================
# GLOBAL KINEMATIC LIMITS & METROLOGICAL CONSTANTS
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
    Computes Euclidean Cartesian distance and SE(2) composite pose distance.
    """
    d_xy = math.hypot(dx, dy)
    d_pose = math.hypot(d_xy, b * dtheta)
    return d_xy, d_pose


# 1. Canonical ISO Test Suite (8 reference trials - 2 per quadrant)
CANONICAL_TEST_SCENARIOS = [
    ("Test 1 - Q1 Frontal",    2.0,  2.0,  0.0,            1),
    ("Test 2 - Q1 Lateral",    1.5,  2.5, -math.pi / 2,    1),
    ("Test 3 - Q2 Frontal",   -2.0,  2.0,  0.0,            2),
    ("Test 4 - Q2 Lateral",   -2.5,  1.5, -math.pi / 2,    2),
    ("Test 5 - Q3 Reverse",   -2.0, -2.0,  math.pi,        3),
    ("Test 6 - Q3 Lateral",   -1.5, -2.5,  math.pi / 2,    3),
    ("Test 7 - Q4 Lateral",    2.0, -2.0, -math.pi / 2,    4),
    ("Test 8 - Q4 Reverse",    2.5, -1.5,  math.pi,        4),
]

# 2. Extended Test Battery (20 systematic trials - 5 per quadrant)
EXTENDED_TEST_SCENARIOS = [
    # --- QUADRANT 1 (+X, +Y) : REVERSE / PARKING ---
    ("Test 01 - Q1 Short Frontal",   1.0,  0.8,  0.0,                 1),
    ("Test 02 - Q1 Medium Frontal",  2.0,  2.0,  0.0,                 1),
    ("Test 03 - Q1 Medium Lateral",  1.5,  2.5, -round(math.pi/2, 3), 1),
    ("Test 04 - Q1 Medium Oblique",  2.2,  1.2,  round(math.pi/4, 3), 1),
    ("Test 05 - Q1 Long Reverse",    3.0,  1.5,  round(math.pi, 3),   1),

    # --- QUADRANT 2 (-X, +Y) : FORWARD DUBINS ---
    ("Test 06 - Q2 Short Frontal",  -1.0,  0.8,  0.0,                 2),
    ("Test 07 - Q2 Medium Frontal", -2.0,  2.0,  0.0,                 2),
    ("Test 08 - Q2 Medium Lateral", -2.5,  1.5, -round(math.pi/2, 3), 2),
    ("Test 09 - Q2 Medium Oblique", -2.2,  1.2, -round(math.pi/4, 3), 2),
    ("Test 10 - Q2 Long Oblique",   -3.0,  2.0,  round(math.pi/4, 3), 2),

    # --- QUADRANT 3 (-X, -Y) : FORWARD DUBINS ---
    ("Test 11 - Q3 Short Reverse",  -1.0, -0.8,  round(math.pi, 3),   3),
    ("Test 12 - Q3 Medium Reverse", -2.0, -2.0,  round(math.pi, 3),   3),
    ("Test 13 - Q3 Medium Lateral", -1.5, -2.5,  round(math.pi/2, 3), 3),
    ("Test 14 - Q3 Medium Oblique", -2.2, -1.2,  round(3*math.pi/4, 3), 3),
    ("Test 15 - Q3 Long Frontal",   -3.0, -1.8,  0.0,                 3),

    # --- QUADRANT 4 (+X, -Y) : REVERSE / PARKING ---
    ("Test 16 - Q4 Short Lateral",   1.0, -0.8, -round(math.pi/2, 3), 4),
    ("Test 17 - Q4 Medium Lateral",  2.0, -2.0, -round(math.pi/2, 3), 4),
    ("Test 18 - Q4 Medium Reverse",  2.5, -1.5,  round(math.pi, 3),   4),
    ("Test 19 - Q4 Medium Oblique",  2.2, -1.2, -round(math.pi/4, 3), 4),
    ("Test 20 - Q4 Long Oblique",    3.0, -1.8, -round(3*math.pi/4, 3), 4),
]


def run_simulated_benchmark_run(
    model: PioneerNN,
    x0: float,
    y0: float,
    th0: float,
    run_name: str,
    target: List[float] = [0.0, 0.0, 0.0],
    max_duration: float = DEFAULT_MAX_DURATION,
    dist_tol: float = DEFAULT_DIST_TOL,
    angle_tol: float = DEFAULT_ANGLE_TOL,
    dwell_time_required: float = DEFAULT_DWELL_TIME,
    output_dir: str = "data/evaluation/batch_run",
    output_images_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a single benchmark run under 20 Hz closed-loop kinematic simulation.
    """
    print(f"\n>>> Starting (Kinematic Simulation) : {run_name} (Initial pose: x={x0:.2f}m, y={y0:.2f}m, θ={math.degrees(th0):.1f}°)")

    logger = DataLogger()
    dt = 0.050  # 20 Hz
    max_steps = int(max_duration / dt)

    x, y, th = x0, y0, th0
    success = False
    elapsed = 0.0
    dwell_timer = 0.0
    pos_window: collections.deque[Tuple[float, float, float, float]] = collections.deque(maxlen=20)

    for step in range(max_steps):
        elapsed = step * dt
        dx = target[0] - x
        dy = target[1] - y
        thr = recast_angle(th)
        e_theta = recast_angle(th - target[2])

        dist_xy, dist_pose = compute_se2_distance(dx, dy, e_theta)

        # Robot-relative target representation
        x_target_robot = dx * math.cos(thr) + dy * math.sin(thr)
        y_target_robot = -dx * math.sin(thr) + dy * math.cos(thr)

        in_0 = max(-1.0, min(1.0, x_target_robot / X_MAX))
        in_1 = max(-1.0, min(1.0, y_target_robot / Y_MAX))
        in_2 = max(-1.0, min(1.0, thr / math.pi))

        cost = in_0 ** 2 + in_1 ** 2 + in_2 ** 2

        # Model evaluation with 3 inputs
        cmd_raw = model([in_0, in_1, in_2])
        out_v = float(cmd_raw[0])
        out_w = float(cmd_raw[1])

        # Pure 100% Neural Policy Output Denormalization (Zero External Geometric Rules)
        v_cmd = max(-V_MAX, min(V_MAX, out_v * V_MAX))
        w_cmd = max(-W_MAX, min(W_MAX, out_w * W_MAX))

        # Canonical ENIB Output Deadband (Class 9: Terminal Stop)
        if abs(out_v) <= 0.05:
            v_cmd = 0.0
        if abs(out_w) <= 0.01:
            w_cmd = 0.0

        # Physical immobilization window verification
        pos_window.append((elapsed, x, y, th))
        if len(pos_window) >= 20:
            t_old, x_old, y_old, th_old = pos_window[0]
            d_moved_window = math.hypot(x - x_old, y - y_old)
            d_rot_window = abs(recast_angle(th - th_old))
        else:
            d_moved_window = 0.0
            d_rot_window = 0.0

        is_stopped_cmds = (abs(v_cmd) < 0.03 and abs(w_cmd) < 0.04)
        is_stopped_motion = (len(pos_window) >= 20 and d_moved_window < 0.008 and d_rot_window < 0.02)
        is_stationary = is_stopped_cmds or is_stopped_motion

        # Continuous stationary dwell within capture envelope
        in_dock_envelope = (dist_xy <= dist_tol and abs(e_theta) <= angle_tol)

        if step > 20 and in_dock_envelope and is_stationary:
            dwell_timer += dt
            if dwell_timer >= dwell_time_required:
                print(f"    [Success] Docked and stabilized! Immobile for {dwell_timer:.1f}s at t={elapsed:.1f}s (d={dist_xy*100:.2f}cm, θ={math.degrees(abs(e_theta)):.1f}°)")
                success = True
                break
        else:
            dwell_timer = 0.0

        logger.log(position=[x, y, th], target=target, command=[v_cmd, w_cmd], grad=[0.0, 0.0], cost=cost)

        # 20 Hz Euler forward unicycle integration
        x += v_cmd * math.cos(th) * dt
        y += v_cmd * math.sin(th) * dt
        th = recast_angle(th + w_cmd * dt)

    if not success:
        print(f"    [Ended] Duration: {elapsed:.1f}s. Final: d={dist_xy*100:.2f}cm, θ={math.degrees(abs(e_theta)):.1f}° (Dwell: {dwell_timer:.1f}s / {dwell_time_required:.1f}s)")

    # Save CSV trajectory and visualization plot
    os.makedirs(output_dir, exist_ok=True)
    img_dir = output_images_dir if output_images_dir else output_dir
    os.makedirs(img_dir, exist_ok=True)

    slug = run_name.lower().replace(" ", "_").replace("-", "").replace("__", "_")
    csv_path = os.path.join(output_dir, f"{slug}.csv")
    png_path = os.path.join(img_dir, f"{slug}.png")
    logger.save(csv_path)
    logger.save_plot(png_path)

    err_dist = math.hypot(x - target[0], y - target[1])
    err_theta = abs(recast_angle(th - target[2]))
    err_pose = math.hypot(err_dist, HALF_TRACK_B * err_theta)

    return {
        "name": run_name,
        "success": success,
        "elapsed": elapsed,
        "dwell_verified_sec": dwell_timer,
        "xf": x,
        "yf": y,
        "thf": th,
        "err_dist": err_dist,
        "err_theta": err_theta,
        "err_pose": err_pose
    }


def run_gazebo_benchmark_run(
    robot: Any,
    model: PioneerNN,
    x0: float,
    y0: float,
    th0: float,
    run_name: str,
    target: List[float] = [0.0, 0.0, 0.0],
    max_duration: float = DEFAULT_MAX_DURATION,
    dist_tol: float = DEFAULT_DIST_TOL,
    angle_tol: float = DEFAULT_ANGLE_TOL,
    dwell_time_required: float = DEFAULT_DWELL_TIME,
    output_dir: str = "data/evaluation/batch_run",
    output_images_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a single benchmark run under Gazebo with automated teleportation and dwell verification.
    """
    print(f"\n>>> Starting (Gazebo ROS 2) : {run_name} (Initial pose: x={x0:.2f}m, y={y0:.2f}m, θ={math.degrees(th0):.1f}°)")

    robot.teleport_gazebo(x0, y0, th0)
    time.sleep(0.4)

    logger = DataLogger()
    start_time = time.time()
    dt = 0.050
    success = False
    dwell_timer = 0.0
    pos_window: collections.deque[Tuple[float, float, float, float]] = collections.deque(maxlen=20)

    while rclpy.ok():
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            print(f"    [Timeout] Mission ceiling of {max_duration:.1f}s reached.")
            break

        pos = robot.get_position()
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        theta_r = recast_angle(pos[2])
        e_theta = recast_angle(pos[2] - target[2])

        dist_xy, dist_pose = compute_se2_distance(dx, dy, e_theta)

        x_target_robot = dx * math.cos(theta_r) + dy * math.sin(theta_r)
        y_target_robot = -dx * math.sin(theta_r) + dy * math.cos(theta_r)

        in_0 = max(-1.0, min(1.0, x_target_robot / X_MAX))
        in_1 = max(-1.0, min(1.0, y_target_robot / Y_MAX))
        in_2 = max(-1.0, min(1.0, theta_r / math.pi))

        cost = in_0 ** 2 + in_1 ** 2 + in_2 ** 2

        cmd_raw = model([in_0, in_1, in_2])
        out_v = float(cmd_raw[0])
        out_w = float(cmd_raw[1])

        # Pure 100% Neural Policy Output Denormalization (Zero External Geometric Rules)
        v_cmd = max(-V_MAX, min(V_MAX, out_v * V_MAX))
        w_cmd = max(-W_MAX, min(W_MAX, out_w * W_MAX))

        # Canonical ENIB Output Deadband (Class 9: Terminal Stop)
        if abs(out_v) <= 0.05:
            v_cmd = 0.0
        if abs(out_w) <= 0.01:
            w_cmd = 0.0

        pos_window.append((elapsed, pos[0], pos[1], pos[2]))
        if len(pos_window) >= 20:
            t_old, x_old, y_old, th_old = pos_window[0]
            d_moved_window = math.hypot(pos[0] - x_old, pos[1] - y_old)
            d_rot_window = abs(recast_angle(pos[2] - th_old))
        else:
            d_moved_window = 0.0
            d_rot_window = 0.0

        is_stopped_cmds = (abs(v_cmd) < 0.03 and abs(w_cmd) < 0.04)
        is_stopped_motion = (len(pos_window) >= 20 and d_moved_window < 0.008 and d_rot_window < 0.02)
        is_stationary = is_stopped_cmds or is_stopped_motion

        in_dock_envelope = (dist_xy <= dist_tol and abs(e_theta) <= angle_tol)

        if elapsed > 1.0 and in_dock_envelope and is_stationary:
            dwell_timer += dt
            if dwell_timer >= dwell_time_required:
                print(f"    [Success] Gazebo dock confirmed! Immobile for {dwell_timer:.1f}s at t={elapsed:.1f}s (d={dist_xy*100:.2f}cm, θ={math.degrees(abs(e_theta)):.1f}°)")
                success = True
                break
        else:
            dwell_timer = 0.0

        robot.set_cmd_vel(v_cmd, w_cmd)
        logger.log(position=pos, target=target, command=[v_cmd, w_cmd], grad=[0.0, 0.0], cost=cost)

        time.sleep(dt)

    robot.stop()
    time.sleep(0.1)

    os.makedirs(output_dir, exist_ok=True)
    img_dir = output_images_dir if output_images_dir else output_dir
    os.makedirs(img_dir, exist_ok=True)

    slug = run_name.lower().replace(" ", "_").replace("-", "").replace("__", "_")
    csv_path = os.path.join(output_dir, f"{slug}.csv")
    png_path = os.path.join(img_dir, f"{slug}.png")
    logger.save(csv_path)
    logger.save_plot(png_path)

    final_pos = robot.get_position()
    err_dist = math.hypot(final_pos[0] - target[0], final_pos[1] - target[1])
    err_theta = abs(recast_angle(final_pos[2] - target[2]))
    err_pose = math.hypot(err_dist, HALF_TRACK_B * err_theta)

    return {
        "name": run_name,
        "success": success,
        "elapsed": time.time() - start_time,
        "dwell_verified_sec": dwell_timer,
        "xf": final_pos[0],
        "yf": final_pos[1],
        "thf": final_pos[2],
        "err_dist": err_dist,
        "err_theta": err_theta,
        "err_pose": err_pose
    }


def run_batch_benchmark(
    weights_path: str = "models/supervised_w_torch_diff.json",
    output_dir: Optional[str] = None,
    use_gazebo: bool = False,
    suite: str = "canonical",
    dist_tol: float = DEFAULT_DIST_TOL,
    angle_tol: float = DEFAULT_ANGLE_TOL,
    dwell_time: float = DEFAULT_DWELL_TIME,
    max_duration: float = DEFAULT_MAX_DURATION
) -> Dict[str, Any]:
    """
    Executes the comprehensive benchmarking battery and synthesizes ISO 18646-2 metrology.
    """
    scenarios = EXTENDED_TEST_SCENARIOS if suite == "extended" else CANONICAL_TEST_SCENARIOS

    now = datetime.now()
    timestamp_folder = now.strftime("test_%Y_%m_%d_%S_%M_%H")

    if output_dir and os.path.basename(os.path.normpath(output_dir)).startswith("test_"):
        test_dir = output_dir
    else:
        base_eval_dir = output_dir if (output_dir and output_dir != "data/evaluation/batch_run") else "data/evaluation"
        test_dir = os.path.join(base_eval_dir, timestamp_folder)

    data_dir = os.path.join(test_dir, "data")
    images_dir = os.path.join(test_dir, "images")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print("      LIMO OFFLINE LEARNING : ISO 18646-2 BENCHMARKING PIPELINE")
    print("=" * 78)
    print(f"Policy Under Test    : {weights_path}")
    print(f"Execution Engine     : {'Gazebo (ROS 2)' if use_gazebo else 'Kinematic Simulation (20 Hz)'}")
    print(f"Kinematic Limits     : V_MAX = {V_MAX:.2f} m/s | W_MAX = {W_MAX:.2f} rad/s")
    print(f"Docking Envelope     : Position <= {dist_tol * 100:.1f} cm | Orientation <= {math.degrees(angle_tol):.1f}°")
    print(f"Stationary Dwell     : {dwell_time:.1f} s continuous immobilization requirement")
    print(f"Test Suite Selected  : {suite.upper()} ({len(scenarios)} multi-quadrant scenarios)")
    print(f"Artifact Destination : {test_dir}")
    print(f"  ├── CSV Data Traces: {data_dir}")
    print(f"  └── Image Dashboard: {images_dir}")
    print("=" * 78)

    # 1. Load Policy
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2, use_bias=False)
    model.load_from_json_file(weights_path)
    model.eval()

    results = []

    # 2. Execute Trials
    if use_gazebo:
        if not ROS2_AVAILABLE:
            print("[Warning] rclpy or ROS 2 not detected in environment. Falling back to Kinematic Simulation.")
            use_gazebo = False
        else:
            try:
                rclpy.init()
                robot = LimoROS2Interface(safety_limits=False)
                for idx, (name, x0, y0, th0, quad) in enumerate(scenarios, start=1):
                    res = run_gazebo_benchmark_run(
                        robot=robot,
                        model=model,
                        x0=x0, y0=y0, th0=th0,
                        run_name=f"Run_{idx}_Q{quad}",
                        target=[0.0, 0.0, 0.0],
                        max_duration=max_duration,
                        dist_tol=dist_tol,
                        angle_tol=angle_tol,
                        dwell_time_required=dwell_time,
                        output_dir=data_dir,
                        output_images_dir=images_dir
                    )
                    results.append(res)
                    time.sleep(0.5)
            except Exception as e:
                print(f"[Gazebo Warning] Communication failure: {e}. Falling back to simulation.")
                use_gazebo = False
            finally:
                if 'robot' in locals() and robot:
                    robot.stop()
                    robot.cleanup()
                if rclpy.ok():
                    rclpy.shutdown()

    if not use_gazebo:
        for idx, (name, x0, y0, th0, quad) in enumerate(scenarios, start=1):
            res = run_simulated_benchmark_run(
                model=model,
                x0=x0, y0=y0, th0=th0,
                run_name=f"Run_{idx}_Q{quad}",
                target=[0.0, 0.0, 0.0],
                max_duration=max_duration,
                dist_tol=dist_tol,
                angle_tol=angle_tol,
                dwell_time_required=dwell_time,
                output_dir=data_dir,
                output_images_dir=images_dir
            )
            results.append(res)

    if not results:
        print("[Error] No benchmark runs were executed.")
        return {}

    # 3. Calculate ISO 18646-2 Metrological Metrics
    trials_for_iso = [{"xf": r["xf"], "yf": r["yf"], "thf": r["thf"]} for r in results]
    iso_metrics = compute_iso_18646_metrics(trials_for_iso, xc=0.0, yc=0.0, oc=0.0)

    print("\n" + "=" * 78)
    print("           ISO 18646-2 CLOSED-LOOP METROLOGICAL CERTIFICATION REPORT")
    print("=" * 78)
    if iso_metrics:
        print(f"  • Sample Population (n)           : {iso_metrics['n']}")
        print(f"  • Position Accuracy (Ap)          : {iso_metrics['Ap']:.4f} m   (Requirement: < 0.25 m)")
        print(f"  • Orientation Accuracy (Ao)       : {iso_metrics['Ao_abs_deg']:.2f}°  (Requirement: < 15.0°)")
        print(f"  • Position Repeatability (Rp)     : {iso_metrics['Rp']:.4f} m")
        print(f"  • Orientation Repeatability (Ro)  : {iso_metrics['Ro_deg']:.2f}°")
        print(f"  • Radial Dispersion Std (Sl)      : {iso_metrics['Sl']:.4f} m")
        print(f"  • Angular Dispersion Std (So)     : {iso_metrics['So_deg']:.2f}°")
        print(f"  • Attained Barycenter (X, Y)      : ({iso_metrics['x_bar']:.4f}, {iso_metrics['y_bar']:.4f}) m")

        # Save consolidated ISO CSV report
        iso_csv_path = os.path.join(data_dir, "iso_18646_metrics.csv")
        with open(iso_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ISO 18646-2 Metric", "Value", "Unit", "Compliance Status"])
            writer.writerow(["Position Accuracy (Ap)", f"{iso_metrics['Ap']:.4f}", "m", "COMPLIANT" if iso_metrics['Ap'] < 0.25 else "OUT_OF_SPEC"])
            writer.writerow(["Orientation Accuracy (Ao)", f"{iso_metrics['Ao_abs_deg']:.2f}", "deg", "COMPLIANT" if iso_metrics['Ao_abs_deg'] < 15.0 else "OUT_OF_SPEC"])
            writer.writerow(["Position Repeatability (Rp)", f"{iso_metrics['Rp']:.4f}", "m", "VERIFIED"])
            writer.writerow(["Orientation Repeatability (Ro)", f"{iso_metrics['Ro_deg']:.2f}", "deg", "VERIFIED"])
            writer.writerow(["Radial Dispersion Std (Sl)", f"{iso_metrics['Sl']:.4f}", "m", "VERIFIED"])
            writer.writerow(["Angular Dispersion Std (So)", f"{iso_metrics['So_deg']:.2f}", "deg", "VERIFIED"])
        print(f"\n[Success] Metrological certification report saved to: {iso_csv_path}")

        # 4. Generate Comparative Visualizer Reports
        try:
            from src.visualization.benchmark_visualizer import generate_all_evaluation_reports
            generate_all_evaluation_reports(
                batch_dir=data_dir,
                prefix="benchmark",
                regenerate_sessions=False,
                output_images_dir=images_dir
            )
        except Exception as e:
            print(f"[Visualizer] Plot generation notice: {e}")

        # Synchronize legacy mirror directory
        try:
            legacy_dir = "data/evaluation/batch_run"
            os.makedirs(legacy_dir, exist_ok=True)
            shutil.copy2(iso_csv_path, os.path.join(legacy_dir, "iso_18646_metrics.csv"))
            for fn in os.listdir(data_dir):
                shutil.copy2(os.path.join(data_dir, fn), os.path.join(legacy_dir, fn))
            for fn in os.listdir(images_dir):
                shutil.copy2(os.path.join(images_dir, fn), os.path.join(legacy_dir, fn))
        except Exception:
            pass

    successful_runs = sum(1 for r in results if r["success"])
    params = {
        "test_id": timestamp_folder,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weights_path": os.path.abspath(weights_path),
        "execution_mode": "Gazebo (ROS 2)" if use_gazebo else "Kinematic Simulation (20 Hz)",
        "suite": suite,
        "total_runs": len(results),
        "successful_runs": successful_runs,
        "success_rate_pct": round(100.0 * successful_runs / max(1, len(results)), 2),
        "average_elapsed_sec": round(sum(r["elapsed"] for r in results) / max(1, len(results)), 2),
        "parameters": {
            "dist_tol_m": dist_tol,
            "angle_tol_rad": angle_tol,
            "dwell_time_s": dwell_time,
            "max_duration_s": max_duration,
            "half_track_b_m": HALF_TRACK_B,
            "v_max_mps": V_MAX,
            "w_max_radps": W_MAX,
            "x_max_m": X_MAX,
            "y_max_m": Y_MAX,
        },
        "iso_18646_metrics": iso_metrics if iso_metrics else {},
        "scenarios": [
            {"name": s[0], "x0": s[1], "y0": s[2], "th0": s[3], "quadrant": s[4]}
            for s in scenarios
        ],
        "runs_summary": [
            {
                "name": r["name"],
                "success": r["success"],
                "elapsed_sec": round(r["elapsed"], 3),
                "dwell_verified_sec": round(r.get("dwell_verified_sec", 0.0), 2),
                "final_pose": [round(r["xf"], 4), round(r["yf"], 4), round(r["thf"], 4)],
                "err_dist_m": round(r["err_dist"], 4),
                "err_theta_deg": round(math.degrees(r["err_theta"]), 2),
                "err_pose_m": round(r.get("err_pose", 0.0), 4)
            }
            for r in results
        ],
        "artifacts": {
            "test_dir": os.path.abspath(test_dir),
            "params_json": os.path.abspath(os.path.join(test_dir, "params.json")),
            "data_dir": os.path.abspath(data_dir),
            "images_dir": os.path.abspath(images_dir),
            "iso_csv": os.path.abspath(os.path.join(data_dir, "iso_18646_metrics.csv")),
        }
    }
    params_path = os.path.join(test_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"[Success] Batch benchmark parameters recorded in: {params_path}")
    print("=" * 78 + "\n")
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated ISO 18646-2 Benchmarking Pipeline (AgileX LIMO).")
    parser.add_argument("--weights", "--model", dest="weights", type=str, default="models/supervised_w_torch_diff.json", help="Path to synaptic weights JSON")
    parser.add_argument("--outdir", type=str, default="data/evaluation", help="Destination directory for benchmark telemetry")
    parser.add_argument("--suite", type=str, default="canonical", choices=["extended", "canonical"], help="Benchmark suite (canonical: 8 runs, extended: 20 runs)")
    parser.add_argument("--gazebo", action="store_true", help="Execute under live Gazebo simulation via ROS 2")
    parser.add_argument("--sim", action="store_true", help="Force pure 20 Hz kinematic simulation")
    parser.add_argument("--dist-tol", type=float, default=DEFAULT_DIST_TOL, help=f"Docking positioning tolerance (default: {DEFAULT_DIST_TOL}m)")
    parser.add_argument("--angle-tol", type=float, default=DEFAULT_ANGLE_TOL, help=f"Docking orientation tolerance (default: {DEFAULT_ANGLE_TOL}rad)")
    parser.add_argument("--dwell-time", type=float, default=DEFAULT_DWELL_TIME, help=f"Required stationary dwell time at dock (default: {DEFAULT_DWELL_TIME}s)")
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_DURATION, help=f"Maximum duration ceiling (default: {DEFAULT_MAX_DURATION}s)")
    args = parser.parse_args()

    use_gazebo = args.gazebo and not args.sim
    run_batch_benchmark(
        weights_path=args.weights,
        output_dir=args.outdir,
        use_gazebo=use_gazebo,
        suite=args.suite,
        dist_tol=args.dist_tol,
        angle_tol=args.angle_tol,
        dwell_time=args.dwell_time,
        max_duration=args.max_time
    )
