"""
LIMO Dataset Generator (Robot-Centric Relative Coordinates & Discrete Ternary Actions)
========================================================================================
Generates the offline learning dataset for the AgileX LIMO mobile robot under the
canonical robot-relative reference frame:

1. Inputs:
   - x_target_robot : Target X coordinate in the robot's local body frame (m)
   - y_target_robot : Target Y coordinate in the robot's local body frame (m)
   - theta_robot    : Absolute robot heading in the global frame (rad) [-pi, +pi]

2. Outputs (Discrete Ternary Actions):
   - v_robot : Linear velocity command in {-V_MAX, 0.0, +V_MAX} (m/s)
   - w_robot : Angular velocity command in {-W_MAX, 0.0, +W_MAX} (rad/s)

3. Spatial Domain & Grid:
   - x_max = 5.0 m, y_max = 5.0 m
   - Regular grid: -x_max <= x_target <= x_max with step 0.5 m
   - Regular grid: -y_max <= y_target <= y_max with step 0.5 m
   - Absolute heading: -pi <= theta_robot <= pi
   - High-density terminal docking subgrid around (0, 0) for precise braking (d <= 0.20 m)
   - Closed-loop dynamic rollouts at 20 Hz (dt = 0.05 s)

Project: LIMO Offline Learning (CERV / ENIB)
"""

from __future__ import annotations

import os
import sys
import csv
import math
import argparse
from typing import List, Dict, Any, Tuple

# Project root path resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Kinematic constants
V_MAX = 0.24       # Maximum linear velocity (m/s)
W_MAX = 0.50       # Maximum angular velocity (rad/s)
X_MAX = 5.0        # Spatial boundary X (m)
Y_MAX = 5.0        # Spatial boundary Y (m)
D_BRAKE = 0.20     # Braking distance threshold (m)
D_STOP = 0.05      # Terminal dock deadband (m)
DT = 0.050         # 20 Hz control loop period (s)

COLUMNS = ["x_target_robot", "y_target_robot", "theta_robot", "v_robot", "w_robot"]


def recast_angle(angle_rad: float) -> float:
    """Normalize angle into the principal interval (-pi, +pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compute_discrete_action(
    x_tr: float,
    y_tr: float,
    theta_r: float,
    v_max: float = V_MAX,
    w_max: float = W_MAX,
    d_brake: float = D_BRAKE,
    d_stop: float = D_STOP
) -> Tuple[float, float]:
    """
    Computes the optimal discrete action in {-v_max, 0, v_max} and {-w_max, 0, w_max}
    given the robot-centric target coordinates (x_tr, y_tr) and absolute heading theta_r.
    """
    d = math.hypot(x_tr, y_tr)
    e_theta = recast_angle(theta_r)

    # 1. Terminal Docked Deadband (d <= d_stop, e.g. 5 cm)
    if d <= d_stop:
        v = 0.0
        if abs(e_theta) <= 0.06:
            w = 0.0
        elif e_theta > 0:
            w = -w_max  # Turn clockwise to bring theta_r -> 0
        else:
            w = w_max   # Turn counter-clockwise to bring theta_r -> 0
        return v, w

    # 2. Braking & Alignment Zone (d <= d_brake, e.g. 20 cm)
    # The linear speed drops to 0.0 so the network learns to brake smoothly!
    if d <= d_brake:
        v = 0.0
        if abs(e_theta) <= 0.08:
            w = 0.0
        elif e_theta > 0:
            w = -w_max
        else:
            w = w_max
        return v, w

    # 3. Reverse Docking Zone (Target is directly behind: x_tr < -d_brake and |y_tr| <= 1.5 m)
    if x_tr < -d_brake and abs(y_tr) <= 1.5:
        v = -v_max
        # Steering rear towards target:
        # If target is behind to the left (y_tr > 0.08), swing rear left by turning clockwise (w = -w_max)
        if y_tr > 0.08:
            w = -w_max
        elif y_tr < -0.08:
            w = w_max
        else:
            w = 0.0
        return v, w

    # 4. General Approach Zone
    # Bearing angle to target in the robot's local body frame
    phi = math.atan2(y_tr, x_tr)

    # If the target is far to the side (|phi| > 60 degrees = ~1.05 rad),
    # execute pure rotation on the spot to face the target
    if abs(phi) > 1.05:
        v = 0.0
        w = w_max if phi > 0 else -w_max
        return v, w

    # Target is in front (|phi| <= 60 degrees): Drive forward!
    v = v_max

    # Near the dock (d < 1.0 m), blend target bearing with dock alignment (theta_r -> 0)
    if d < 1.0:
        steer_err = phi + 0.5 * (-e_theta)
    else:
        steer_err = phi

    if steer_err > 0.08:
        w = w_max   # Turn left
    elif steer_err < -0.08:
        w = -w_max  # Turn right
    else:
        w = 0.0     # Drive straight

    return v, w


def generate_macro_grid(
    x_max: float = X_MAX,
    y_max: float = Y_MAX,
    step_xy: float = 0.5,
    num_angles: int = 25
) -> List[Tuple[float, float, float, float, float]]:
    """
    Generates systematic grid samples over [-x_max, x_max] and [-y_max, y_max] every step_xy.
    """
    samples: List[Tuple[float, float, float, float, float]] = []

    # Coordinates from -x_max to +x_max
    n_steps_x = int(round(2 * x_max / step_xy))
    xs = [round(-x_max + i * step_xy, 2) for i in range(n_steps_x + 1)]

    n_steps_y = int(round(2 * y_max / step_xy))
    ys = [round(-y_max + i * step_xy, 2) for i in range(n_steps_y + 1)]

    # Angles across [-pi, +pi]
    thetas = [round(-math.pi + i * (2 * math.pi / num_angles), 4) for i in range(num_angles + 1)]

    for x in xs:
        for y in ys:
            for th in thetas:
                v, w = compute_discrete_action(x, y, th)
                samples.append((x, y, th, v, w))

    return samples


def generate_dense_docking_subgrid() -> List[Tuple[float, float, float, float, float]]:
    """
    Generates high-density samples around the docking station (|x| <= 0.6 m, |y| <= 0.6 m)
    to provide high resolution for the braking threshold and zero-velocity equilibria.
    """
    samples: List[Tuple[float, float, float, float, float]] = []

    # Fine positions from -0.6 to +0.6 with step 0.05 m
    fine_steps = [round(v * 0.05, 3) for v in range(-12, 13)]
    # Fine angles around 0
    fine_thetas = [round(th * 0.05, 3) for th in range(-10, 11)] + [
        round(math.pi / 4, 3), round(-math.pi / 4, 3),
        round(math.pi / 2, 3), round(-math.pi / 2, 3),
        round(math.pi, 3), round(-math.pi, 3)
    ]

    for x in fine_steps:
        for y in fine_steps:
            for th in fine_thetas:
                v, w = compute_discrete_action(x, y, th)
                samples.append((x, y, th, v, w))

    return samples


def simulate_rollout(
    start_x_tr: float,
    start_y_tr: float,
    start_theta_r: float,
    max_steps: int = 200
) -> List[Tuple[float, float, float, float, float]]:
    """
    Simulates a closed-loop trajectory rollout at 20 Hz in the robot's relative frame.
    In global frame, target is at (0, 0, 0), robot starts with relative pose.
    """
    rollout_samples: List[Tuple[float, float, float, float, float]] = []

    # Convert initial robot-relative target to global robot pose (assuming target at origin 0,0,0)
    # x_tr = (0 - x_r)*cos(th) + (0 - y_r)*sin(th)
    # y_tr = -(0 - x_r)*sin(th) + (0 - y_r)*cos(th)
    # Inverting gives robot world pose:
    th = recast_angle(start_theta_r)
    cos_th = math.cos(th)
    sin_th = math.sin(th)
    rx = -(start_x_tr * cos_th - start_y_tr * sin_th)
    ry = -(start_x_tr * sin_th + start_y_tr * cos_th)

    for _ in range(max_steps):
        th_norm = recast_angle(th)
        cos_t = math.cos(th_norm)
        sin_t = math.sin(th_norm)

        # Compute relative target in robot frame
        dx = 0.0 - rx
        dy = 0.0 - ry
        x_tr = dx * cos_t + dy * sin_t
        y_tr = -dx * sin_t + dy * cos_t

        v, w = compute_discrete_action(x_tr, y_tr, th_norm)
        rollout_samples.append((round(x_tr, 4), round(y_tr, 4), round(th_norm, 4), v, w))

        # Check stopping condition
        dist = math.hypot(x_tr, y_tr)
        if dist <= D_STOP and abs(th_norm) <= 0.05:
            break

        # Forward Euler kinematic update of unicycle at 20 Hz
        rx += v * math.cos(th_norm) * DT
        ry += v * math.sin(th_norm) * DT
        th += w * DT

    return rollout_samples


def generate_rollouts() -> List[Tuple[float, float, float, float, float]]:
    """
    Simulates multiple rollouts starting from diverse points across the 5x5m arena.
    """
    samples: List[Tuple[float, float, float, float, float]] = []

    start_positions = [
        (-4.0, 0.0), (-3.0, 1.5), (-3.0, -1.5), (-2.0, 2.0), (-2.0, -2.0),
        (-1.5, 0.0), (-1.0, 1.0), (-1.0, -1.0), (-0.5, 0.5), (-0.5, -0.5),
        (1.0, 0.0), (2.0, 1.0), (2.0, -1.0), (3.0, 0.0), (1.5, 1.5), (1.5, -1.5),
        (0.0, 2.0), (0.0, -2.0), (0.0, 3.5), (0.0, -3.5), (4.0, 2.0), (-4.0, -2.0)
    ]
    start_headings = [
        0.0, round(math.pi / 4, 3), round(-math.pi / 4, 3),
        round(math.pi / 2, 3), round(-math.pi / 2, 3),
        round(3 * math.pi / 4, 3), round(-3 * math.pi / 4, 3), round(math.pi, 3)
    ]

    for sx, sy in start_positions:
        for sth in start_headings:
            traj = simulate_rollout(sx, sy, sth)
            samples.extend(traj)

    return samples


def apply_bilateral_symmetry(
    samples: List[Tuple[float, float, float, float, float]]
) -> List[Tuple[float, float, float, float, float]]:
    """
    Applies sagittal reflection symmetry to perfectly balance left/right turning:
    y_target -> -y_target, theta_robot -> -theta_robot, w_robot -> -w_robot.
    """
    augmented: List[Tuple[float, float, float, float, float]] = []
    seen = set()

    for x, y, th, v, w in samples:
        key = (round(x, 3), round(y, 3), round(th, 3))
        if key not in seen:
            seen.add(key)
            augmented.append((x, y, th, v, w))

        # Mirror sample
        mx = x
        my = -y
        mth = recast_angle(-th)
        mv = v
        mw = -w

        mkey = (round(mx, 3), round(my, 3), round(mth, 3))
        if mkey not in seen:
            seen.add(mkey)
            augmented.append((mx, my, mth, mv, mw))

    return augmented


def main():
    parser = argparse.ArgumentParser(description="Generate LIMO robot-centric relative dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.csv",
        help="Target output CSV file path (default: data/dataset.csv)."
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=X_MAX,
        help="Max X coordinate limit in meters (default: 5.0)."
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=Y_MAX,
        help="Max Y coordinate limit in meters (default: 5.0)."
    )
    parser.add_argument(
        "--step-xy",
        type=float,
        default=0.5,
        help="Spatial grid resolution step in meters (default: 0.5)."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("\n" + "=" * 65)
    print("--- GENERATING ROBOT-CENTRIC DISCRETE DATASET (V2) ---")
    print(f"Spatial Arena Limit  : [-{args.x_max}, +{args.x_max}] x [-{args.y_max}, +{args.y_max}] m")
    print(f"Grid Step Resolution : {args.step_xy} m")
    print(f"Linear Actions       : {{-{V_MAX}, 0.0, +{V_MAX}}} m/s")
    print(f"Angular Actions      : {{-{W_MAX}, 0.0, +{W_MAX}}} rad/s")
    print(f"Braking Threshold    : {D_BRAKE} m (v=0.0)")
    print(f"Output File          : {args.output}")
    print("=" * 65 + "\n")

    # 1. Macro Grid Generation
    print("1/4 Generating systematic macro-grid (0.5m interval)...")
    grid_samples = generate_macro_grid(x_max=args.x_max, y_max=args.y_max, step_xy=args.step_xy)
    print(f"    Macro-grid samples: {len(grid_samples):,}")

    # 2. Dense Docking Subgrid Generation
    print("2/4 Generating high-density terminal docking subgrid...")
    dock_samples = generate_dense_docking_subgrid()
    print(f"    Docking subgrid samples: {len(dock_samples):,}")

    # 3. Dynamic Rollouts
    print("3/4 Generating closed-loop dynamic trajectory rollouts (20 Hz)...")
    rollout_samples = generate_rollouts()
    print(f"    Rollout samples: {len(rollout_samples):,}")

    all_raw = grid_samples + dock_samples + rollout_samples

    # 4. Bilateral Symmetry Augmentation & Deduplication
    print("4/4 Applying bilateral sagittal symmetry and deduplication...")
    final_dataset = apply_bilateral_symmetry(all_raw)
    print(f"    Consolidated unique dataset size: {len(final_dataset):,} samples")

    # Save to CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in final_dataset:
            writer.writerow(row)

    # Compute Statistics
    v_counts = { -V_MAX: 0, 0.0: 0, V_MAX: 0 }
    w_counts = { -W_MAX: 0, 0.0: 0, W_MAX: 0 }
    for _, _, _, v, w in final_dataset:
        v_key = round(v, 2)
        w_key = round(w, 2)
        if v_key in v_counts:
            v_counts[v_key] += 1
        if w_key in w_counts:
            w_counts[w_key] += 1

    print("\n--- DATASET STATISTICAL DISTRIBUTION ---")
    print(f"v_robot = -{V_MAX} m/s : {v_counts[-V_MAX]:>6} ({100.0 * v_counts[-V_MAX] / len(final_dataset):.1f}%)")
    print(f"v_robot =  0.00 m/s : {v_counts[0.0]:>6} ({100.0 * v_counts[0.0] / len(final_dataset):.1f}%)")
    print(f"v_robot = +{V_MAX} m/s : {v_counts[V_MAX]:>6} ({100.0 * v_counts[V_MAX] / len(final_dataset):.1f}%)")
    print(f"w_robot = -{W_MAX} rad/s: {w_counts[-W_MAX]:>6} ({100.0 * w_counts[-W_MAX] / len(final_dataset):.1f}%)")
    print(f"w_robot =  0.00 rad/s: {w_counts[0.0]:>6} ({100.0 * w_counts[0.0] / len(final_dataset):.1f}%)")
    print(f"w_robot = +{W_MAX} rad/s: {w_counts[W_MAX]:>6} ({100.0 * w_counts[W_MAX] / len(final_dataset):.1f}%)")
    print(f"\n[Success] Dataset saved to: {args.output}\n")


if __name__ == "__main__":
    main()
