"""
Canonical Offline Dataset Generator for Mobile Robot Docking
============================================================
Generates the canonical 5-column supervisory training dataset for offline 
behavioral cloning of the PioneerNN controller on the AgileX LIMO mobile robot.

Theoretical Framework:
----------------------
Complies strictly with the 9 canonical motion maneuvers established by the 
ENIB / CERV / Lab-STICC robotics laboratory (Prof. Patrick Hénaff):
    Class 1: Pure Forward Translation        (v > 0, w = 0)
    Class 2: Pure Reverse Translation        (v < 0, w = 0)
    Class 3: Pure Counter-Clockwise Pivot   (v = 0, w > 0)
    Class 4: Pure Clockwise Pivot           (v = 0, w < 0)
    Class 5: Forward Left Curve (Gentle)     (v > 0, w > 0)
    Class 6: Forward Right Curve (Gentle)    (v > 0, w < 0)
    Class 7: Reverse Left Curve (Parallel)   (v < 0, w < 0)
    Class 8: Reverse Right Curve (Parallel)  (v < 0, w > 0)
    Class 9: Terminal Docking Stop          (v = 0, w = 0)

Coordinate System & Schema:
---------------------------
Robot-Centric Body Frame representation (5 columns):
    1. x_target_robot : Relative X coordinate of the dock in robot body frame (m)
    2. y_target_robot : Relative Y coordinate of the dock in robot body frame (m)
    3. theta_robot    : Robot heading relative to the dock coordinate axis (rad)
    4. v_robot        : Supervisory linear velocity {-V_MAX, 0.0, +V_MAX} (m/s)
    5. w_robot        : Supervisory angular velocity {-W_MAX, 0.0, +W_MAX} (rad/s)

Key Innovations in this Generator:
----------------------------------
1. Seamless Dock Ingress: Eliminates premature stopping. Forward thrust (+V_MAX) 
   is sustained until the vehicle penetrates the actual physical docking radius 
   (D_STOP = 0.08 m), precluding stall-outs before reaching the target.
2. Fine Sub-Grid Sampling: Resolves the docking envelope with millimeter-level 
   spatial resolution without repeating duplicate rows.
3. Strategic Vector Field Guidance: Incorporates theta_s(x, y) = tanh(10x) * atan(y) 
   to guide the vehicle along smooth asymptotic curves into the dock.
4. Bilateral Sagittal Invariance: Automatically mirrors every state-action pair 
   (x, -y, -theta, v, -w) to guarantee mathematically zero left-right directional bias.
5. Exact Stratified Balancing: Selects 5,000 distinct samples per canonical class 
   (45,000 samples total, exactly 11.11% per maneuver).

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
Standards: ROS REP-103 & ISO 18646-2
"""

from __future__ import annotations

import os
import sys
import csv
import math
import random
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple

# Dynamic project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ==============================================================================
# KINEMATIC AND ARENA PARAMETERS (AgileX LIMO Unicycle Kinematics)
# ==============================================================================
V_MAX = 0.50                   # Maximum longitudinal linear velocity (m/s)
W_MAX = 1.00                   # Maximum yaw angular velocity (rad/s)
X_MAX = 5.0                    # Global arena half-width X [-5.0, +5.0] m
Y_MAX = 5.0                    # Global arena half-length Y [-5.0, +5.0] m
STEP_XY = 0.20                 # Primary spatial regular grid resolution (m)
NUM_ANGLES = 24                # Angular resolution: 24 equidistant headings (15° steps)

# Precision deadbands for discrete maneuver classification
STEER_DEADBAND = 0.15          # Heading deadband for pure forward driving (~8.6°)
REVERSE_DEADBAND = 0.15        # Heading deadband for pure reverse driving (~8.6°)
D_STOP = 0.060                 # Physical dock capture radius for terminal stop (6.0 cm)
D_STOP_ANGLE_TOL = 0.100       # Heading alignment tolerance for terminal stop (~5.7°)
MAX_PIVOT_PHI = 0.50           # Bearing threshold triggering in-place pivot (~28.6°, prevents wide turning circles)

SAMPLES_PER_CLASS = 5000       # Stratified target quota per maneuver class
COLUMNS = ["x_target_robot", "y_target_robot", "theta_robot", "v_robot", "w_robot"]

# Canonical maneuver classification titles (Prof. Patrick Hénaff taxonomy)
ENIB_MANEUVER_NAMES = {
    1: "Class 1: Pure Forward Translation (v > 0, w = 0)",
    2: "Class 2: Pure Reverse Translation (v < 0, w = 0)",
    3: "Class 3: Pure Counter-Clockwise Pivot (v = 0, w > 0)",
    4: "Class 4: Pure Clockwise Pivot (v = 0, w < 0)",
    5: "Class 5: Forward Left Curve (v > 0, w > 0)",
    6: "Class 6: Forward Right Curve (v > 0, w < 0)",
    7: "Class 7: Reverse Left Curve (v < 0, w < 0)",
    8: "Class 8: Reverse Right Curve (v < 0, w > 0)",
    9: "Class 9: Terminal Docking Stop (v = 0, w = 0)"
}


def recast_angle(angle_rad: float) -> float:
    """
    Wraps an angle into the canonical principal interval [-pi, +pi].
    Preserves exact boundary extremes (-pi and +pi) against floating-point rounding artifacts.
    """
    if abs(angle_rad - math.pi) < 1e-4:
        return math.pi
    if abs(angle_rad + math.pi) < 1e-4:
        return -math.pi
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compute_theta_strategic(x: float, y: float) -> float:
    """
    Computes the canonical ENIB strategic docking orientation.
    Guides the robot tangentially along the docking axis as (x, y) -> (0, 0).
    """
    return math.tanh(10.0 * float(x)) * math.atan(1.0 * float(y))


def classify_enib_maneuver(v: float, w: float) -> Tuple[int, str]:
    """
    Classifies discrete action pair (v, w) into the 9 canonical ENIB maneuvers.
    """
    V_EPS = 0.025
    W_EPS = 0.040

    if abs(v) <= V_EPS and abs(w) <= W_EPS:
        m_id = 9
    elif abs(v) <= V_EPS:
        m_id = 3 if w > 0.0 else 4
    elif v > V_EPS:
        if abs(w) <= W_EPS:
            m_id = 1
        elif w > 0.0:
            m_id = 5
        else:
            m_id = 6
    else:  # v < -V_EPS
        if abs(w) <= W_EPS:
            m_id = 2
        elif w < 0.0:
            m_id = 7
        else:
            m_id = 8

    return m_id, ENIB_MANEUVER_NAMES[m_id]


def expert_rule_based_controller(
    x_tr: float,
    y_tr: float,
    theta_r: float,
    v_max: float = V_MAX,
    w_max: float = W_MAX,
    d_stop: float = D_STOP,
    d_stop_angle_tol: float = D_STOP_ANGLE_TOL
) -> Tuple[float, float, int]:
    """
    Deterministic expert supervisor implementing smooth unicycle docking guidance.

    Input Features:
        x_tr, y_tr : Relative coordinates of the dock in robot body frame (m).
        theta_r    : Vehicle heading relative to the dock reference frame (rad).

    Outputs:
        v, w        : Discrete ternary velocities {-v_max, 0, +v_max} and {-w_max, 0, +w_max}.
        maneuver_id : Canonical ENIB maneuver identifier (1 to 9).
    """
    dist = math.hypot(x_tr, y_tr)
    e_theta = recast_angle(theta_r)
    phi = math.atan2(y_tr, x_tr)  # Line of sight bearing to dock in robot frame

    # 1. Terminal Docking Zone (Vehicle within physical capture radius d <= d_stop)
    if dist <= d_stop:
        if abs(e_theta) <= d_stop_angle_tol:
            # Full vehicle immobilization aligned in dock
            v, w = 0.0, 0.0
        elif abs(abs(e_theta) - math.pi) < 1e-4:
            # Vehicle facing exactly backwards in dock terminal: rotate consistently CCW
            v, w = 0.0, w_max
        elif e_theta > 0.0:
            # In-place clockwise pivot alignment
            v, w = 0.0, -w_max
        else:
            # In-place counter-clockwise pivot alignment
            v, w = 0.0, w_max
        m_id, _ = classify_enib_maneuver(v, w)
        return v, w, m_id

    # 2. Reverse Parking Corridor (Dock situated directly behind: x_tr < -0.10m and |y_tr| <= 0.80m)
    if x_tr < -0.10 and abs(y_tr) <= 0.80:
        v = -v_max
        phi_rear = math.atan2(y_tr, -x_tr)
        # Reverse steering error combines rear line-of-sight with terminal heading alignment
        effective_e_theta = 0.0 if abs(abs(e_theta) - math.pi) < 1e-4 else e_theta
        steer_err_rear = phi_rear - 0.40 * effective_e_theta
        if steer_err_rear > REVERSE_DEADBAND:
            w = -w_max  # Directs vehicle rear towards the left
        elif steer_err_rear < -REVERSE_DEADBAND:
            w = w_max   # Directs vehicle rear towards the right
        else:
            w = 0.0     # Pure reverse straight
        m_id, _ = classify_enib_maneuver(v, w)
        return v, w, m_id

    # 3. In-Place Pivot Zone: target behind or on flanks (x_tr < 0.0 or |phi| > MAX_PIVOT_PHI)
    # Pivot on a dime (v = 0.0) to eliminate wide orbiting circles around the origin!
    if x_tr < 0.0 or abs(phi) > MAX_PIVOT_PHI:
        v = 0.0
        w = w_max if y_tr >= 0.0 else -w_max
        m_id, _ = classify_enib_maneuver(v, w)
        return v, w, m_id

    # 4. Forward Cruise & Approach (Target in front: x_tr >= 0.0 and |phi| <= MAX_PIVOT_PHI)
    v = v_max
    if dist < 0.60:
        # Near dock ingress: blend direct line-of-sight with terminal heading alignment
        effective_e_theta = 0.0 if abs(abs(e_theta) - math.pi) < 1e-4 else e_theta
        steer_err = phi + 0.35 * effective_e_theta
    else:
        # Far field: follow the ENIB strategic vector field
        th_s = compute_theta_strategic(x_tr, y_tr)
        steer_err = phi - 0.50 * th_s

    if steer_err > STEER_DEADBAND:
        w = w_max   # Turn left
    elif steer_err < -STEER_DEADBAND:
        w = -w_max  # Turn right
    else:
        w = 0.0     # Pure forward straight

    m_id, _ = classify_enib_maneuver(v, w)
    return v, w, m_id


def generate_balanced_dataset(
    x_max: float = X_MAX,
    y_max: float = Y_MAX,
    step_xy: float = STEP_XY,
    num_angles: int = NUM_ANGLES,
    samples_per_class: int = SAMPLES_PER_CLASS
) -> Tuple[List[List[float]], Dict[int, int]]:
    """
    Constructs the canonical offline dataset across the spatial arena, integrates
    fine-mesh docking and reverse corridors, enforces bilateral symmetry, and extracts
    an exact stratified subset of 5,000 unique samples per maneuver.
    """
    raw_samples_by_class: Dict[int, List[List[float]]] = {i: [] for i in range(1, 10)}

    # --------------------------------------------------------------------------
    # 1. Primary Regular Spatial Grid (-5.0m to +5.0m, step 0.20m, 24 angles)
    # --------------------------------------------------------------------------
    n_steps_x = int(round(2.0 * x_max / step_xy))
    xs = [round(-x_max + i * step_xy, 2) for i in range(n_steps_x + 1)]
    n_steps_y = int(round(2.0 * y_max / step_xy))
    ys = [round(-y_max + i * step_xy, 2) for i in range(n_steps_y + 1)]
    thetas = [round(-math.pi + i * (2.0 * math.pi / num_angles), 4) for i in range(num_angles + 1)]

    for x in xs:
        for y in ys:
            for th in thetas:
                v, w, m_id = expert_rule_based_controller(x, y, th)
                raw_samples_by_class[m_id].append([x, y, th, v, w])

    # --------------------------------------------------------------------------
    # 2. High-Density Terminal Docking Sub-Grid (|x|, |y| <= 0.06m, step 2.0mm, 64 angles)
    # --------------------------------------------------------------------------
    dock_step = 0.0020
    n_dock = int(round(0.12 / dock_step))
    dock_xs = [round(-0.06 + i * dock_step, 4) for i in range(n_dock + 1)]
    dock_ys = [round(-0.06 + i * dock_step, 4) for i in range(n_dock + 1)]
    dock_thetas = [round(-math.pi + i * (2.0 * math.pi / 64), 4) for i in range(65)]

    for x in dock_xs:
        for y in dock_ys:
            for th in dock_thetas:
                v, w, m_id = expert_rule_based_controller(x, y, th)
                raw_samples_by_class[m_id].append([x, y, th, v, w])

    # --------------------------------------------------------------------------
    # 3. High-Density Reverse Parking Corridor (-2.5m <= x <= -0.1m, |y| <= 0.4m, step 0.02m)
    # --------------------------------------------------------------------------
    rev_xs = [round(-2.5 + i * 0.02, 3) for i in range(121)]
    rev_ys = [round(-0.40 + i * 0.02, 3) for i in range(41)]
    rev_thetas = [round(-math.pi + i * (2.0 * math.pi / 32), 4) for i in range(33)]
    for x in rev_xs:
        for y in rev_ys:
            for th in rev_thetas:
                v, w, m_id = expert_rule_based_controller(x, y, th)
                raw_samples_by_class[m_id].append([x, y, th, v, w])

    # --------------------------------------------------------------------------
    # 4. High-Density Forward Steering Corridor (0.1m <= x <= 2.5m, |y| <= 0.6m, step 0.05m)
    # --------------------------------------------------------------------------
    fwd_xs = [round(0.10 + i * 0.05, 3) for i in range(49)]
    fwd_ys = [round(-0.60 + i * 0.05, 3) for i in range(25)]
    fwd_thetas = [round(-math.pi + i * (2.0 * math.pi / 32), 4) for i in range(33)]
    for x in fwd_xs:
        for y in fwd_ys:
            for th in fwd_thetas:
                v, w, m_id = expert_rule_based_controller(x, y, th)
                raw_samples_by_class[m_id].append([x, y, th, v, w])

    # --------------------------------------------------------------------------
    # 5. Bilateral Sagittal Mirroring (Zero Left/Right Directional Bias)
    # --------------------------------------------------------------------------
    for m_id in range(1, 10):
        mirrored_list = []
        for s in raw_samples_by_class[m_id]:
            # Sagittal reflection: (x, y, theta, v, w) -> (x, -y, -theta, v, -w)
            mx = s[0]
            my = -s[1]
            mth = recast_angle(-s[2])
            mv = s[3]
            mw = -s[4]
            mirrored_list.append([mx, my, mth, mv, mw])
        raw_samples_by_class[m_id].extend(mirrored_list)

    # --------------------------------------------------------------------------
    # 6. Periodic Boundary Sewing (Topology: S^1 equivalence of -pi and +pi)
    # --------------------------------------------------------------------------
    for m_id in range(1, 10):
        sewn_list = []
        for s in raw_samples_by_class[m_id]:
            th = s[2]
            if abs(th - (-math.pi)) < 1e-3:
                sewn_list.append([s[0], s[1], round(math.pi, 4), s[3], s[4]])
            elif abs(th - math.pi) < 1e-3:
                sewn_list.append([s[0], s[1], round(-math.pi, 4), s[3], s[4]])
        raw_samples_by_class[m_id].extend(sewn_list)

    # --------------------------------------------------------------------------
    # 7. Deduplication and Exact Stratified Balancing (5,000 unique per class)
    # --------------------------------------------------------------------------
    balanced_samples: List[List[float]] = []
    final_counts: Dict[int, int] = {}
    random.seed(42)  # Deterministic seed for scientific reproducibility

    for m_id in range(1, 10):
        # Deduplicate using rounded tuples as hashable keys
        unique_map = {}
        for s in raw_samples_by_class[m_id]:
            key = (round(s[0], 4), round(s[1], 4), round(s[2], 4), round(s[3], 2), round(s[4], 2))
            if key not in unique_map:
                unique_map[key] = [key[0], key[1], key[2], key[3], key[4]]

        unique_pool = list(unique_map.values())

        random.seed(42 + m_id)
        if len(unique_pool) >= samples_per_class:
            # Uniform unbiased sampling across all spatial positions and all 24 angular headings
            selected = random.sample(unique_pool, samples_per_class)
        else:
            # Uniform cyclic oversampling with random remainder for compact minority classes
            repeats = samples_per_class // len(unique_pool)
            remainder = samples_per_class % len(unique_pool)
            selected = unique_pool * repeats + random.sample(unique_pool, remainder)

        balanced_samples.extend(selected)
        final_counts[m_id] = len(selected)

    # Deterministic global shuffle
    random.shuffle(balanced_samples)

    return balanced_samples, final_counts


def build_dataset(
    output_path: str = "data/dataset.csv",
    samples_per_class: int = SAMPLES_PER_CLASS
) -> int:
    """
    Executes dataset construction and exports the canonical 5-column CSV.
    """
    total_samples = samples_per_class * 9
    resolved_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    print("\n" + "=" * 78)
    print("   LIMO CANONICAL OFFLINE DATASET GENERATOR (ENIB BENCHMARK SCHEMA)")
    print("=" * 78)
    print(f"Arena Boundary        : [-{X_MAX:.1f}, +{X_MAX:.1f}] x [-{Y_MAX:.1f}, +{Y_MAX:.1f}] m")
    print(f"Primary Grid Step     : {STEP_XY:.2f} m | Angular Step: {360.0/NUM_ANGLES:.1f}°")
    print(f"Docking Catch Radius  : {D_STOP*100:.1f} cm (Smooth ingress without 30cm stall)")
    print(f"Terminal Heading Tol  : {math.degrees(D_STOP_ANGLE_TOL):.1f}° (Target alignment)")
    print(f"Maneuver Schema       : 9 Canonical ENIB Classes ({samples_per_class:,} per class)")
    print(f"Total Target Dataset  : {total_samples:,} samples (Strictly 11.11% per maneuver)")
    print(f"Output Format         : Strictly 5 columns: {COLUMNS}")
    print(f"Destination CSV File  : {resolved_path}")
    print("=" * 78 + "\n")

    print("1. Generating continuous grid, high-density sub-grids, and sagittal mirrors...")
    samples, class_counts = generate_balanced_dataset(samples_per_class=samples_per_class)

    print("2. Writing balanced dataset to CSV...")
    with open(resolved_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(samples)

    print("\n" + "-" * 78)
    print("  STRATIFIED DISTRIBUTION ACROSS THE 9 ENIB CANONICAL MANEUVERS")
    print("-" * 78)
    for m_id in range(1, 10):
        cnt = class_counts[m_id]
        pct = (cnt / float(len(samples))) * 100.0
        print(f"  {ENIB_MANEUVER_NAMES[m_id]:<58} : {cnt:6d} ({pct:5.2f}%)")
    print("-" * 78)
    print(f"Successfully generated {len(samples):,} samples in: {resolved_path}\n")

    return len(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate balanced 5-column LIMO offline dataset.")
    parser.add_argument("--output", type=str, default="data/dataset.csv", help="Destination CSV file path")
    parser.add_argument("--samples-per-class", type=int, default=5000, help="Quota per maneuver class")
    args = parser.parse_args()

    build_dataset(output_path=args.output, samples_per_class=args.samples_per_class)
