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

# ==============================================================================
# ⚙️ CONFIGURAÇÕES GLOBAIS DE GERAÇÃO DO DATASET (MODIFIQUE DIRETAMENTE AQUI)
# ==============================================================================
V_MAX = 0.50                   # Velocidade linear máxima em m/s {-V_MAX, 0.0, +V_MAX}
W_MAX = 1.00                   # Velocidade angular máxima em rad/s {-W_MAX, 0.0, +W_MAX}
X_MAX = 5.0                    # Limite da arena em X [-X_MAX, +X_MAX] (m)
Y_MAX = 5.0                    # Limite da arena em Y [-Y_MAX, +Y_MAX] (m)
STEP_XY = 0.2                # Resolução da grade espacial (m): de 0.25 em 0.25m
NUM_ANGLES = 24                # 24 ângulos discretos a cada 15 graus (passo de pi/12 rad)

# Reforço de Parada na Doca: (x=0, y=0, theta=0) -> (v=0, w=0)
NUM_DOCK_REINFORCEMENTS = 100  # Quantidade de réplicas da pose (0,0,0) parada para ancorar na rede

# Filtros e Modos de Geração
PURE_GRID_ONLY = True          # True = Apenas grade pura (sem valores decimais quebrados nem rollouts)
INCLUDE_DOCK_SUBGRID = False   # True = Inclui sub-grade fina (0.05m) | False = Mantém apenas múltiplos de 0.5m
INCLUDE_ROLLOUTS = False       # True = Inclui trajetórias dinâmicas contínuas | False = Sem números quebrados

D_BRAKE = 0.20                 # Distância de frenagem (m)
D_STOP = 0.05                  # Zona morta de parada final (m)
DT = 0.050                     # Período de controle 20 Hz (s)

COLUMNS = ["x_target_robot", "y_target_robot", "theta_robot", "v_robot", "w_robot"]
# ==============================================================================


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

    # 2. Braking Zone (d <= d_brake, e.g. 20 cm)
    # The linear speed drops to 0.0 so the network learns to brake smoothly!
    if d <= d_brake:
        v = 0.0
        if y_tr > 0.04:
            w = w_max   # Target at left -> Turn left
        elif y_tr < -0.04:
            w = -w_max  # Target at right -> Turn right
        elif abs(e_theta) > 0.08:
            w = -w_max if e_theta > 0 else w_max
        else:
            w = 0.0
        return v, w

    # 3. Reverse Docking Zone (Target is directly behind: x_tr < -d_brake and |y_tr| <= 1.5 m)
    if x_tr < -d_brake and abs(y_tr) <= 1.5:
        v = -v_max
        # When reversing like a car:
        # If target is to the left (y_tr > 0.05), turn left (w = +w_max)
        # If target is to the right (y_tr < -0.05), turn right (w = -w_max)
        if y_tr > 0.05:
            w = w_max
        elif y_tr < -0.05:
            w = -w_max
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

    # Target is in front: Drive forward and steer towards target
    v = v_max
    if phi > 0.05:
        w = w_max   # Target at left -> Turn left
    elif phi < -0.05:
        w = -w_max  # Target at right -> Turn right
    else:
        w = 0.0     # Drive straight

    return v, w


def generate_macro_grid(
    x_max: float = X_MAX,
    y_max: float = Y_MAX,
    step_xy: float = STEP_XY,
    num_angles: int = NUM_ANGLES,
    v_max: float = V_MAX,
    w_max: float = W_MAX
) -> List[Tuple[float, float, float, float, float]]:
    """
    Generates systematic grid samples over [-x_max, x_max] and [-y_max, y_max] every step_xy
    and across num_angles discrete directions (-150 deg to +180 deg in 30 deg steps).
    """
    samples: List[Tuple[float, float, float, float, float]] = []

    # Coordinates from -x_max to +x_max (ex: 21 steps from -5.0 to +5.0)
    n_steps_x = int(round(2 * x_max / step_xy))
    xs = [round(-x_max + i * step_xy, 2) for i in range(n_steps_x + 1)]

    n_steps_y = int(round(2 * y_max / step_xy))
    ys = [round(-y_max + i * step_xy, 2) for i in range(n_steps_y + 1)]

    # 12 ângulos discretos a cada 30 graus (-150° a +180°), contendo exatamente theta=0.0
    thetas = [round(-math.pi + (i + 1) * (2 * math.pi / num_angles), 4) for i in range(num_angles)]

    for x in xs:
        for y in ys:
            for th in thetas:
                v, w = compute_discrete_action(x, y, th, v_max=v_max, w_max=w_max)
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
        r_th = round(th, 4)
        if abs(abs(r_th) - 3.1416) < 1e-3:
            r_th = 3.1416
        key = (round(x, 2), round(y, 2), r_th)
        if key not in seen:
            seen.add(key)
            augmented.append((round(x, 2), round(y, 2), r_th, v, w))

        # Mirror sample
        mx = round(x, 2)
        my = round(-y, 2)
        mth = round(recast_angle(-th), 4)
        if abs(abs(mth) - 3.1416) < 1e-3:
            mth = 3.1416
        mv = v
        mw = -w

        mkey = (mx, my, mth)
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
        help=f"Max X coordinate limit in meters (default: {X_MAX})."
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=Y_MAX,
        help=f"Max Y coordinate limit in meters (default: {Y_MAX})."
    )
    parser.add_argument(
        "--step-xy",
        type=float,
        default=STEP_XY,
        help=f"Spatial grid resolution step in meters (default: {STEP_XY})."
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=NUM_ANGLES,
        help=f"Number of discrete angles across [-pi, +pi] (default: {NUM_ANGLES})."
    )
    parser.add_argument(
        "--dock-reinforce",
        type=int,
        default=NUM_DOCK_REINFORCEMENTS,
        help=f"Number of extra (0,0,0)->(0,0) stop samples to append (default: {NUM_DOCK_REINFORCEMENTS})."
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=V_MAX,
        help=f"Maximum linear velocity in m/s (default: {V_MAX})."
    )
    parser.add_argument(
        "--w-max",
        type=float,
        default=W_MAX,
        help=f"Maximum angular velocity in rad/s (default: {W_MAX})."
    )
    parser.add_argument(
        "--pure-grid",
        action="store_true",
        default=PURE_GRID_ONLY,
        help="Only include the regular grid and dock reinforcement, without decimal rollouts (default: True)."
    )
    parser.add_argument(
        "--include-subgrid",
        action="store_true",
        default=INCLUDE_DOCK_SUBGRID,
        help="Include dense fine docking subgrid (step 0.05m)."
    )
    parser.add_argument(
        "--include-rollouts",
        action="store_true",
        default=INCLUDE_ROLLOUTS,
        help="Include dynamic Euler trajectory rollouts at 20 Hz."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("\n" + "=" * 68)
    print("      LIMO OFFLINE LEARNING : GERADOR DE DATASET CANÔNICO")
    print("=" * 68)
    print(f"Arena Espacial       : [-{args.x_max:.1f}, +{args.x_max:.1f}] x [-{args.y_max:.1f}, +{args.y_max:.1f}] m")
    n_pts_x = int(round(2 * args.x_max / args.step_xy)) + 1
    n_pts_y = int(round(2 * args.y_max / args.step_xy)) + 1
    print(f"Resolução da Grade   : {args.step_xy:.2f} m ({n_pts_x}x{n_pts_y} = {n_pts_x * n_pts_y} posições)")
    print(f"Ângulos Discretos    : {args.num_angles} direções (a cada {360.0/args.num_angles:.1f}°)")
    print(f"Ações Lineares (v)   : {{-{args.v_max:.2f}, 0.00, +{args.v_max:.2f}}} m/s")
    print(f"Ações Angulares (w)  : {{-{args.w_max:.2f}, 0.00, +{args.w_max:.2f}}} rad/s")
    print(f"Reforço na Doca (0)  : {args.dock_reinforce} réplicas da pose (x=0, y=0, θ=0) -> (v=0, w=0)")
    print(f"Modo Grade Pura      : {'Ativado (sem valores decimais quebrados)' if args.pure_grid else 'Desativado'}")
    print(f"Arquivo de Saída     : {args.output}")
    print("=" * 68 + "\n")

    # 1. Macro Grid Generation (ex: 21 x 21 x 12 = 5,292 amostras puras)
    print(f"1. Gerando grade sistemática ({args.step_xy}m, {args.num_angles} ângulos)...")
    grid_samples = generate_macro_grid(
        x_max=args.x_max,
        y_max=args.y_max,
        step_xy=args.step_xy,
        num_angles=args.num_angles,
        v_max=args.v_max,
        w_max=args.w_max
    )
    print(f"   -> Amostras da grade pura: {len(grid_samples):,}")

    all_raw = list(grid_samples)

    if not args.pure_grid and args.include_subgrid:
        print("   -> Gerando sub-grade densa de docagem (passo 0.05m)...")
        dock_samples = generate_dense_docking_subgrid()
        all_raw.extend(dock_samples)

    if not args.pure_grid and args.include_rollouts:
        print("   -> Gerando rollouts dinâmicos a 20 Hz...")
        rollout_samples = generate_rollouts()
        all_raw.extend(rollout_samples)

    # 2. Simetria Sagital e Deduplicação
    print("2. Aplicando simetria sagital e deduplicação...")
    final_dataset = apply_bilateral_symmetry(all_raw)
    print(f"   -> Amostras únicas pós-simetria: {len(final_dataset):,}")

    # 3. Reforço da Doca (x=0, y=0, theta=0 -> v=0, w=0)
    if args.dock_reinforce > 0:
        print(f"3. Inserindo {args.dock_reinforce} réplicas de ancoragem na doca (0,0,0) -> (0,0)...")
        for _ in range(args.dock_reinforce):
            final_dataset.append((0.0, 0.0, 0.0, 0.0, 0.0))

    # Salva no arquivo CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in final_dataset:
            writer.writerow(row)

    # Estatísticas
    v_counts = { -args.v_max: 0, 0.0: 0, args.v_max: 0 }
    w_counts = { -args.w_max: 0, 0.0: 0, args.w_max: 0 }
    dock_zeros = 0
    for x, y, th, v, w in final_dataset:
        v_key = round(v, 2)
        w_key = round(w, 2)
        if v_key in v_counts:
            v_counts[v_key] += 1
        if w_key in w_counts:
            w_counts[w_key] += 1
        if abs(x) < 1e-4 and abs(y) < 1e-4 and abs(th) < 1e-4:
            dock_zeros += 1

    print("\n" + "-" * 50)
    print("  DISTRIBUIÇÃO ESTATÍSTICA DO DATASET GERADO")
    print("-" * 50)
    print(f"Total de amostras   : {len(final_dataset):,}")
    print(f"Amostras na Doca (0): {dock_zeros} ({100.0 * dock_zeros / len(final_dataset):.2f}%)")
    print(f"v_robot = -{args.v_max:.2f} m/s : {v_counts[-args.v_max]:>6} ({100.0 * v_counts[-args.v_max] / len(final_dataset):.1f}%)")
    print(f"v_robot =  0.00 m/s : {v_counts[0.0]:>6} ({100.0 * v_counts[0.0] / len(final_dataset):.1f}%)")
    print(f"v_robot = +{args.v_max:.2f} m/s : {v_counts[args.v_max]:>6} ({100.0 * v_counts[args.v_max] / len(final_dataset):.1f}%)")
    print(f"w_robot = -{args.w_max:.2f} rad/s: {w_counts[-args.w_max]:>6} ({100.0 * w_counts[-args.w_max] / len(final_dataset):.1f}%)")
    print(f"w_robot =  0.00 rad/s: {w_counts[0.0]:>6} ({100.0 * w_counts[0.0] / len(final_dataset):.1f}%)")
    print(f"w_robot = +{args.w_max:.2f} rad/s: {w_counts[args.w_max]:>6} ({100.0 * w_counts[args.w_max] / len(final_dataset):.1f}%)")
    print("-" * 50)
    print(f"\n[Sucesso] Dataset salvo em: {args.output}\n")


if __name__ == "__main__":
    main()
