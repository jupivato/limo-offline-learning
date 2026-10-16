"""
Banc d'Essai Automatisé Multi-Quadrants (Batch Benchmark ISO 18646-2)
=====================================================================
Exécute la batterie de 8 tests canoniques multi-quadrants, enregistre les
trajectoires, génère les graphiques et calcule l'ensemble des métriques
métrologiques de la norme ISO 18646-2 (Ap, Ao, Rp, Ro).

Modes d'exécution :
1. Mode Simulation Cinématique Boucle Fermée (défaut / autonome) :
   Simule la dynamique différentielle du LIMO à 20 Hz avec le réseau PioneerNN.
   Permet d'évaluer instantanément les performances métrologiques sans dépendre
   d'une instance Gazebo active.
2. Mode Gazebo / ROS 2 (activé via --gazebo ou si Gazebo est actif) :
   Interagit directement avec Gazebo via /odom et /cmd_vel avec téléportation.

Projet : LIMO Offline Learning (CERV / ENIB)
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
# Inclusion automatique du dossier racine du projet dans sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Inclusion automatique du chemin dist-packages snap si présent
_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
for _p in [
    f"/home/juliapivato/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages",
    os.path.expanduser(f"~/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/.local/lib/python{_py_ver}/site-packages"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from src.core.pioneer_nn import PioneerNN
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
# ⚙️ CONFIGURAÇÕES GLOBAIS DE DESNORMALIZAÇÃO E LIMITES CINEMÁTICOS (MODIFIQUE AQUI)
# ==============================================================================
V_MAX = 0.50  # Velocidade linear máxima de desnormalização em m/s (dobrada de 0.24 para 0.50)
W_MAX = 1.00  # Velocidade angular máxima de desnormalização em rad/s (dobrada de 0.50 para 1.00)
X_MAX = 5.0   # Limite da arena em X (m)
Y_MAX = 5.0   # Limite da arena em Y (m)
ALPHA = [1.0 / X_MAX, 1.0 / Y_MAX, 1.0 / math.pi]
# ==============================================================================


def recast_angle(angle_rad: float) -> float:
    """Recalcule un angle dans l'intervalle principal (-pi, +pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


# 1. Suite Canonique ISO (8 essais de référence - 2 par quadrant)
CANONICAL_TEST_SCENARIOS = [
    ("Test 1 - Q1 Frontal",    2.0,  2.0,  0.0,            1),
    ("Test 2 - Q1 Latéral",    1.5,  2.5, -math.pi / 2,    1),
    ("Test 3 - Q2 Frontal",   -2.0,  2.0,  0.0,            2),
    ("Test 4 - Q2 Latéral",   -2.5,  1.5, -math.pi / 2,    2),
    ("Test 5 - Q3 Inversé",   -2.0, -2.0,  math.pi,        3),
    ("Test 6 - Q3 Latéral",   -1.5, -2.5,  math.pi / 2,    3),
    ("Test 7 - Q4 Latéral",    2.0, -2.0, -math.pi / 2,    4),
    ("Test 8 - Q4 Inversé",    2.5, -1.5,  math.pi,        4),
]

# 2. Bateria Expandida Multi-Distância & Multi-Rotação (20 ensaios sistemáticos - 5 por quadrant)
# Distâncias: Curta (~1.2m), Média (~2.5m), Longa (~3.5m)
# Rotações: 0°, ±45°, ±90°, ±135°, 180°
EXTENDED_TEST_SCENARIOS = [
    # --- QUADRANTE 1 (+X, +Y) : BALIZA / RÉ ---
    ("Test 01 - Q1 Curto Frontal",   1.0,  0.8,  0.0,                 1),
    ("Test 02 - Q1 Médio Frontal",   2.0,  2.0,  0.0,                 1),
    ("Test 03 - Q1 Médio Latéral",   1.5,  2.5, -round(math.pi/2, 3), 1),
    ("Test 04 - Q1 Médio Oblíquo",   2.2,  1.2,  round(math.pi/4, 3), 1),
    ("Test 05 - Q1 Longo Inversé",   3.0,  1.5,  round(math.pi, 3),   1),

    # --- QUADRANTE 2 (-X, +Y) : AVANÇO DUBINS ---
    ("Test 06 - Q2 Curto Frontal",  -1.0,  0.8,  0.0,                 2),
    ("Test 07 - Q2 Médio Frontal",  -2.0,  2.0,  0.0,                 2),
    ("Test 08 - Q2 Médio Latéral",  -2.5,  1.5, -round(math.pi/2, 3), 2),
    ("Test 09 - Q2 Médio Oblíquo",  -2.2,  1.2, -round(math.pi/4, 3), 2),
    ("Test 10 - Q2 Longo Oblíquo",  -3.0,  2.0,  round(math.pi/4, 3), 2),

    # --- QUADRANTE 3 (-X, -Y) : AVANÇO DUBINS ---
    ("Test 11 - Q3 Curto Inversé",  -1.0, -0.8,  round(math.pi, 3),   3),
    ("Test 12 - Q3 Médio Inversé",  -2.0, -2.0,  round(math.pi, 3),   3),
    ("Test 13 - Q3 Médio Latéral",  -1.5, -2.5,  round(math.pi/2, 3), 3),
    ("Test 14 - Q3 Médio Oblíquo",  -2.2, -1.2,  round(3*math.pi/4, 3), 3),
    ("Test 15 - Q3 Longo Frontal",  -3.0, -1.8,  0.0,                 3),

    # --- QUADRANTE 4 (+X, -Y) : BALIZA / RÉ ---
    ("Test 16 - Q4 Curto Latéral",   1.0, -0.8, -round(math.pi/2, 3), 4),
    ("Test 17 - Q4 Médio Latéral",   2.0, -2.0, -round(math.pi/2, 3), 4),
    ("Test 18 - Q4 Médio Inversé",   2.5, -1.5,  round(math.pi, 3),   4),
    ("Test 19 - Q4 Médio Oblíquo",   2.2, -1.2, -round(math.pi/4, 3), 4),
    ("Test 20 - Q4 Longo Oblíquo",   3.0, -1.8, -round(3*math.pi/4, 3), 4),
]


def run_simulated_benchmark_run(
    model: PioneerNN,
    x0: float,
    y0: float,
    th0: float,
    run_name: str,
    target: list[float] = [0.0, 0.0, 0.0],
    max_duration: float = 60.0,
    output_dir: str = "data/evaluation/batch_run",
    output_images_dir: Optional[str] = None,
    dist_tol: float = 0.20
) -> Dict[str, Any]:
    """Exécute un essai en simulation cinématique à 20 Hz en boucle fermée."""
    print(f"\n>>> Démarrage (Simulation) : {run_name} (Départ: x={x0:.2f}, y={y0:.2f}, th={th0:.2f} rad)")

    logger = DataLogger()
    dt = 0.050  # 20 Hz
    max_steps = int(max_duration / dt)

    x, y, th = x0, y0, th0
    success = False
    elapsed = 0.0
    pos_history: collections.deque[Tuple[float, float, float, float]] = collections.deque(maxlen=30)
    zero_cmd_count = 0

    for step in range(max_steps):
        elapsed = step * dt
        dx = target[0] - x
        dy = target[1] - y
        theta_r = recast_angle(th)
        dist = math.hypot(dx, dy)
        e_theta = recast_angle(th - target[2])

        # Coordonnées de la cible dans le repère local du robot
        x_target_robot = dx * math.cos(theta_r) + dy * math.sin(theta_r)
        y_target_robot = -dx * math.sin(theta_r) + dy * math.cos(theta_r)
        theta_robot_abs = theta_r

        # Normalisation des entrées
        in_0 = max(-1.0, min(1.0, x_target_robot / X_MAX))
        in_1 = max(-1.0, min(1.0, y_target_robot / Y_MAX))
        in_2 = max(-1.0, min(1.0, theta_robot_abs / math.pi))

        cost = in_0 ** 2 + in_1 ** 2 + in_2 ** 2

        # Condition de convergence
        if step > 10 and dist < dist_tol and abs(e_theta) < 0.35:
            print(f"    [Succès] Atteint en {elapsed:.1f}s ! (Erreur dist={dist:.3f}m, angle={abs(e_theta):.3f}rad [{math.degrees(abs(e_theta)):.1f}°])")
            success = True
            break

        # Détection d'immobilisation (Stall Detection) : coupure anticipée dès que le robot s'arrête
        pos_history.append((elapsed, x, y, th))
        if elapsed > 1.5 and len(pos_history) >= 30:
            old_t, old_x, old_y, old_th = pos_history[0]
            d_moved = math.hypot(x - old_x, y - old_y)
            d_rot = abs(recast_angle(th - old_th))
            if d_moved < 0.030 and d_rot < 0.050:
                print(f"    [Parada Detectada] Robô parou totalmente seus movimentos (deslocamento < 3cm em 1.5s a t={elapsed:.1f}s). Encerrando ensaio.")
                if dist < dist_tol and abs(e_theta) < 0.35:
                    success = True
                break

        # Inférence neuronale
        cmd_raw = model([in_0, in_1, in_2])
        if hasattr(cmd_raw, "tolist"):
            cmd = cmd_raw.tolist()
        else:
            cmd = list(cmd_raw)

        v_cmd = max(-V_MAX, min(V_MAX, float(cmd[0]) * V_MAX))
        w_cmd = max(-W_MAX, min(W_MAX, float(cmd[1]) * W_MAX))

        # Arrêt immédiat si la politique commande l'arrêt persistant
        if abs(v_cmd) < 0.02 and abs(w_cmd) < 0.03:
            zero_cmd_count += 1
        else:
            zero_cmd_count = 0

        if elapsed > 1.0 and zero_cmd_count >= 15:
            print(f"    [Parada Detectada] Comandos de velocidade zerados continuamente (v=0, w=0 a t={elapsed:.1f}s). Encerrando ensaio.")
            if dist < dist_tol and abs(e_theta) < 0.35:
                success = True
            break

        logger.log(position=[x, y, th], target=target, command=[v_cmd, w_cmd], grad=[0.0, 0.0], cost=cost)

        # Intégration cinématique différentielle (Euler explicite à 20 Hz)
        x += v_cmd * math.cos(th) * dt
        y += v_cmd * math.sin(th) * dt
        th = recast_angle(th + w_cmd * dt)

    if not success:
        print(f"    [Fin de course] Durée {elapsed:.1f}s. (Pose finale: x={x:.3f}, y={y:.3f}, th={th:.3f})")

    # Sauvegarde des traces CSV et tracés
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

    return {
        "name": run_name,
        "success": success,
        "elapsed": elapsed,
        "xf": x,
        "yf": y,
        "thf": th,
        "err_dist": err_dist,
        "err_theta": err_theta,
    }


def run_gazebo_benchmark_run(
    robot: Any,
    model: PioneerNN,
    x0: float,
    y0: float,
    th0: float,
    run_name: str,
    target: List[float] = [0.0, 0.0, 0.0],
    max_duration: float = 60.0,
    output_dir: str = "data/evaluation/batch_run",
    output_images_dir: Optional[str] = None,
    dist_tol: float = 0.25
) -> Dict[str, Any]:
    """Exécute un essai individuel sous Gazebo avec téléportation automatique."""
    print(f"\n>>> Démarrage (Gazebo) : {run_name} (Départ: x={x0:.2f}, y={y0:.2f}, th={th0:.2f} rad)")

    robot.teleport_gazebo(x0, y0, th0)
    time.sleep(0.3)

    logger = DataLogger()
    start_time = time.time()
    dt = 0.050
    success = False
    pos_history: collections.deque[Tuple[float, float, float, float]] = collections.deque(maxlen=30)
    zero_cmd_count = 0

    while rclpy.ok():
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            print(f"    [Timeout] Temps limite de {max_duration}s atteint.")
            break

        pos = robot.get_position()
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        theta_r = recast_angle(pos[2])
        dist = math.hypot(dx, dy)
        e_theta = recast_angle(pos[2] - target[2])

        # Coordonnées de la cible dans le repère local du robot
        x_target_robot = dx * math.cos(theta_r) + dy * math.sin(theta_r)
        y_target_robot = -dx * math.sin(theta_r) + dy * math.cos(theta_r)
        theta_robot_abs = theta_r

        # Normalisation des entrées
        in_0 = max(-1.0, min(1.0, x_target_robot / X_MAX))
        in_1 = max(-1.0, min(1.0, y_target_robot / Y_MAX))
        in_2 = max(-1.0, min(1.0, theta_robot_abs / math.pi))

        cost = in_0 ** 2 + in_1 ** 2 + in_2 ** 2

        if elapsed > 0.5 and dist < dist_tol and abs(e_theta) < 0.35:
            print(f"    [Succès] Atteint en {elapsed:.1f}s ! (Erreur dist={dist:.3f}m, angle={abs(e_theta):.3f}rad [{math.degrees(abs(e_theta)):.1f}°])")
            success = True
            break

        # Détection d'immobilisation (Stall Detection) : coupure anticipée dès que le robot s'arrête
        pos_history.append((elapsed, pos[0], pos[1], pos[2]))
        if elapsed > 1.5 and len(pos_history) >= 30:
            old_t, old_x, old_y, old_th = pos_history[0]
            d_moved = math.hypot(pos[0] - old_x, pos[1] - old_y)
            d_rot = abs(recast_angle(pos[2] - old_th))
            if d_moved < 0.030 and d_rot < 0.050:
                print(f"    [Parada Detectada] Robô parou totalmente seus movimentos (deslocamento < 3cm em 1.5s a t={elapsed:.1f}s). Encerrando ensaio.")
                if dist < dist_tol and abs(e_theta) < 0.35:
                    success = True
                break

        cmd_raw = model([in_0, in_1, in_2])
        if hasattr(cmd_raw, "tolist"):
            cmd = cmd_raw.tolist()
        else:
            cmd = list(cmd_raw)

        v_cmd = max(-V_MAX, min(V_MAX, float(cmd[0]) * V_MAX))
        w_cmd = max(-W_MAX, min(W_MAX, float(cmd[1]) * W_MAX))

        # Arrêt immédiat si la politique commande l'arrêt persistant
        if abs(v_cmd) < 0.02 and abs(w_cmd) < 0.03:
            zero_cmd_count += 1
        else:
            zero_cmd_count = 0

        if elapsed > 1.0 and zero_cmd_count >= 15:
            print(f"    [Parada Detectada] Comandos de velocidade zerados continuamente (v=0, w=0 a t={elapsed:.1f}s). Encerrando ensaio.")
            if dist < dist_tol and abs(e_theta) < 0.35:
                success = True
            break

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
    return {
        "name": run_name,
        "success": success,
        "elapsed": time.time() - start_time,
        "xf": final_pos[0],
        "yf": final_pos[1],
        "thf": final_pos[2],
        "err_dist": math.hypot(final_pos[0] - target[0], final_pos[1] - target[1]),
        "err_theta": abs(recast_angle(final_pos[2] - target[2])),
    }

def run_batch_benchmark(
    weights_path: str = "models/supervised_w_torch_diff.json",
    output_dir: Optional[str] = None,
    use_gazebo: bool = False,
    suite: str = "extended"
) -> Dict[str, Any]:
    """Exécute l'ensemble des scénarios de test et synthétise les métriques ISO 18646-2."""
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

    print("\n" + "=" * 76)
    print("      LIMO OFFLINE LEARNING : BANC D'ESSAI AUTOMATISÉ ISO 18646-2")
    print("=" * 76)
    print(f"Modèle testé        : {weights_path}")
    print(f"Mode d'exécution    : {'Gazebo (ROS 2)' if use_gazebo else 'Simulation Cinématique (20 Hz)'}")
    print(f"Desnormalização     : V_MAX = {V_MAX:.2f} m/s | W_MAX = {W_MAX:.2f} rad/s")
    print(f"Suite d'essais      : {suite.upper()} ({len(scenarios)} scénarios multi-distances/orientations)")
    print(f"Dossier du test     : {test_dir}")
    print(f"  ├── Données CSV   : {data_dir}")
    print(f"  └── Graphiques    : {images_dir}")
    print("=" * 76)

    # 1. Chargement du modèle neural
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2)
    model.load_from_json_file(weights_path)
    model.eval()

    results = []

    # 2. Exécution selon le mode choisi
    if use_gazebo:
        if not ROS2_AVAILABLE:
            print("[Attention] rclpy ou ROS 2 n'est pas disponible. Basculement en mode Simulation Cinématique.")
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
                        output_dir=data_dir,
                        output_images_dir=images_dir
                    )
                    results.append(res)
                    time.sleep(0.5)
            except Exception as e:
                print(f"[Avertissement Gazebo] Erreur lors de la connexion Gazebo: {e}")
                print("Basculement en mode Simulation Cinématique...")
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
                output_dir=data_dir,
                output_images_dir=images_dir
            )
            results.append(res)

    if not results:
        print("Aucun essai complété.")
        return {}

    # 3. Calcul des métriques ISO 18646-2
    trials_for_iso = [{"xf": r["xf"], "yf": r["yf"], "thf": r["thf"]} for r in results]
    iso_metrics = compute_iso_18646_metrics(trials_for_iso, xc=0.0, yc=0.0, oc=0.0)

    print("\n" + "=" * 76)
    print("           RÉSULTATS DE LA NORME ISO 18646-2 (BOUCLE FERMÉE)")
    print("=" * 76)
    if iso_metrics:
        print(f"  • Nombre d'essais (n)             : {iso_metrics['n']}")
        print(f"  • Précision de position (Ap)      : {iso_metrics['Ap']:.4f} m  (exigence : < 0.25 m)")
        print(f"  • Précision d'orientation (Ao)    : {iso_metrics['Ao_abs_deg']:.2f}° (exigence : < 15.0°)")
        print(f"  • Répétabilité de position (Rp)   : {iso_metrics['Rp']:.4f} m")
        print(f"  • Répétabilité d'orientation (Ro) : {iso_metrics['Ro_deg']:.2f}°")
        print(f"  • Écart-type radial (Sl)          : {iso_metrics['Sl']:.4f} m")
        print(f"  • Écart-type angulaire (So)       : {iso_metrics['So_deg']:.2f}°")
        print(f"  • Barycentre atteint (X, Y)       : ({iso_metrics['x_bar']:.4f}, {iso_metrics['y_bar']:.4f}) m")

        # Sauvegarde du rapport CSV consolidé dans data/
        iso_csv_path = os.path.join(data_dir, "iso_18646_metrics.csv")
        with open(iso_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Métrique ISO 18646-2", "Valeur", "Unité", "Statut"])
            writer.writerow(["Précision de position (Ap)", f"{iso_metrics['Ap']:.4f}", "m", "CONFORME" if iso_metrics['Ap'] < 0.25 else "TOLÉRANCE"])
            writer.writerow(["Précision d'orientation (Ao)", f"{iso_metrics['Ao_abs_deg']:.2f}", "deg", "CONFORME" if iso_metrics['Ao_abs_deg'] < 15.0 else "TOLÉRANCE"])
            writer.writerow(["Répétabilité de position (Rp)", f"{iso_metrics['Rp']:.4f}", "m", "OK"])
            writer.writerow(["Répétabilité d'orientation (Ro)", f"{iso_metrics['Ro_deg']:.2f}", "deg", "OK"])
            writer.writerow(["Écart-type radial (Sl)", f"{iso_metrics['Sl']:.4f}", "m", "OK"])
            writer.writerow(["Écart-type angulaire (So)", f"{iso_metrics['So_deg']:.2f}", "deg", "OK"])
        print(f"\nRapport ISO exporté dans : {iso_csv_path}")

        # 4. Génération de la suite graphique unifiée comparative dans images/
        try:
            from src.evaluation.benchmark_visualizer import generate_all_evaluation_reports
            generate_all_evaluation_reports(
                batch_dir=data_dir,
                prefix="benchmark",
                regenerate_sessions=False,
                output_images_dir=images_dir
            )
        except Exception as e:
            print(f"[Visualizer] Note : Erreur lors de la génération graphique : {e}")

        # Espelhamento para compatibilidade com scripts legados
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

    # 5. Sauvegarde de params.json dans test_yyyy_mm_dd_sec_min_hr
    successful_runs = sum(1 for r in results if r["success"])
    params = {
        "test_id": timestamp_folder,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weights_path": os.path.abspath(weights_path),
        "execution_mode": "Gazebo (ROS 2)" if use_gazebo else "Simulation Cinématique (20 Hz)",
        "suite": suite,
        "total_runs": len(results),
        "successful_runs": successful_runs,
        "success_rate_pct": round(100.0 * successful_runs / max(1, len(results)), 2),
        "average_elapsed_sec": round(sum(r["elapsed"] for r in results) / max(1, len(results)), 2),
        "kinematic_limits": {
            "v_max_mps": V_MAX,
            "w_max_radps": W_MAX,
            "x_max_m": X_MAX,
            "y_max_m": Y_MAX,
            "alpha": ALPHA,
            "dt_sec": 0.05,
            "dist_tol_m": 0.20 if not use_gazebo else 0.25,
            "angle_tol_rad": 0.35,
            "stall_detection": {
                "enabled": True,
                "window_sec": 1.5,
                "min_displacement_m": 0.030,
                "min_rotation_rad": 0.050,
                "zero_cmd_steps": 15
            }
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
                "final_pose": [round(r["xf"], 4), round(r["yf"], 4), round(r["thf"], 4)],
                "err_dist_m": round(r["err_dist"], 4),
                "err_theta_deg": round(math.degrees(r["err_theta"]), 2)
            }
            for r in results
        ],
        "artifacts": {
            "test_dir": os.path.abspath(test_dir),
            "params_json": os.path.abspath(os.path.join(test_dir, "params.json")),
            "data_dir": os.path.abspath(data_dir),
            "images_dir": os.path.abspath(images_dir),
            "iso_csv": os.path.abspath(os.path.join(data_dir, "iso_18646_metrics.csv")),
            "trajectory_plot": os.path.abspath(os.path.join(images_dir, "benchmark_trajectory.png")),
            "dashboard_html": os.path.abspath(os.path.join(images_dir, "benchmark_dashboard.html"))
        }
    }
    params_path = os.path.join(test_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"[Sucesso] Parâmetros e metadados do teste salvos em: {params_path}")

    print("=" * 76 + "\n")
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banc d'essai automatisé ISO 18646-2 pour LIMO.")
    parser.add_argument("--weights", "--model", dest="weights", type=str, default="models/supervised_w_torch_diff.json", help="Chemin du JSON des poids")
    parser.add_argument("--outdir", type=str, default="data/evaluation", help="Dossier de sortie des métriques")
    parser.add_argument("--suite", type=str, default="extended", choices=["extended", "canonical"], help="Suite d'essais (extended: 20 essais, canonical: 8 essais)")
    parser.add_argument("--gazebo", action="store_true", help="Force l'exécution dans Gazebo via ROS 2")
    parser.add_argument("--sim", action="store_true", help="Force le mode Simulation Cinématique pure")
    args = parser.parse_args()

    use_gazebo = args.gazebo and not args.sim
    run_batch_benchmark(
        weights_path=args.weights,
        output_dir=args.outdir,
        use_gazebo=use_gazebo,
        suite=args.suite
    )
