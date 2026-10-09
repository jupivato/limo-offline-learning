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

import os
import sys
import time
import math
import csv
import argparse
from typing import List, Dict, Any, Optional

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

ALPHA = [1.0 / 3.0, 1.0 / 3.0, 1.0 / math.pi]


def recast_angle(angle_rad: float) -> float:
    """Recalcule un angle dans l'intervalle principal (-pi, +pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def theta_strategic(x: float, y: float) -> float:
    """Orientation stratégique ENIB évitant les minima locaux."""
    return math.tanh(10.0 * x) * math.atan(1.0 * y)


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
    target: List[float] = [0.0, 0.0, 0.0],
    max_duration: float = 60.0,
    output_dir: str = "data/evaluation/batch_run",
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

    for step in range(max_steps):
        elapsed = step * dt
        e_x = x - target[0]
        e_y = y - target[1]
        dist = math.hypot(e_x, e_y)
        e_theta = recast_angle(th - target[2])

        th_s = theta_strategic(x, y)
        in_0 = e_x * ALPHA[0]
        in_1 = e_y * ALPHA[1]
        in_2 = (e_theta - th_s) * ALPHA[2]

        cost = (ALPHA[0] ** 2 * e_x ** 2 +
                ALPHA[1] ** 2 * e_y ** 2 +
                ALPHA[2] ** 2 * (e_theta - th_s) ** 2)

        # Condition de convergence
        if step > 10 and dist < dist_tol and abs(e_theta) < 0.35:
            print(f"    [Succès] Atteint en {elapsed:.1f}s ! (Erreur dist={dist:.3f}m, angle={abs(e_theta):.3f}rad [{math.degrees(abs(e_theta)):.1f}°])")
            success = True
            break

        # Inférence neuronale
        cmd_raw = model([in_0, in_1, in_2])
        if hasattr(cmd_raw, "tolist"):
            cmd = cmd_raw.tolist()
        else:
            cmd = list(cmd_raw)

        v_cmd = max(-0.16, min(0.26, float(cmd[0])))
        w_cmd = max(-0.45, min(0.45, float(cmd[1])))

        logger.log(position=[x, y, th], target=target, command=[v_cmd, w_cmd], grad=[0.0, 0.0], cost=cost)

        # Intégration cinématique différentielle (Euler explicite à 20 Hz)
        x += v_cmd * math.cos(th) * dt
        y += v_cmd * math.sin(th) * dt
        th = recast_angle(th + w_cmd * dt)

    if not success:
        print(f"    [Fin de course] Durée {elapsed:.1f}s. (Pose finale: x={x:.3f}, y={y:.3f}, th={th:.3f})")

    # Sauvegarde des traces CSV et tracés
    os.makedirs(output_dir, exist_ok=True)
    slug = run_name.lower().replace(" ", "_").replace("-", "").replace("__", "_")
    csv_path = os.path.join(output_dir, f"{slug}.csv")
    png_path = os.path.join(output_dir, f"{slug}.png")
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

    while rclpy.ok():
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            print(f"    [Timeout] Temps limite de {max_duration}s atteint.")
            break

        pos = robot.get_position()
        e_x = pos[0] - target[0]
        e_y = pos[1] - target[1]
        dist = math.hypot(e_x, e_y)
        e_theta = recast_angle(pos[2] - target[2])

        th_s = theta_strategic(pos[0], pos[1])
        in_0 = e_x * ALPHA[0]
        in_1 = e_y * ALPHA[1]
        in_2 = (e_theta - th_s) * ALPHA[2]

        cost = (ALPHA[0] ** 2 * e_x ** 2 +
                ALPHA[1] ** 2 * e_y ** 2 +
                ALPHA[2] ** 2 * (e_theta - th_s) ** 2)

        if elapsed > 0.5 and dist < dist_tol and abs(e_theta) < 0.35:
            print(f"    [Succès] Atteint en {elapsed:.1f}s ! (Erreur dist={dist:.3f}m, angle={abs(e_theta):.3f}rad [{math.degrees(abs(e_theta)):.1f}°])")
            success = True
            break

        cmd_raw = model([in_0, in_1, in_2])
        if hasattr(cmd_raw, "tolist"):
            cmd = cmd_raw.tolist()
        else:
            cmd = list(cmd_raw)

        v_cmd = max(-0.16, min(0.26, float(cmd[0])))
        w_cmd = max(-0.45, min(0.45, float(cmd[1])))
        robot.set_cmd_vel(v_cmd, w_cmd)
        logger.log(position=pos, target=target, command=[v_cmd, w_cmd], grad=[0.0, 0.0], cost=cost)

        time.sleep(dt)

    robot.stop()
    time.sleep(0.1)

    os.makedirs(output_dir, exist_ok=True)
    slug = run_name.lower().replace(" ", "_").replace("-", "").replace("__", "_")
    csv_path = os.path.join(output_dir, f"{slug}.csv")
    png_path = os.path.join(output_dir, f"{slug}.png")
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
    output_dir: str = "data/evaluation/batch_run",
    use_gazebo: bool = False,
    suite: str = "extended"
) -> None:
    """Exécute l'ensemble des scénarios de test et synthétise les métriques ISO 18646-2."""
    scenarios = EXTENDED_TEST_SCENARIOS if suite == "extended" else CANONICAL_TEST_SCENARIOS

    print("\n" + "=" * 76)
    print("      LIMO OFFLINE LEARNING : BANC D'ESSAI AUTOMATISÉ ISO 18646-2")
    print("=" * 76)
    print(f"Modèle testé        : {weights_path}")
    print(f"Mode d'exécution    : {'Gazebo (ROS 2)' if use_gazebo else 'Simulation Cinématique (20 Hz)'}")
    print(f"Suite d'essais      : {suite.upper()} ({len(scenarios)} scénarios multi-distances/orientations)")
    print(f"Dossier de sortie   : {output_dir}")
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
                        output_dir=output_dir
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
                output_dir=output_dir
            )
            results.append(res)

    if not results:
        print("Aucun essai complété.")
        return

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

        # Sauvegarde du rapport CSV consolidé
        iso_csv_path = os.path.join(output_dir, "iso_18646_metrics.csv")
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

        # 4. Génération de la suite graphique unifiée comparative
        try:
            from src.evaluation.benchmark_visualizer import generate_all_evaluation_reports
            generate_all_evaluation_reports(output_dir)
        except Exception as e:
            print(f"[Visualizer] Note : Erreur lors de la génération graphique : {e}")

    print("=" * 76 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banc d'essai automatisé ISO 18646-2 pour LIMO.")
    parser.add_argument("--weights", "--model", dest="weights", type=str, default="models/supervised_w_torch_diff.json", help="Chemin du JSON des poids")
    parser.add_argument("--outdir", type=str, default="data/evaluation/batch_run", help="Dossier de sortie des métriques")
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
