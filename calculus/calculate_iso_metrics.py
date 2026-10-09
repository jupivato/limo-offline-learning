"""
Calcul des Caractéristiques de Pose Selon la Norme ISO 18646-2
==============================================================
Évalue métrologiquement les performances de navigation en boucle fermée :
- Précision de position (Position accuracy, Ap)
- Précision d'orientation (Orientation accuracy, Ao)
- Répétabilité de position (Position repeatability, Rp)
- Répétabilité d'orientation (Orientation repeatability, Ro)
- Indices d'erreur temporelle intégrale : IAE, ISE, ITSE, ITAE

Application aux fichiers de télémétrie CSV générés par DataLogger.
Génère des rapports structurés au format CSV et Excel (.xlsx).

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import glob
import csv
import math
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def recast_angle(angle_rad: float) -> float:
    """Recaste un angle dans l'intervalle (-pi, +pi] selon l'ISO 18646-2."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compute_iso_18646_metrics(trials: List[Dict[str, float]], xc: float = 0.0, yc: float = 0.0, oc: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Calcule l'ensemble des métriques de la norme ISO 18646-2 pour un groupe d'essais.

    Args:
        trials (List[Dict]): Liste des dictionnaires contenant xf, yf, thf (poses finales).
        xc (float): Coordonnée X de la cible.
        yc (float): Coordonnée Y de la cible.
        oc (float): Orientation de consigne de la cible (rad).

    Returns:
        Optional[Dict[str, Any]]: Métriques métrologiques ISO calculées.
    """
    n = len(trials)
    if n == 0:
        return None

    xs = [t["xf"] for t in trials]
    ys = [t["yf"] for t in trials]
    ths = [t["thf"] for t in trials]

    # 1. Barycentre des positions atteintes
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n

    # 2. Précision de position : Ap = sqrt((x_bar - xc)^2 + (y_bar - yc)^2)
    Ap = math.hypot(x_bar - xc, y_bar - yc)

    # 3. Écarts d'orientation zj = recast(oj - oc) et moyenne z_bar
    zs = [recast_angle(th - oc) for th in ths]
    z_bar = sum(zs) / n
    Ao_rad = z_bar
    Ao_deg = math.degrees(z_bar)
    Ao_abs_deg = abs(Ao_deg)

    # 4. Écarts radiaux au barycentre : lj = sqrt((x_bar - xj)^2 + (y_bar - yj)^2)
    ls = [math.hypot(x_bar - x, y_bar - y) for x, y in zip(xs, ys)]
    l_bar = sum(ls) / n

    # Écart-types d'échantillon (degrés de liberté n - 1)
    if n > 1:
        Sl = math.sqrt(sum((l - l_bar) ** 2 for l in ls) / (n - 1))
        So_rad = math.sqrt(sum((z - z_bar) ** 2 for z in zs) / (n - 1))
    else:
        Sl = 0.0
        So_rad = 0.0

    So_deg = math.degrees(So_rad)

    # 5. Répétabilité de position : Rp = l_bar + 3 * Sl
    Rp = l_bar + 3.0 * Sl

    # 6. Répétabilité d'orientation : Ro = 3 * So
    Ro_rad = 3.0 * So_rad
    Ro_deg = math.degrees(Ro_rad)

    # Statistiques complémentaires
    pos_errors = [math.hypot(x - xc, y - yc) for x, y in zip(xs, ys)]
    mean_err_pos = sum(pos_errors) / n

    return {
        "n": n,
        "xc": xc, "yc": yc, "oc": oc,
        "x_bar": x_bar, "y_bar": y_bar,
        "Ap": Ap,
        "Ao_deg": Ao_deg,
        "Ao_abs_deg": Ao_abs_deg,
        "l_bar": l_bar,
        "Sl": Sl,
        "Rp": Rp,
        "So_deg": So_deg,
        "Ro_deg": Ro_deg,
        "mean_err_pos": mean_err_pos,
        "min_err_pos": min(pos_errors),
        "max_err_pos": max(pos_errors)
    }


def compute_integral_errors(csv_path: str) -> Dict[str, float]:
    """
    Calcule les critères intégraux IAE, ISE, ITSE, ITAE sur l'erreur de distance.
    """
    t_vals, errs = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["timestamp"])
                ex = float(row.get("e_x", 0.0))
                ey = float(row.get("e_y", 0.0))
                e_dist = math.hypot(ex, ey)
                t_vals.append(t)
                errs.append(e_dist)
            except (ValueError, KeyError):
                continue

    if len(t_vals) < 2:
        return {"IAE": 0.0, "ISE": 0.0, "ITSE": 0.0, "ITAE": 0.0}

    iae = 0.0
    ise = 0.0
    itse = 0.0
    itae = 0.0

    for i in range(1, len(t_vals)):
        dt = t_vals[i] - t_vals[i - 1]
        mid_t = 0.5 * (t_vals[i] + t_vals[i - 1])
        mid_e = 0.5 * (errs[i] + errs[i - 1])

        iae += mid_e * dt
        ise += (mid_e ** 2) * dt
        itse += mid_t * (mid_e ** 2) * dt
        itae += mid_t * mid_e * dt

    return {
        "IAE": round(iae, 2),
        "ISE": round(ise, 2),
        "ITSE": round(itse, 2),
        "ITAE": round(itae, 2)
    }


def parse_final_pose(csv_path: str) -> Optional[Dict[str, Any]]:
    """Lit la pose finale (dernière ligne) d'un fichier de télémétrie CSV."""
    last_row = None
    first_row = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if first_row is None:
                first_row = row
            last_row = row

    if not last_row or not first_row:
        return None

    try:
        x0 = float(first_row.get("x", 0.0))
        y0 = float(first_row.get("y", 0.0))
        th0 = float(first_row.get("theta", 0.0))
        xf = float(last_row.get("x", 0.0))
        yf = float(last_row.get("y", 0.0))
        thf = float(last_row.get("theta", 0.0))
        tx = float(last_row.get("tx", 0.0))
        ty = float(last_row.get("ty", 0.0))
        tth = float(last_row.get("ttheta", 0.0))

        quadrant = 1
        if (x0 - tx) >= 0 and (y0 - ty) >= 0:
            quadrant = 1
        elif (x0 - tx) < 0 and (y0 - ty) >= 0:
            quadrant = 2
        elif (x0 - tx) < 0 and (y0 - ty) < 0:
            quadrant = 3
        else:
            quadrant = 4

        return {
            "file": os.path.basename(csv_path),
            "x0": x0, "y0": y0, "th0": th0,
            "xf": xf, "yf": yf, "thf": thf,
            "tx": tx, "ty": ty, "tth": tth,
            "quadrant": quadrant
        }
    except ValueError:
        return None


def evaluate_directory(input_dir: str = "data/evaluation", output_dir: str = "calculus") -> None:
    """
    Scanne les CSV d'un dossier, calcule les métriques ISO et génère les rapports.
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"[calculate_iso_metrics] Aucun fichier CSV trouvé dans {input_dir}.")
        return

    trials = []
    trials_by_quad = defaultdict(list)

    for path in sorted(csv_files):
        parsed = parse_final_pose(path)
        if parsed:
            # Ajout des erreurs intégrales
            integ = compute_integral_errors(path)
            parsed.update(integ)
            trials.append(parsed)
            trials_by_quad[parsed["quadrant"]].append(parsed)

    if not trials:
        print("[calculate_iso_metrics] Aucun essai valide extrait.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Export du détail de chaque essai
    trials_csv = os.path.join(output_dir, "all_trials_detailed.csv")
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    # 2. Métriques globales et par quadrant
    quad_summary = []
    print("\n" + "=" * 70)
    print("--- RÉSULTATS MÉTHODOLOGIQUES ISO 18646-2 ---")
    print(f"Nombre total d'essais analysés : {len(trials)}")
    print("=" * 70)

    for q in sorted(trials_by_quad.keys()):
        group = trials_by_quad[q]
        metrics = compute_iso_18646_metrics(group)
        if metrics:
            metrics["quadrant"] = q
            quad_summary.append(metrics)
            print(
                f"Quadrant {q} (n={metrics['n']}) | "
                f"Précision Pos (Ap): {metrics['Ap']:.3f} m | "
                f"Répétabilité Pos (Rp): {metrics['Rp']:.3f} m | "
                f"Précision Angle (Ao): {metrics['Ao_abs_deg']:.2f}°"
            )

    global_metrics = compute_iso_18646_metrics(trials)
    if global_metrics:
        global_metrics["quadrant"] = "GLOBAL"
        quad_summary.append(global_metrics)
        print("-" * 70)
        print(
            f"GLOBAL (n={global_metrics['n']})     | "
            f"Précision Pos (Ap): {global_metrics['Ap']:.3f} m | "
            f"Répétabilité Pos (Rp): {global_metrics['Rp']:.3f} m | "
            f"Précision Angle (Ao): {global_metrics['Ao_abs_deg']:.2f}°"
        )
    print("=" * 70 + "\n")

    summary_csv = os.path.join(output_dir, "iso_metrics_by_quadrant.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(quad_summary[0].keys()))
        writer.writeheader()
        writer.writerows(quad_summary)

    print(f"[Succès] Rapports métrologiques sauvegardés dans {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moteur de calcul métrologique ISO 18646-2.")
    parser.add_argument("--dir", type=str, default="data/evaluation", help="Dossier contenant les CSV")
    parser.add_argument("--outdir", type=str, default="calculus", help="Dossier de sortie des métriques")
    args = parser.parse_args()

    evaluate_directory(input_dir=args.dir, output_dir=args.outdir)
