"""
ISO 18646-2 Mobile Robot Positioning Metrology Calculation Engine
=================================================================
Evaluates closed-loop autonomous navigation performance under ISO 18646-2:
- Position Accuracy (Ap)
- Orientation Accuracy (Ao)
- Position Repeatability (Rp)
- Orientation Repeatability (Ro)
- Integral Time-Error Indices: IAE, ISE, ITSE, ITAE

Processes telemetry CSV logs recorded by DataLogger.
Generates structured metric summaries in CSV format.

Project: LIMO Offline Learning (CERV / ENIB)
Standards: ROS REP-103 & ISO 18646-2
"""

from __future__ import annotations

import os
import re
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
    """Wraps an angle into the canonical interval (-pi, +pi] per ISO 18646-2."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compute_iso_18646_metrics(
    trials: List[Dict[str, float]],
    xc: float = 0.0,
    yc: float = 0.0,
    oc: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Computes the complete set of ISO 18646-2 metrological characteristics for a group of trials.

    Args:
        trials (List[Dict]): List of dictionaries containing final poses: xf, yf, thf.
        xc (float): Target X coordinate (m).
        yc (float): Target Y coordinate (m).
        oc (float): Nominal target orientation (rad).

    Returns:
        Optional[Dict[str, Any]]: Computed ISO 18646-2 metrological indicators.
    """
    n = len(trials)
    if n == 0:
        return None

    xs = [t["xf"] for t in trials]
    ys = [t["yf"] for t in trials]
    ths = [t["thf"] for t in trials]

    # 1. Attained barycenter coordinates
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n

    # 2. Position Accuracy: Ap = sqrt((x_bar - xc)^2 + (y_bar - yc)^2)
    Ap = math.hypot(x_bar - xc, y_bar - yc)

    # 3. Orientation deviation zj = recast(oj - oc) and sample mean z_bar
    zs = [recast_angle(th - oc) for th in ths]
    z_bar = sum(zs) / n
    Ao_rad = z_bar
    Ao_deg = math.degrees(z_bar)
    Ao_abs_deg = abs(Ao_deg)

    # 4. Radial deviations from barycenter: lj = sqrt((x_bar - xj)^2 + (y_bar - yj)^2)
    ls = [math.hypot(x_bar - x, y_bar - y) for x, y in zip(xs, ys)]
    l_bar = sum(ls) / n

    # Sample standard deviations (degrees of freedom n - 1)
    if n > 1:
        Sl = math.sqrt(sum((l - l_bar) ** 2 for l in ls) / (n - 1))
        So_rad = math.sqrt(sum((z - z_bar) ** 2 for z in zs) / (n - 1))
    else:
        Sl = 0.0
        So_rad = 0.0

    So_deg = math.degrees(So_rad)

    # 5. Position Repeatability: Rp = l_bar + 3 * Sl
    Rp = l_bar + 3.0 * Sl

    # 6. Orientation Repeatability: Ro = 3 * So
    Ro_rad = 3.0 * So_rad
    Ro_deg = math.degrees(Ro_rad)

    # Complementary statistical metrics
    pos_errors = [math.hypot(x - xc, y - yc) for x, y in zip(xs, ys)]
    mean_err_pos = sum(pos_errors) / n
    abs_zs_deg = [math.degrees(abs(z)) for z in zs]
    mean_abs_angle_err_deg = sum(abs_zs_deg) / n

    return {
        "n": n,
        "xc": xc, "yc": yc, "oc": oc,
        "x_bar": x_bar, "y_bar": y_bar,
        "Ap": Ap,
        "Ao_deg": Ao_deg,
        "Ao_abs_deg": Ao_abs_deg,
        "mean_abs_angle_err_deg": mean_abs_angle_err_deg,
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
    Computes integral error performance indices (IAE, ISE, ITSE, ITAE) on Euclidean distance error.
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
    """Extracts the final resting pose (last row) from a telemetry CSV file."""
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
        ex = float(last_row.get("e_x", 0.0))
        ey = float(last_row.get("e_y", 0.0))
        eth = float(last_row.get("e_theta", 0.0))

        # Infer target coordinates from pose and error if not explicitly logged
        tx = float(last_row.get("tx", xf - ex))
        ty = float(last_row.get("ty", yf - ey))
        tth = float(last_row.get("ttheta", recast_angle(thf - eth)))

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
    Scans telemetry CSV files in a directory, computes ISO metrics, and exports reports.
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files and os.path.isdir(os.path.join(input_dir, "data")):
        input_dir = os.path.join(input_dir, "data")
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Exclude generated metric summaries and aggregated files to process trial runs only
    csv_files = [
        c for c in csv_files
        if not any(k in os.path.basename(c).lower() for k in ["iso", "summary", "trial", "metrics"])
    ]

    if not csv_files:
        print(f"[calculate_iso_metrics] No CSV files found in: {input_dir}")
        return

    trials = []
    trials_by_quad = defaultdict(list)
    trials_by_scenario = defaultdict(list)

    for path in sorted(csv_files):
        parsed = parse_final_pose(path)
        if parsed:
            integ = compute_integral_errors(path)
            parsed.update(integ)
            trials.append(parsed)
            trials_by_quad[parsed["quadrant"]].append(parsed)
            fname = parsed["file"].lower()
            m = re.search(r"run_(\d+)_q(\d+)", fname)
            if m:
                scen_id = int(m.group(1))
            else:
                scen_id = len(trials_by_scenario) + 1
            trials_by_scenario[scen_id].append(parsed)

    if not trials:
        print("[calculate_iso_metrics] No valid trials extracted.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Export detailed individual trial metrics
    trials_csv = os.path.join(output_dir, "all_trials_detailed.csv")
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    # 2. Compute per-starting-pose (scenario) ISO 18646-2 metrics (Clause 4.2 Unidirectional)
    scenario_summary = []
    print("\n" + "=" * 78)
    print("--- ISO 18646-2 UNIDIRECTIONAL METROLOGY BY STARTING POSITION ---")
    print("=" * 78)
    for scen_id in sorted(trials_by_scenario.keys()):
        s_group = trials_by_scenario[scen_id]
        s_metrics = compute_iso_18646_metrics(
            s_group,
            xc=s_group[0].get("tx", 0.0),
            yc=s_group[0].get("ty", 0.0),
            oc=s_group[0].get("tth", 0.0)
        )
        if s_metrics:
            nominal_trials = [t for t in s_group if not t["file"].lower().startswith("rep")]
            ref = nominal_trials[0] if nominal_trials else s_group[0]
            succ_count = sum(
                1 for t in s_group
                if math.hypot(t["xf"] - t["tx"], t["yf"] - t["ty"]) <= 0.08
                and abs(recast_angle(t["thf"] - t["tth"])) <= 0.26
            )
            row = {
                "scenario_id": scen_id,
                "scenario_name": f"Pose {scen_id} - Q{ref['quadrant']}",
                "quadrant": ref["quadrant"],
                "x0": round(ref["x0"], 2),
                "y0": round(ref["y0"], 2),
                "th0_deg": round(math.degrees(ref["th0"]), 1),
                "reps": s_metrics["n"],
                "successes": succ_count,
                "success_rate_pct": round(100.0 * succ_count / s_metrics["n"], 1),
                "Ap_m": round(s_metrics["Ap"], 4),
                "Rp_m": round(s_metrics["Rp"], 4),
                "Ao_deg": round(s_metrics["Ao_abs_deg"], 2),
                "Ro_deg": round(s_metrics["Ro_deg"], 2),
                "Sl_m": round(s_metrics["Sl"], 4),
                "So_deg": round(s_metrics["So_deg"], 2),
                "mean_dist_err_m": round(s_metrics["mean_err_pos"], 4),
            }
            scenario_summary.append(row)
            print(
                f"Pose {scen_id:02d} (Q{ref['quadrant']}, n={row['reps']}) | "
                f"Ap: {row['Ap_m']*100:.2f} cm | Rp: {row['Rp_m']*100:.2f} cm | "
                f"Ao: {row['Ao_deg']:.2f}° | Ro: {row['Ro_deg']:.2f}°"
            )

    if scenario_summary:
        scen_csv = os.path.join(output_dir, "iso_metrics_by_scenario.csv")
        with open(scen_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scenario_summary[0].keys()))
            writer.writeheader()
            writer.writerows(scenario_summary)

    # 3. Compute global and per-quadrant ISO 18646-2 metrics
    quad_summary = []
    print("\n" + "=" * 70)
    print("--- ISO 18646-2 METROLOGICAL EVALUATION RESULTS ---")
    print(f"Total Evaluated Trials: {len(trials)}")
    print("=" * 70)

    for q in sorted(trials_by_quad.keys()):
        group = trials_by_quad[q]
        metrics = compute_iso_18646_metrics(
            group,
            xc=group[0].get("tx", 0.0),
            yc=group[0].get("ty", 0.0),
            oc=group[0].get("tth", 0.0)
        )
        if metrics:
            metrics["quadrant"] = q
            quad_summary.append(metrics)
            print(
                f"Quadrant {q} (n={metrics['n']}) | "
                f"Pos Accuracy (Ap): {metrics['Ap']:.3f} m | "
                f"Pos Repeatability (Rp): {metrics['Rp']:.3f} m | "
                f"Angle Accuracy (Ao): {metrics['Ao_abs_deg']:.2f}°"
            )

    global_metrics = compute_iso_18646_metrics(
        trials,
        xc=trials[0].get("tx", 0.0),
        yc=trials[0].get("ty", 0.0),
        oc=trials[0].get("tth", 0.0)
    )
    if global_metrics:
        global_metrics["quadrant"] = "GLOBAL"
        quad_summary.append(global_metrics)
        print("-" * 70)
        print(
            f"GLOBAL (n={global_metrics['n']})     | "
            f"Pos Accuracy (Ap): {global_metrics['Ap']:.3f} m | "
            f"Pos Repeatability (Rp): {global_metrics['Rp']:.3f} m | "
            f"Angle Accuracy (Ao): {global_metrics['Ao_abs_deg']:.2f}°"
        )
    print("=" * 70 + "\n")

    summary_csv = os.path.join(output_dir, "iso_metrics_by_quadrant.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(quad_summary[0].keys()))
        writer.writeheader()
        writer.writerows(quad_summary)

    print(f"[Success] Metrological reports exported to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISO 18646-2 Metrological Calculation Engine.")
    parser.add_argument("--dir", type=str, default="data/evaluation", help="Directory containing CSV logs")
    parser.add_argument("--outdir", type=str, default="calculus", help="Output directory for reports")
    args = parser.parse_args()

    evaluate_directory(input_dir=args.dir, output_dir=args.outdir)
