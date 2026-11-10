"""
Confusion Matrix & Dataset Diagnostics Generator — AgileX LIMO
==============================================================
Evaluates the trained PioneerNN model across the 45,000 canonical dataset samples,
computes discrete classification confusion matrices for linear and angular commands,
and renders high-resolution (300 DPI) diagnostic figures for research publications.

Generated Plots:
1. Linear Velocity Confusion Matrix (Reverse, Stopped, Forward)
2. Angular Velocity Confusion Matrix (Clockwise, Straight, Counter-Clockwise)
3. 2D Spatial Coverage Density Map (45,000 samples)
4. Comparative Distribution Histograms (ENIB Supervisor vs Neural Controller)

Project: LIMO Offline Learning (CERV / ENIB)
Standards: ROS REP-103 & ISO 18646-2
"""

import os
import sys
import csv
import json
import math
from typing import Tuple, List, Dict, Any

# Dynamic project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.pioneer_nn import PioneerNN, compute_k_theta

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

DATASET_CSV = "data/dataset.csv"
WEIGHTS_PATH = "models/supervised_w_torch_diff.json"
OUT_IMG_DIR = "docs/images"
OUT_EVAL_DIR = "data/evaluation"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_EVAL_DIR, exist_ok=True)

# ==============================================================================
# GLOBAL KINEMATIC BOUNDARIES & NORMALIZATION LIMITS
# ==============================================================================
V_MAX = 0.50  # Maximum forward linear velocity (m/s)
W_MAX = 1.00  # Maximum angular yaw velocity (rad/s)
X_MAX = 5.0   # Arena spatial boundary X (m)
Y_MAX = 5.0   # Arena spatial boundary Y (m)
# ==============================================================================


def load_dataset_and_predict(dataset_path: str, weights_path: str):
    """Loads dataset, executes batch inference, and returns targets and predictions."""
    print(f"1. Loading neural model weights: {weights_path}")
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2, use_bias=True)
    model.load_from_json_file(weights_path)
    model.eval()

    print(f"2. Loading canonical dataset: {dataset_path}")
    inputs = []
    targets_v = []
    targets_w = []
    coords_x = []
    coords_y = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xr = float(row.get("x_target_robot", row.get("x_r", 0.0)))
                yr = float(row.get("y_target_robot", row.get("y_r", 0.0)))
                thr = float(row.get("theta_robot", row.get("theta_r", 0.0)))
                raw_v = float(row.get("v_robot", row.get("v_r", 0.0)))
                raw_w = float(row.get("w_robot", row.get("w_r", 0.0)))

                thr_norm = math.atan2(math.sin(thr), math.cos(thr))
                in_0 = max(-1.0, min(1.0, xr / X_MAX))
                in_1 = max(-1.0, min(1.0, yr / Y_MAX))
                in_2 = max(-1.0, min(1.0, thr_norm / math.pi))

                norm_v = max(-1.0, min(1.0, raw_v / V_MAX))
                norm_w = max(-1.0, min(1.0, raw_w / W_MAX))

                inputs.append([in_0, in_1, in_2])
                targets_v.append(norm_v)
                targets_w.append(norm_w)
                coords_x.append(xr)
                coords_y.append(yr)
            except (KeyError, ValueError):
                continue

    print(f"   Valid samples loaded: {len(inputs):,}")

    print("3. Executing batch inference across entire dataset...")
    import torch
    with torch.no_grad():
        X_tensor = torch.tensor(inputs, dtype=torch.float32)
        preds = model(X_tensor).numpy()

    preds_v = preds[:, 0]
    preds_w = preds[:, 1]

    return np.array(inputs), np.array(targets_v), np.array(targets_w), preds_v, preds_w, np.array(coords_x), np.array(coords_y)


def discretize_action(values, thresholds=(-0.15, 0.15)):
    """Discretizes continuous values into 3 classes: 0=Negative, 1=Neutral/Stopped, 2=Positive."""
    classes = np.zeros_like(values, dtype=int)
    classes[values < thresholds[0]] = 0
    classes[(values >= thresholds[0]) & (values <= thresholds[1])] = 1
    classes[values > thresholds[1]] = 2
    return classes


def compute_confusion_matrix(y_true, y_pred, n_classes=3):
    """Computes an n_classes x n_classes confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_pretty_confusion_matrix(cm, class_names, title, save_path, cmap_color="Blues"):
    """Plots a styled confusion matrix with absolute counts and recall percentages."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap_color, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Proportion (Recall)", rotation=-90, va="bottom", fontsize=10, fontweight="bold")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")

    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > thresh else "#0f172a"
            text = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    ax.set_ylabel("Ground-Truth Class (ENIB Supervisor / Dataset)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Class (PioneerNN)", fontsize=11, fontweight="bold", labelpad=8)

    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.01, f"Overall Class Accuracy: {accuracy*100:.2f}% | Total: {np.sum(cm):,} samples",
                 ha="center", fontsize=10, style="italic", color="#475569")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Saved] {save_path}")


def plot_dataset_spatial_distribution(x_coords, y_coords, save_path):
    """Generates 2D hexbin density map of samples collected in the 5x5m arena."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)

    hb = ax.hexbin(x_coords, y_coords, gridsize=35, cmap="viridis", mincnt=1, zorder=2)
    cb = fig.colorbar(hb, ax=ax, orientation="vertical", pad=0.03)
    cb.set_label("Collected Sample Density", fontsize=10, fontweight="bold")

    # Mark docking target (0, 0)
    ax.scatter([0.0], [0.0], color="#dc2626", s=180, marker="*", zorder=5, label="Dock Target (0.0, 0.0)")
    circle_dock = plt.Circle((0, 0), 0.08, color="#dc2626", fill=False, linestyle="--", linewidth=1.5, zorder=4, label="Docking Radius (0.08 m)")
    ax.add_patch(circle_dock)

    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Relative Position X (meters)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Relative Position Y (meters)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(f"LIMO Dataset Spatial Coverage — {len(x_coords):,} Samples\n5m x 5m Arena (Multi-Quadrant Sagittal Symmetry)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Saved] {save_path}")


def plot_prediction_histograms(targets_v, preds_v, targets_w, preds_w, save_path):
    """Plots comparative histograms between supervisor discrete targets and continuous network predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # 1. Linear Velocity
    ax1.set_facecolor("#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax1.hist(targets_v * V_MAX, bins=30, alpha=0.55, color="#dc2626", label="ENIB Supervisor Target", zorder=2)
    ax1.hist(preds_v * V_MAX, bins=30, alpha=0.65, color="#2563eb", label="PioneerNN Prediction (Continuous Tanh)", zorder=3)
    ax1.set_xlabel("Linear Velocity v (m/s)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
    ax1.set_title(f"Distribution: Linear Velocity v\n(Reverse -{V_MAX:.2f}, Stopped 0.0, Forward +{V_MAX:.2f})", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=9)

    # 2. Angular Velocity
    ax2.set_facecolor("#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax2.hist(targets_w * W_MAX, bins=30, alpha=0.55, color="#dc2626", label="ENIB Supervisor Target", zorder=2)
    ax2.hist(preds_w * W_MAX, bins=30, alpha=0.65, color="#16a34a", label="PioneerNN Prediction (Continuous Tanh)", zorder=3)
    ax2.set_xlabel("Angular Velocity ω (rad/s)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
    ax2.set_title(f"Distribution: Angular Velocity ω\n(Clockwise -{W_MAX:.2f}, Straight 0.0, Counter-Clockwise +{W_MAX:.2f})", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Saved] {save_path}")


def main():
    print("\n" + "=" * 68)
    print("  CONFUSION MATRIX & DATASET DIAGNOSTICS GENERATION")
    print("=" * 68 + "\n")

    inputs, targets_v, targets_w, preds_v, preds_w, coords_x, coords_y = load_dataset_and_predict(
        DATASET_CSV, WEIGHTS_PATH
    )

    true_v_cls = discretize_action(targets_v, thresholds=(-0.15, 0.15))
    pred_v_cls = discretize_action(preds_v, thresholds=(-0.15, 0.15))

    true_w_cls = discretize_action(targets_w, thresholds=(-0.15, 0.15))
    pred_w_cls = discretize_action(preds_w, thresholds=(-0.15, 0.15))

    cm_v = compute_confusion_matrix(true_v_cls, pred_v_cls, 3)
    cm_w = compute_confusion_matrix(true_w_cls, pred_w_cls, 3)

    classes_v = [f"Reverse (-{V_MAX:.2f} m/s)", "Stopped (0.0 m/s)", f"Forward (+{V_MAX:.2f} m/s)"]
    classes_w = [f"Clockwise (-{W_MAX:.2f} rad/s)", "Straight (0.0 rad/s)", f"Counter-Clockwise (+{W_MAX:.2f} rad/s)"]

    print("\n4. Generating high-resolution diagnostic plots...")
    path_cm_v = os.path.join(OUT_IMG_DIR, "matriz_confusao_velocidade_linear.png")
    plot_pretty_confusion_matrix(
        cm_v, classes_v,
        "Confusion Matrix — Linear Velocity (v)\nCanonical LIMO Dataset vs PioneerNN",
        path_cm_v, cmap_color="Blues"
    )

    path_cm_w = os.path.join(OUT_IMG_DIR, "matriz_confusao_velocidade_angular.png")
    plot_pretty_confusion_matrix(
        cm_w, classes_w,
        "Confusion Matrix — Angular Velocity (ω)\nCanonical LIMO Dataset vs PioneerNN",
        path_cm_w, cmap_color="Greens"
    )

    path_spatial = os.path.join(OUT_IMG_DIR, "distribuicao_espacial_dataset.png")
    plot_dataset_spatial_distribution(coords_x, coords_y, path_spatial)

    path_hist = os.path.join(OUT_IMG_DIR, "histogramas_predicao_vs_real.png")
    plot_prediction_histograms(targets_v, preds_v, targets_w, preds_w, path_hist)

    acc_v = float(np.trace(cm_v) / np.sum(cm_v))
    acc_w = float(np.trace(cm_w) / np.sum(cm_w))
    metrics_summary = {
        "dataset_total_samples": len(targets_v),
        "linear_velocity_accuracy": acc_v,
        "angular_velocity_accuracy": acc_w,
        "confusion_matrix_linear": cm_v.tolist(),
        "confusion_matrix_angular": cm_w.tolist(),
    }
    json_path = os.path.join(OUT_EVAL_DIR, "confusion_metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  [Saved] {json_path}")

    print("\n" + "=" * 68)
    print(f"  Linear Velocity Classification Accuracy  : {acc_v*100:.2f}%")
    print(f"  Angular Velocity Classification Accuracy : {acc_w*100:.2f}%")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
