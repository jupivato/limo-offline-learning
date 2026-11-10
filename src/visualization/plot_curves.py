"""
Offline Training Convergence Curve Visualizer
=============================================
Visualizes the evolution of the Mean Squared Error (MSE) loss function for
training and validation sets, along with learning rate scheduling dynamics.

Generates two output formats:
- High-resolution PNG image (300 DPI) suitable for academic publications.
- Pure standalone vector SVG for web integration and interactive reports.

Project: LIMO Offline Learning (CERV / ENIB)
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_training_history(
    history_file_path: str = "models/training_history.json",
    output_dir: str = "data/evaluation"
) -> Dict[str, str]:
    """
    Plots and saves offline training convergence curves.

    Args:
        history_file_path (str): Filepath to the training history JSON file.
        output_dir (str): Target directory to save generated plots.

    Returns:
        Dict[str, str]: Generated filepaths (PNG, SVG).
    """
    if not os.path.exists(history_file_path):
        candidates = [
            os.path.join(_project_root, history_file_path),
            os.path.join(_project_root, "models", "training_history.json")
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                history_file_path = c
                found = True
                break
        if not found:
            raise FileNotFoundError(f"History file not found: {history_file_path}")

    if not os.path.isabs(output_dir) and os.getcwd() != _project_root:
        output_dir = os.path.join(_project_root, output_dir)

    with open(history_file_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    lr_history = history.get("learning_rate", history.get("lr", []))
    if not lr_history or len(lr_history) != len(epochs):
        lr_history = [0.001] * len(epochs)

    if not epochs or not train_loss:
        raise ValueError("Training history contains no valid epoch data.")

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "training_curves.png")
    svg_path = os.path.join(output_dir, "training_curves.svg")

    if MATPLOTLIB_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
        fig.patch.set_facecolor("#ffffff")

        # Subplot 1: MSE Loss Curve
        ax1.set_facecolor("#fafbfc")
        ax1.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
        ax1.plot(epochs, train_loss, label="Training Loss (Train MSE)", color="#1f77b4", linewidth=2.0)
        ax1.plot(epochs, val_loss, label="Validation Loss (Val MSE)", color="#d62728", linewidth=2.0, linestyle="--")

        # Marker for best validation epoch
        min_val_idx = val_loss.index(min(val_loss))
        best_epoch = epochs[min_val_idx]
        best_val = val_loss[min_val_idx]
        ax1.scatter([best_epoch], [best_val], color="#2ca02c", s=80, zorder=5,
                    label=f"Best Val: {best_val:.6f} (Epoch {best_epoch})")

        opt_name = history.get("optimizer", "SGD")
        opt_desc = "AdamW" if "ADAMW" in str(opt_name).upper() else "Pure SGD"

        ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight="bold")
        ax1.set_title(f"Supervised Learning Convergence — PioneerNN ({opt_desc})",
                      fontsize=13, fontweight="bold", pad=12)
        ax1.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10)

        # Subplot 2: Learning Rate Schedule
        ax2.set_facecolor("#fafbfc")
        ax2.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
        ax2.plot(epochs, lr_history, label=f"Learning Rate ({opt_desc})", color="#9467bd", linewidth=1.8)
        ax2.set_yscale("log")
        ax2.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Learning Rate (LR)", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10)

        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.savefig(svg_path, format="svg")
        plt.close(fig)
        print(f"[Success] Generated PNG and SVG plots: {png_path} | {svg_path}")

    return {
        "png": png_path,
        "svg": svg_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline training convergence curve visualizer for AgileX LIMO.")
    parser.add_argument("--history", type=str, default="models/training_history.json",
                        help="Path to training history JSON file (default: models/training_history.json)")
    parser.add_argument("--outdir", type=str, default="data/evaluation",
                        help="Output directory for generated plots (default: data/evaluation)")
    args = parser.parse_args()

    plot_training_history(history_file_path=args.history, output_dir=args.outdir)
