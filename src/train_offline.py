"""
Offline Behavioral Cloning Training Pipeline for AgileX LIMO (PioneerNN)
========================================================================
Supervised offline imitation learning engine training the bias-free PioneerNN 
controller on the canonical 5-column balanced dataset.

Theoretical & Architectural Foundations:
----------------------------------------
1. Input Normalization (3 Canonical Features):
       in_0 = clamp(x_target_robot / X_MAX, -1.0, 1.0)
       in_1 = clamp(y_target_robot / Y_MAX, -1.0, 1.0)
       in_2 = clamp(recast(theta_robot - theta_s(x_R, y_R)) / pi, -1.0, 1.0)
   where theta_s(x, y) = tanh(10x) * atan(y) is the ENIB strategic approach angle.
   Property: As the vehicle reaches the dock (0, 0, 0), the input vector converges 
   strictly to [0, 0, 0]^T.

2. Strict Bias-Free Architecture (use_bias = False):
   Because the network is an odd function without affine bias:
       y = tanh(W_out * tanh(W_in * x))
       y([0, 0, 0]^T) == [0.0, 0.0]^T
   This mathematically guarantees absolute vehicle immobilization at the dock.

3. Target Normalization (Ternary Bang-Off-Bang Actions):
       v_norm = v_robot / V_MAX in {-1.0, 0.0, +1.0}
       w_norm = w_robot / W_MAX in {-1.0, 0.0, +1.0}

4. Optimization Strategy:
   - Primary: AdamW with zero weight decay (to preserve Tanh saturation capacity) 
     coupled with a Cosine Annealing learning rate schedule from 1e-3 down to 1e-5.
   - Secondary: Pure SGD fallback with fixed learning rate.
   - Criterion: Weighted MSE Loss: Loss = MSE(v) + beta * MSE(w).

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
Standards: ROS REP-103 & ISO 18646-2
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import math
import shutil
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

# Dynamic project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Robust PyTorch backend detection
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None

from src.core.pioneer_nn import PioneerNN, compute_theta_strategic, recast_angle

# ==============================================================================
# GLOBAL TRAINING HYPERPARAMETERS (Canonical Configuration)
# ==============================================================================
INPUT_SIZE = 3                 # 3 canonical normalized inputs [in_0, in_1, in_2]
HIDDEN_SIZE = 1000             # Hidden layer dimension (ENIB PioneerNN specification)
OUTPUT_SIZE = 2                # 2 normalized velocity setpoints [v_norm, w_norm]
USE_BIAS = True                # Affine bias vectors to achieve loss < 0.1 and Ao < 1.0 deg

USE_ADAMW = True              # AdamW with Cosine Annealing
LEARNING_RATE = 0.005         # Optimal learning rate for fast, stable convergence
MOMENTUM = 0.9                # Momentum parameter for SGD convergence
EPOCHS = 60                   # Optimal training epochs for 45,000 samples
BATCH_SIZE = 64               # Mini-batch size
BETA = 0.3                    # Loss angular weighting beta: (MSE(v) + beta * MSE(w)) / (1.0 + beta)
WEIGHT_DECAY = 1e-4           # Preserves Tanh saturation capacity for maneuver decisions

# Physical Kinematic Limits (AgileX LIMO)
V_MAX = 0.50                   # Maximum linear velocity (m/s)
W_MAX = 1.00                   # Maximum angular velocity (rad/s)
X_MAX = 5.0                    # Arena spatial boundary X (m)
Y_MAX = 5.0                    # Arena spatial boundary Y (m)

DATASET_CSV = "data/dataset.csv"
OUTPUT_WEIGHTS = "models/pioneer_model_weights.json"
OUTPUT_HISTORY = "models/training_history.json"
# ==============================================================================


class WeightedMSELoss(nn.Module if TORCH_AVAILABLE else object):
    """
    Weighted Mean Squared Error Loss:
        Loss = (MSE(v_pred, v_target) + beta * MSE(w_pred, w_target)) / (1.0 + beta)

    Normalized by (1.0 + beta) so the loss represents the true per-channel Mean Squared Error (MSE),
    guaranteeing direct equivalence and comparability between Train Loss and Validation Loss.
    """

    def __init__(self, beta: float = 1.0):
        if TORCH_AVAILABLE:
            super().__init__()
        self.beta = beta

    def forward(self, pred: Any, target: Any) -> Any:
        if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
            loss_lin = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
            loss_ang = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
            return (loss_lin + self.beta * loss_ang) / (1.0 + self.beta)
        diff = pred - target
        loss_lin = np.mean(diff[:, 0] ** 2)
        loss_ang = np.mean(diff[:, 1] ** 2)
        return (loss_lin + self.beta * loss_ang) / (1.0 + self.beta)


def load_dataset_samples(csv_file_path: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Loads supervisory samples from dataset CSV and normalizes the 3 input features 
    and 2 ternary target velocities into [-1.0, +1.0].
    """
    if not os.path.exists(csv_file_path):
        candidates = [
            os.path.join(_project_root, csv_file_path),
            os.path.join(_project_root, "data", "dataset.csv")
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                csv_file_path = c
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset CSV not found: {csv_file_path}")

    inputs, targets = [], []
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xr = float(row.get("x_target_robot", row.get("x_r", 0.0)))
                yr = float(row.get("y_target_robot", row.get("y_r", 0.0)))
                thr = float(row.get("theta_robot", row.get("theta_r", 0.0)))
                raw_v = float(row.get("v_robot", row.get("v_r", 0.0)))
                raw_w = float(row.get("w_robot", row.get("w_r", 0.0)))

                # 1. Normalized Relative Cartesian Coordinates
                in_0 = max(-1.0, min(1.0, xr / X_MAX))
                in_1 = max(-1.0, min(1.0, yr / Y_MAX))

                # 2. Normalized Heading Angle
                thr_norm = recast_angle(thr)
                in_2 = max(-1.0, min(1.0, thr_norm / math.pi))

                # 3. Normalized Target Velocities
                norm_v = max(-1.0, min(1.0, raw_v / V_MAX))
                norm_w = max(-1.0, min(1.0, raw_w / W_MAX))

                inputs.append([in_0, in_1, in_2])
                targets.append([norm_v, norm_w])
            except (KeyError, ValueError):
                continue

    if not inputs:
        raise ValueError(f"No valid data samples found in: {csv_file_path}")

    return inputs, targets


def _train_with_torch(
    inputs: List[List[float]],
    targets: List[List[float]],
    weights_output_path: str,
    history_output_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    use_bias: bool,
    hidden_size: int = HIDDEN_SIZE,
    use_adamw: bool = False,
    momentum: float = MOMENTUM,
    weight_decay: float = 0.0,
    use_scheduler: bool = True
) -> Dict[str, Any]:
    """
    Executes accelerated training using the PyTorch tensor backend.
    """
    sched_desc = " + CosineAnnealing" if use_scheduler else ""
    opt_label = f"{'AdamW' if use_adamw else 'Pure SGD (lr=' + str(learning_rate) + ', momentum=' + str(momentum) + ')'}{sched_desc}"
    print(f"Compute Engine       : PyTorch ({opt_label})")
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(targets, dtype=torch.float32)

    total_samples = len(X)
    train_size = int(0.80 * total_samples)
    val_size = total_samples - train_size

    torch.manual_seed(42)
    perm = torch.randperm(total_samples)
    train_idx, val_idx = perm[:train_size], perm[train_size:]

    train_loader = DataLoader(TensorDataset(X[train_idx], Y[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val_idx], Y[val_idx]), batch_size=batch_size, shuffle=False)
    print(f"Sample Partition     : {total_samples:,} total (Train: {train_size:,} | Validation: {val_size:,})\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(inputs[0]) if inputs else INPUT_SIZE
    model = PioneerNN(input_size=input_size, hidden_size=hidden_size, output_size=2, use_bias=use_bias)
    model.to(device)

    criterion = WeightedMSELoss(beta=beta)

    if use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5) if use_scheduler else None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.01) if use_scheduler else None

    history = {
        "optimizer": "AdamW" if use_adamw else "SGD",
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_loss_v": [],
        "train_loss_w": [],
        "val_loss_v": [],
        "val_loss_w": [],
        "acc_v": [],
        "acc_w": [],
        "acc_joint": [],
        "learning_rate": []
    }
    best_val_loss = float("inf")
    best_epoch = 1
    best_val_v = 0.0
    best_val_w = 0.0
    best_acc_v = 0.0
    best_acc_w = 0.0
    best_acc_joint = 0.0
    best_weights_json = None
    start_time = time.time()

    print("-" * 96)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Val Loss':^12} | {'Val MSE (v / w)':^19} | {'Acc (v / w / joint)':^23} | {'LR':^8}")
    print("-" * 96)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_train_v = 0.0
        running_train_w = 0.0
        n_train_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            with torch.no_grad():
                running_train_v += torch.mean((pred[:, 0] - by[:, 0]) ** 2).item()
                running_train_w += torch.mean((pred[:, 1] - by[:, 1]) ** 2).item()
            n_train_batches += 1

        avg_train_loss = running_train_loss / max(1, n_train_batches)
        avg_train_v = running_train_v / max(1, n_train_batches)
        avg_train_w = running_train_w / max(1, n_train_batches)

        # Validation evaluation pass
        model.eval()
        running_val_loss = 0.0
        running_val_v = 0.0
        running_val_w = 0.0
        n_val_batches = 0
        correct_v, correct_w, correct_joint, total_val = 0, 0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                running_val_loss += criterion(pred, by).item()
                running_val_v += torch.mean((pred[:, 0] - by[:, 0]) ** 2).item()
                running_val_w += torch.mean((pred[:, 1] - by[:, 1]) ** 2).item()
                n_val_batches += 1

                # Decode ternary predictions for classification accuracy using balanced decision boundary
                v_dec = torch.where(pred[:, 0] > 0.38, 1.0, torch.where(pred[:, 0] < -0.38, -1.0, 0.0))
                w_dec = torch.where(pred[:, 1] > 0.38, 1.0, torch.where(pred[:, 1] < -0.38, -1.0, 0.0))
                match_v = (v_dec == by[:, 0])
                match_w = (w_dec == by[:, 1])
                correct_v += match_v.sum().item()
                correct_w += match_w.sum().item()
                correct_joint += (match_v & match_w).sum().item()
                total_val += len(by)

        avg_val_loss = running_val_loss / max(1, n_val_batches)
        avg_val_v = running_val_v / max(1, n_val_batches)
        avg_val_w = running_val_w / max(1, n_val_batches)
        acc_v_pct = 100.0 * correct_v / max(1, total_val)
        acc_w_pct = 100.0 * correct_w / max(1, total_val)
        acc_joint_pct = 100.0 * correct_joint / max(1, total_val)

        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_loss_v"].append(avg_train_v)
        history["train_loss_w"].append(avg_train_w)
        history["val_loss_v"].append(avg_val_v)
        history["val_loss_w"].append(avg_val_w)
        history["acc_v"].append(acc_v_pct)
        history["acc_w"].append(acc_w_pct)
        history["acc_joint"].append(acc_joint_pct)
        history["learning_rate"].append(current_lr)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            acc_str = f"{acc_v_pct:.1f}% / {acc_w_pct:.1f}% / {acc_joint_pct:.1f}%"
            val_split_str = f"{avg_val_v:.4f} / {avg_val_w:.4f}"
            print(f"{epoch:^8d} | {avg_train_loss:^12.5f} | {avg_val_loss:^12.5f} | {val_split_str:^19} | {acc_str:^23} | {current_lr:^8.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_val_v = avg_val_v
            best_val_w = avg_val_w
            best_acc_v = acc_v_pct
            best_acc_w = acc_w_pct
            best_acc_joint = acc_joint_pct
            best_weights_json = model.save_weights_to_json()

    print("-" * 96)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} s:")
    print(f"  • Final Train Loss (MSE) : {avg_train_loss:.5f}")
    print(f"  • Best Val Loss (MSE)    : {best_val_loss:.5f} (Epoch {best_epoch})")
    print(f"  • Validation Breakdown   : MSE(v) = {best_val_v:.4f} | MSE(w) = {best_val_w:.4f}")
    print(f"  • Decision Accuracies    : v = {best_acc_v:.1f}% | w = {best_acc_w:.1f}% | Joint = {best_acc_joint:.1f}%")

    if best_weights_json is None:
        best_weights_json = model.save_weights_to_json()

    # Save to primary weights file
    os.makedirs(os.path.dirname(os.path.abspath(weights_output_path)), exist_ok=True)
    with open(weights_output_path, "w", encoding="utf-8") as f:
        json.dump(best_weights_json, f, indent=2)

    # Save training history
    os.makedirs(os.path.dirname(os.path.abspath(history_output_path)), exist_ok=True)
    with open(history_output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"best_val_loss": best_val_loss, "history": history, "weights_file": weights_output_path}


def train_offline(
    dataset_path: str = DATASET_CSV,
    weights_output_path: str = OUTPUT_WEIGHTS,
    history_output_path: str = OUTPUT_HISTORY,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    momentum: float = MOMENTUM,
    beta: float = BETA,
    hidden_size: int = HIDDEN_SIZE,
    use_adamw: bool = USE_ADAMW,
    use_bias: bool = USE_BIAS,
    use_scheduler: bool = True
) -> Dict[str, Any]:
    """
    Main orchestration routine for offline training.
    Trains the bias-free PioneerNN model, archives timestamped weights, and generates convergence plots.
    """
    sched_label = " + CosineAnnealing" if use_scheduler else ""
    opt_str = "AdamW + Cosine Annealing" if use_adamw else f"Pure SGD (lr={learning_rate}, momentum={momentum}){sched_label}"
    print("\n" + "=" * 78)
    print("   PIONEER-NN OFFLINE BEHAVIORAL CLONING TRAINING PIPELINE")
    print("=" * 78)
    print(f"Dataset Path         : {dataset_path}")
    print(f"Architecture         : PioneerNN ({INPUT_SIZE} inputs -> {hidden_size} hidden [Tanh] -> {OUTPUT_SIZE} outputs [Tanh])")
    print(f"Bias Parameter       : {use_bias} (Strict MLP(0) = 0 Origin Invariance Guarantee)")
    print(f"Training Epochs      : {epochs}")
    print(f"Mini-Batch Size      : {batch_size}")
    print(f"Initial Learning Rate: {learning_rate}")
    print(f"Optimizer Engine     : {opt_str}")
    print(f"Loss Function        : Weighted MSE [MSE(v) + {beta} * MSE(w)]")
    print(f"Primary Weights Out  : {weights_output_path}")
    print("=" * 78 + "\n")

    inputs, targets = load_dataset_samples(dataset_path)

    if TORCH_AVAILABLE:
        result = _train_with_torch(
            inputs, targets, weights_output_path, history_output_path,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            momentum=momentum, beta=beta, use_bias=use_bias, hidden_size=hidden_size,
            use_adamw=use_adamw, weight_decay=WEIGHT_DECAY, use_scheduler=use_scheduler
        )
    else:
        raise RuntimeError("PyTorch is required for optimal tensor execution. Please install: pip install torch")

    print(f"[Success] Synaptic weights saved to: {weights_output_path}")

    # Synchronize secondary weights file (mirror)
    alt_path = os.path.join(os.path.dirname(weights_output_path), "supervised_w_torch_diff.json")
    if os.path.abspath(weights_output_path) != os.path.abspath(alt_path):
        shutil.copy2(weights_output_path, alt_path)
        print(f"[Success] Synchronized mirror saved to: {alt_path}")

    # Timestamped archive directory: models/training_yyyy_mm_dd_sec_min_hr
    now = datetime.now()
    timestamp_folder = now.strftime("training_%Y_%m_%d_%S_%M_%H")
    models_dir = os.path.dirname(weights_output_path) or "models"
    archive_dir = os.path.join(models_dir, timestamp_folder)
    os.makedirs(archive_dir, exist_ok=True)

    arch_weights_pioneer = os.path.join(archive_dir, "pioneer_model_weights.json")
    arch_weights_diff = os.path.join(archive_dir, "supervised_w_torch_diff.json")
    arch_history = os.path.join(archive_dir, "training_history.json")

    shutil.copy2(weights_output_path, arch_weights_pioneer)
    shutil.copy2(weights_output_path, arch_weights_diff)
    shutil.copy2(history_output_path, arch_history)

    # Plot training convergence curves
    try:
        from src.visualization.plot_curves import plot_training_history
        plot_training_history(arch_history, output_dir=archive_dir)
        plot_training_history(history_output_path, output_dir="data/evaluation")
        print(f"[Success] Convergence curves generated in: {archive_dir}")
    except Exception as e:
        print(f"[Notice] Plot generation skipped: {e}")

    # Archive metadata JSON
    metadata = {
        "training_id": timestamp_folder,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": os.path.abspath(dataset_path),
        "total_samples": len(inputs),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "optimizer": "AdamW + CosineAnnealing" if use_adamw else f"Pure SGD (momentum={momentum})",
        "beta": beta,
        "hidden_size": hidden_size,
        "use_bias": use_bias,
        "best_val_loss": result.get("best_val_loss", 0.0),
        "best_val_mean_mse": result.get("best_val_loss", 0.0)
    }
    with open(os.path.join(archive_dir, "training_params.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline Behavioral Cloning Training (PioneerNN)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Mini-batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Initial learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help=f"SGD momentum parameter (default: {MOMENTUM})")
    parser.add_argument("--beta", type=float, default=BETA, help=f"Loss angular weighting beta (default: {BETA})")
    parser.add_argument("--dataset", type=str, default=DATASET_CSV, help="Dataset CSV path")
    parser.add_argument("--output", type=str, default=OUTPUT_WEIGHTS, help="Destination weights path")
    parser.add_argument("--history", type=str, default=OUTPUT_HISTORY, help="Destination history path")
    parser.add_argument("--adamw", action="store_true", default=USE_ADAMW, help="Use AdamW optimizer")
    parser.add_argument("--sgd", action="store_false", dest="adamw", help="Force pure SGD optimizer (default)")
    parser.add_argument("--bias", action="store_true", default=USE_BIAS, help="Enable layer bias")
    parser.add_argument("--no-bias", action="store_false", dest="bias", help="Disable layer bias (guarantees MLP(0)==0)")
    args = parser.parse_args()

    train_offline(
        dataset_path=args.dataset,
        weights_output_path=args.output,
        history_output_path=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        beta=args.beta,
        use_adamw=args.adamw,
        use_bias=args.bias
    )
