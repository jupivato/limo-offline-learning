"""
Offline Behavioral Cloning Training Pipeline for AgileX LIMO
=============================================================
Trains the PioneerNN neural network on the robot-centric discrete dataset
using Mean Squared Error (Weighted MSE Loss).

Features:
- Triple training engine:
  1. PyTorch Native (high-performance GPU/CPU tensor engine)
  2. Vectorized NumPy (high-speed fallback if PyTorch has driver/ABI mismatch)
  3. Pure Python standard library (guaranteed zero-dependency fallback)
- Robot-centric normalized inputs in [-1.0, 1.0] and ternary targets in {-1.0, 0.0, 1.0}.
- Pure Constant SGD optimizer (strictly constant learning rate, zero momentum, zero weight decay).
- Default 150 epochs at learning rate 0.001.
- Canonical JSON weight export (PioneerNN) and training history logging.

Project: LIMO Offline Learning (CERV / ENIB)
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import math
import random
import argparse
from typing import Tuple, Dict, Any, List, Optional

# Project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Auto-detect and add snap/local packages
for _p in [
    "/home/juliapivato/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
    "/home/juliapivato/snap/antigravity-cli/common/local/lib/python3.12/dist-packages",
    os.path.expanduser("~/local/lib/python3.14/dist-packages"),
    os.path.expanduser("~/local/lib/python3.12/dist-packages"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_TORCH_IMPORT_ERROR: Optional[str] = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except Exception as _err:
    _TORCH_IMPORT_ERROR = f"{type(_err).__name__}: {_err}"
    TORCH_AVAILABLE = False
    torch = None
    Dataset = object

    class _DummyNN:
        Module = object

    nn = _DummyNN()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None

from src.core.pioneer_nn import PioneerNN

# Robot-centric normalization factors and kinematic limits
X_MAX = 5.0   # Arena X limit (m)
Y_MAX = 5.0   # Arena Y limit (m)
V_MAX = 0.24  # Max linear speed (m/s)
W_MAX = 0.50  # Max angular speed (rad/s)
ALPHA = [1.0 / X_MAX, 1.0 / Y_MAX, 1.0 / math.pi]


def load_dataset_samples(csv_file_path: str) -> Tuple[List[List[float]], List[List[float]]]:
    """Loads and normalizes dataset rows into inputs and targets."""
    if not os.path.exists(csv_file_path):
        candidates = [
            os.path.join(_project_root, csv_file_path),
            os.path.join(_project_root, "data", "dataset.csv"),
            os.path.abspath(csv_file_path)
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                csv_file_path = c
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Jeu de données introuvable : {csv_file_path}")

    inputs: List[List[float]] = []
    targets: List[List[float]] = []

    with open(csv_file_path, "r", encoding="utf-8") as f:
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
                targets.append([norm_v, norm_w])
            except (KeyError, ValueError):
                continue

    if not inputs:
        raise ValueError(f"Aucune donnée valide trouvée dans {csv_file_path}")

    return inputs, targets


if TORCH_AVAILABLE:
    class DemonstrationDataset(Dataset):
        """PyTorch Dataset wrapper for robot-centric samples."""

        def __init__(self, csv_file_path: str):
            inputs, targets = load_dataset_samples(csv_file_path)
            self.X = torch.tensor(inputs, dtype=torch.float32)
            self.y = torch.tensor(targets, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> Tuple[Any, Any]:
            return self.X[idx], self.y[idx]

    class WeightedMSELoss(nn.Module):
        """Weighted MSE Loss: Loss = MSE(v) + beta * MSE(w)."""

        def __init__(self, beta: float = 1.0):
            super().__init__()
            self.beta = beta

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            loss_lin = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
            loss_ang = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
            return loss_lin + self.beta * loss_ang


def train_offline_torch(
    dataset_path: str,
    weights_output_path: str,
    history_output_path: str,
    epochs: int = 150,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    beta: float = 1.0,
    patience: int = 150,
    use_bias: bool = False,
    hidden_size: int = 1000
) -> Dict[str, Any]:
    """Executes offline training using PyTorch with pure constant SGD."""
    print("Moteur de calcul     : PyTorch (Accélération tensorielle)")
    full_dataset = DemonstrationDataset(dataset_path)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    train_subset, val_subset = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f"Échantillons : {total_len} au total (Train: {train_len} | Val: {val_len})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique de calcul : {device}")

    model = PioneerNN(input_size=3, hidden_size=hidden_size, output_size=2, use_bias=use_bias)
    model.to(device)

    criterion = WeightedMSELoss(beta=beta)
    # PURE CONSTANT SGD: zero momentum, zero weight decay, strictly constant learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_weights_json = None
    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = running_train_loss / max(1, train_batches)

        model.eval()
        running_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                running_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = running_val_loss / max(1, val_batches)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rate"].append(current_lr)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Époque {epoch:3d}/{epochs} | Loss Train: {avg_train_loss:.6f} | Loss Val: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_weights_json = model.save_weights_to_json()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Arrêt anticipé à l'époque {epoch} (pas de gain depuis {patience} époques).")
                break

    if best_weights_json is None:
        best_weights_json = model.save_weights_to_json()

    return save_and_finalize_model(
        best_weights_json, history, best_val_loss, start_training_time,
        weights_output_path, history_output_path
    )


def train_offline_numpy(
    dataset_path: str,
    weights_output_path: str,
    history_output_path: str,
    epochs: int = 150,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    beta: float = 1.0,
    patience: int = 150,
    hidden_size: int = 1000
) -> Dict[str, Any]:
    """Executes offline training using vectorized NumPy engine with pure constant SGD."""
    print("Moteur de calcul     : NumPy Vectorisé (SGD Pur à Taux Constant)")
    inputs, targets = load_dataset_samples(dataset_path)

    X = np.array(inputs, dtype=np.float32)
    Y = np.array(targets, dtype=np.float32)

    total_len = len(X)
    indices = np.arange(total_len)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_len = int(0.8 * total_len)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"Échantillons : {total_len} au total (Train: {train_len} | Val: {len(val_idx)})")

    # Xavier initialization for bias-free MLP (3 -> 1000 -> 2)
    limit1 = math.sqrt(6.0 / (3 + hidden_size))
    W1 = np.random.uniform(-limit1, limit1, (3, hidden_size)).astype(np.float32)

    limit2 = math.sqrt(6.0 / (hidden_size + 2))
    W2 = np.random.uniform(-limit2, limit2, (hidden_size, 2)).astype(np.float32)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")
    best_W1, best_W2 = W1.copy(), W2.copy()
    epochs_no_improve = 0
    lr = learning_rate
    start_training_time = time.time()

    num_train = len(X_train)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(num_train)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        train_loss_acc = 0.0
        n_batches = 0

        for b in range(0, num_train, batch_size):
            xb = X_shuffled[b:b + batch_size]
            yb = Y_shuffled[b:b + batch_size]
            B = len(xb)

            # Forward pass
            H = np.tanh(np.dot(xb, W1))
            Y_hat = np.tanh(np.dot(H, W2))

            # Weighted MSE Loss
            diff = Y_hat - yb
            batch_loss = np.mean(diff[:, 0] ** 2) + beta * np.mean(diff[:, 1] ** 2)
            train_loss_acc += batch_loss
            n_batches += 1

            # Backpropagation
            dY = (2.0 / B) * diff
            dY[:, 1] *= beta
            dY = dY * (1.0 - Y_hat ** 2)

            dW2 = np.dot(H.T, dY)
            dH = np.dot(dY, W2.T) * (1.0 - H ** 2)
            dW1 = np.dot(xb.T, dH)

            # Pure SGD update W1 & W2
            W1 -= lr * dW1
            W2 -= lr * dW2

        avg_train_loss = train_loss_acc / max(1, n_batches)

        # Validation pass
        H_val = np.tanh(np.dot(X_val, W1))
        Y_hat_val = np.tanh(np.dot(H_val, W2))
        diff_val = Y_hat_val - Y_val
        avg_val_loss = float(np.mean(diff_val[:, 0] ** 2) + beta * np.mean(diff_val[:, 1] ** 2))

        history["epochs"].append(epoch)
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(avg_val_loss)
        history["learning_rate"].append(lr)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Époque {epoch:3d}/{epochs} | Loss Train: {avg_train_loss:.6f} | Loss Val: {avg_val_loss:.6f} | LR: {lr:.6f}")

        # Check improvement
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_W1 = W1.copy()
            best_W2 = W2.copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Arrêt anticipé à l'époque {epoch} (pas de gain depuis {patience} époques).")
                break

    best_weights_json = {
        "input_size": 3,
        "hidden_size": hidden_size,
        "output_size": 2,
        "use_bias": False,
        "input_weights": best_W1.tolist(),
        "output_weights": best_W2.tolist()
    }

    return save_and_finalize_model(
        best_weights_json, history, best_val_loss, start_training_time,
        weights_output_path, history_output_path
    )


def train_offline_pure_python(
    dataset_path: str,
    weights_output_path: str,
    history_output_path: str,
    epochs: int = 150,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    beta: float = 1.0,
    patience: int = 150,
    hidden_size: int = 1000
) -> Dict[str, Any]:
    """Pure Python training engine requiring zero external libraries."""
    print("Moteur de calcul     : Python Standard (Zéro Dépendance)")
    inputs, targets = load_dataset_samples(dataset_path)

    total_len = len(inputs)
    random.seed(42)
    indices = list(range(total_len))
    random.shuffle(indices)

    train_len = int(0.8 * total_len)
    X_train = [inputs[i] for i in indices[:train_len]]
    Y_train = [targets[i] for i in indices[:train_len]]
    X_val = [inputs[i] for i in indices[train_len:]]
    Y_val = [targets[i] for i in indices[train_len:]]

    print(f"Échantillons : {total_len} au total (Train: {len(X_train)} | Val: {len(X_val)})")

    # Xavier uniform initialization
    lim1 = math.sqrt(6.0 / (3 + hidden_size))
    W1 = [[random.uniform(-lim1, lim1) for _ in range(hidden_size)] for _ in range(3)]

    lim2 = math.sqrt(6.0 / (hidden_size + 2))
    W2 = [[random.uniform(-lim2, lim2) for _ in range(2)] for _ in range(hidden_size)]

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")
    best_W1 = [row[:] for row in W1]
    best_W2 = [row[:] for row in W2]
    start_time = time.time()
    lr = learning_rate

    # Train with sub-epoch sampling for interactive responsiveness in pure Python
    sub_sample_size = min(4000, len(X_train))
    val_sample_size = min(1000, len(X_val))

    for epoch in range(1, epochs + 1):
        sample_indices = random.sample(range(len(X_train)), sub_sample_size)
        loss_acc = 0.0

        for idx in sample_indices:
            x = X_train[idx]
            y = Y_train[idx]

            # Forward
            h = [0.0] * hidden_size
            for j in range(hidden_size):
                h[j] = math.tanh(x[0] * W1[0][j] + x[1] * W1[1][j] + x[2] * W1[2][j])

            pred = [0.0, 0.0]
            for k in range(2):
                acc = sum(h[j] * W2[j][k] for j in range(hidden_size))
                pred[k] = math.tanh(acc)

            # Error
            e0 = pred[0] - y[0]
            e1 = pred[1] - y[1]
            loss_acc += e0 ** 2 + beta * (e1 ** 2)

            # Gradient output
            d_out0 = 2.0 * e0 * (1.0 - pred[0] ** 2)
            d_out1 = 2.0 * beta * e1 * (1.0 - pred[1] ** 2)

            # Update W2 & backprop to hidden
            dh = [0.0] * hidden_size
            for j in range(hidden_size):
                dh[j] = (d_out0 * W2[j][0] + d_out1 * W2[j][1]) * (1.0 - h[j] ** 2)
                W2[j][0] -= lr * d_out0 * h[j]
                W2[j][1] -= lr * d_out1 * h[j]

            # Update W1
            for i in range(3):
                xi = x[i]
                for j in range(hidden_size):
                    W1[i][j] -= lr * dh[j] * xi

        avg_train = loss_acc / sub_sample_size

        # Validation
        val_acc = 0.0
        val_sub = random.sample(range(len(X_val)), val_sample_size)
        for idx in val_sub:
            x = X_val[idx]
            y = Y_val[idx]
            h = [math.tanh(x[0] * W1[0][j] + x[1] * W1[1][j] + x[2] * W1[2][j]) for j in range(hidden_size)]
            p0 = math.tanh(sum(h[j] * W2[j][0] for j in range(hidden_size)))
            p1 = math.tanh(sum(h[j] * W2[j][1] for j in range(hidden_size)))
            val_acc += (p0 - y[0]) ** 2 + beta * (p1 - y[1]) ** 2

        avg_val = val_acc / val_sample_size

        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["learning_rate"].append(lr)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Époque {epoch:3d}/{epochs} | Loss Train: {avg_train:.6f} | Loss Val: {avg_val:.6f} | LR: {lr:.6f}")

        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_W1 = [row[:] for row in W1]
            best_W2 = [row[:] for row in W2]

    best_weights_json = {
        "input_size": 3,
        "hidden_size": hidden_size,
        "output_size": 2,
        "use_bias": False,
        "input_weights": best_W1,
        "output_weights": best_W2
    }

    return save_and_finalize_model(
        best_weights_json, history, best_val_loss, start_time,
        weights_output_path, history_output_path
    )


def save_and_finalize_model(
    best_weights_json: Dict[str, Any],
    history: Dict[str, Any],
    best_val_loss: float,
    start_time: float,
    weights_output_path: str,
    history_output_path: str
) -> Dict[str, Any]:
    """Saves model weights and history across primary and compatibility paths."""
    duration = time.time() - start_time
    print(f"\nEntraînement achevé en {duration:.2f} s. Meilleure perte de validation : {best_val_loss:.6f}")

    # Primary save
    os.makedirs(os.path.dirname(os.path.abspath(weights_output_path)), exist_ok=True)
    with open(weights_output_path, "w", encoding="utf-8") as f:
        json.dump(best_weights_json, f, indent=2)
    print(f"[Succès] Poids sauvegardés dans : {weights_output_path}")

    # History save
    os.makedirs(os.path.dirname(os.path.abspath(history_output_path)), exist_ok=True)
    with open(history_output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Succès] Historique sauvegardé dans : {history_output_path}")

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["epochs"]),
        "duration_seconds": duration,
        "weights_file": weights_output_path
    }


def train_offline(
    dataset_path: str = "data/dataset.csv",
    weights_output_path: str = "models/supervised_w_torch_diff.json",
    history_output_path: str = "models/training_history.json",
    epochs: int = 150,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    beta: float = 1.0,
    patience: int = 150,
    use_bias: bool = False,
    hidden_size: int = 1000
) -> Dict[str, Any]:
    """Entry point for offline behavioral cloning training (Pure Constant SGD)."""
    print("\n" + "=" * 65)
    print("--- DÉMARRAGE DE L'ENTRAÎNEMENT HORS LIGNE (SGD PUR CONSTANT) ---")
    print(f"Jeu de données       : {dataset_path}")
    print(f"Poids de sortie      : {weights_output_path}")
    print(f"Époques maximales    : {epochs}")
    print(f"Taille de batch      : {batch_size}")
    print(f"Optimiseur           : Descente de Gradient Stochastique Pure (SGD)")
    print(f"Taux d'apprentissage : {learning_rate} (Strictement Constant)")
    print(f"Coefficient bêta     : {beta}")
    bias_str = "Activés" if use_bias else "Désactivés (v=0, w=0 strict à l'origine)"
    print(f"Biais neuronaux      : {bias_str}")
    print("=" * 65 + "\n")

    if TORCH_AVAILABLE:
        return train_offline_torch(
            dataset_path=dataset_path,
            weights_output_path=weights_output_path,
            history_output_path=history_output_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            patience=patience,
            use_bias=use_bias,
            hidden_size=hidden_size
        )
    elif NUMPY_AVAILABLE:
        print("[Note] PyTorch non disponible. Basculement vers le moteur vectorisé NumPy.\n")
        return train_offline_numpy(
            dataset_path=dataset_path,
            weights_output_path=weights_output_path,
            history_output_path=history_output_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            patience=patience,
            hidden_size=hidden_size
        )
    else:
        print("[Note] Ni PyTorch ni NumPy détectés. Basculement vers le moteur Python pur autonome.\n")
        return train_offline_pure_python(
            dataset_path=dataset_path,
            weights_output_path=weights_output_path,
            history_output_path=history_output_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            patience=patience,
            hidden_size=hidden_size
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement hors ligne pour LIMO (SGD Pur Constant).")
    parser.add_argument("--dataset", type=str, default="data/dataset.csv", help="Chemin du dataset (défaut: data/dataset.csv)")
    parser.add_argument("--output", type=str, default="models/supervised_w_torch_diff.json", help="Chemin du modèle JSON (défaut: models/supervised_w_torch_diff.json)")
    parser.add_argument("--history", type=str, default="models/training_history.json", help="Chemin du fichier JSON d'historique (défaut: models/training_history.json)")
    parser.add_argument("--epochs", type=int, default=150, help="Nombre d'époques (défaut: 150)")
    parser.add_argument("--batch-size", type=int, default=256, help="Taille de batch (défaut: 256)")
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage constant (défaut: 0.001)")
    parser.add_argument("--beta", type=float, default=1.0, help="Pondération de l'erreur angulaire (défaut: 1.0)")
    parser.add_argument("--use-bias", action="store_true", default=False, help="Active les biais neuronaux (par défaut: False)")
    parser.add_argument("--no-bias", action="store_false", dest="use_bias", help="Désactive les biais neuronaux (par défaut: désactivés)")
    parser.add_argument("--patience", type=int, default=150, help="Seuil d'arrêt précoce en nombre d'époques (défaut: 150)")
    parser.add_argument("--hidden-size", type=int, default=1000, help="Nombre de neurones dans la couche cachée (défaut: 1000)")
    args = parser.parse_args()

    train_offline(
        dataset_path=args.dataset,
        weights_output_path=args.output,
        history_output_path=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        patience=args.patience,
        use_bias=args.use_bias,
        hidden_size=args.hidden_size
    )
