"""
Treinamento Offline por Clonagem Comportamental (Behavioral Cloning) — LIMO
==========================================================================
Rede: PioneerNN (3 entradas -> 1000 neurônios ocultos Tanh -> 2 saídas Tanh)
Otimizador: SGD Puro com taxa de aprendizado estritamente constante.

CERV / ENIB — Projet LIMO Offline Learning
"""

import os
import sys
import csv
import json
import time
import math
from typing import Tuple, List, Dict, Any

# Auto-detecção segura de pacotes: APENAS inclui a pasta se a versão do Python for IDÊNTICA
_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
for _p in [
    f"/home/juliapivato/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages",
    os.path.expanduser(f"~/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/.local/lib/python{_py_ver}/site-packages"),
    os.path.expanduser(f"~/local/lib/python{_py_ver}/dist-packages"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Tenta carregar PyTorch; se falhar (ex: ABI diferente ou não instalado), usa NumPy com elegância
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

from src.core.pioneer_nn import PioneerNN

# ==============================================================================
# ⚙️ CONFIGURAÇÕES GLOBAIS DE TREINAMENTO (MODIFIQUE DIRETAMENTE AQUI)
# ==============================================================================
EPOCHS = 150              # Número total de épocas de treinamento
LEARNING_RATE = 0.001     # Taxa de aprendizado constante (SGD Puro, sem scheduler)
BATCH_SIZE = 256          # Tamanho do mini-lote (batch)
BETA = 1.0                # Peso da velocidade angular na perda: MSE(v) + beta * MSE(w)
HIDDEN_SIZE = 1000        # Número de neurônios na camada oculta
USE_BIAS = False          # False = garante velocidade 0.0 na doca (0,0,0) sem tremores

DATASET_CSV = "data/dataset.csv"                         # Arquivo do dataset
OUTPUT_WEIGHTS = "models/supervised_w_torch_diff.json"   # Onde salvar os pesos do modelo
OUTPUT_HISTORY = "models/training_history.json"          # Onde salvar o histórico de treino
# ==============================================================================

# Limites físicos e de normalização do AgileX LIMO
X_MAX = 5.0   # Limite da arena em X (metros)
Y_MAX = 5.0   # Limite da arena em Y (metros)
V_MAX = 0.24  # Velocidade linear máxima de cruzeiro (m/s)
W_MAX = 0.50  # Velocidade angular máxima (rad/s)
ALPHA = [1.0 / X_MAX, 1.0 / Y_MAX, 1.0 / math.pi]


class WeightedMSELoss(nn.Module if TORCH_AVAILABLE else object):
    """Função de custo quadrática média ponderada: Loss = MSE(v) + beta * MSE(w)."""

    def __init__(self, beta: float = 1.0):
        if TORCH_AVAILABLE:
            super().__init__()
        self.beta = beta

    def forward(self, pred: Any, target: Any) -> Any:
        if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
            loss_lin = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
            loss_ang = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
            return loss_lin + self.beta * loss_ang
        diff = pred - target
        return np.mean(diff[:, 0] ** 2) + self.beta * np.mean(diff[:, 1] ** 2)


def load_dataset_samples(csv_file_path: str) -> Tuple[List[List[float]], List[List[float]]]:
    """Carrega o dataset CSV e normaliza entradas e saídas no intervalo [-1.0, +1.0]."""
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
            raise FileNotFoundError(f"Dataset introuvable : {csv_file_path}")

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
        raise ValueError(f"Nenhum dado válido encontrado no arquivo {csv_file_path}")

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
    hidden_size: int
) -> Dict[str, Any]:
    """Treinamento acelerado via PyTorch."""
    print("Moteur de calcul     : PyTorch (SGD Pur a Taxa Constante)")
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(targets, dtype=torch.float32)

    total_samples = len(X)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    torch.manual_seed(42)
    perm = torch.randperm(total_samples)
    train_idx, val_idx = perm[:train_size], perm[train_size:]

    train_loader = DataLoader(TensorDataset(X[train_idx], Y[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val_idx], Y[val_idx]), batch_size=batch_size, shuffle=False)
    print(f"Amostras: {total_samples} no total (Treino: {train_size} | Validação: {val_size})\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PioneerNN(input_size=3, hidden_size=hidden_size, output_size=2, use_bias=use_bias)
    model.to(device)

    criterion = WeightedMSELoss(beta=beta)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0)

    history = {"epochs": [], "train_loss": [], "val_loss": [], "learning_rate": []}
    best_val_loss = float("inf")
    best_weights_json = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        n_train_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = running_train_loss / max(1, n_train_batches)

        # Validação
        model.eval()
        running_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                running_val_loss += criterion(pred, by).item()
                n_val_batches += 1

        avg_val_loss = running_val_loss / max(1, n_val_batches)

        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rate"].append(learning_rate)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Época {epoch:3d}/{epochs} | Treino: {avg_train_loss:.6f} | Validação: {avg_val_loss:.6f} | LR: {learning_rate:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights_json = model.save_weights_to_json()

    elapsed = time.time() - start_time
    print(f"\nTreinamento concluído em {elapsed:.2f} s. Melhor perda de validação: {best_val_loss:.6f}")

    if best_weights_json is None:
        best_weights_json = model.save_weights_to_json()

    os.makedirs(os.path.dirname(os.path.abspath(weights_output_path)), exist_ok=True)
    with open(weights_output_path, "w", encoding="utf-8") as f:
        json.dump(best_weights_json, f, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(history_output_path)), exist_ok=True)
    with open(history_output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"best_val_loss": best_val_loss, "history": history, "weights_file": weights_output_path}


def _train_with_numpy(
    inputs: List[List[float]],
    targets: List[List[float]],
    weights_output_path: str,
    history_output_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    hidden_size: int
) -> Dict[str, Any]:
    """Treinamento de alta velocidade via NumPy (garantido sem erros de ABI C++)."""
    print("Moteur de calcul     : NumPy Vectorisé (SGD Pur a Taxa Constante)")
    X = np.array(inputs, dtype=np.float32)
    Y = np.array(targets, dtype=np.float32)

    total_len = len(X)
    train_len = int(0.8 * total_len)
    np.random.seed(42)
    indices = np.random.permutation(total_len)
    X_train, Y_train = X[indices[:train_len]], Y[indices[:train_len]]
    X_val, Y_val = X[indices[train_len:]], Y[indices[train_len:]]

    print(f"Amostras: {total_len} no total (Treino: {train_len} | Validação: {len(X_val)})\n")

    # Inicialização Xavier Uniforme
    limit1 = math.sqrt(6.0 / (3 + hidden_size))
    W1 = np.random.uniform(-limit1, limit1, (3, hidden_size)).astype(np.float32)
    limit2 = math.sqrt(6.0 / (hidden_size + 2))
    W2 = np.random.uniform(-limit2, limit2, (hidden_size, 2)).astype(np.float32)

    history = {"epochs": [], "train_loss": [], "val_loss": [], "learning_rate": []}
    best_val_loss = float("inf")
    best_W1, best_W2 = W1.copy(), W2.copy()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(X_train))
        X_shuff = X_train[perm]
        Y_shuff = Y_train[perm]

        train_loss_acc = 0.0
        n_batches = 0

        for b in range(0, len(X_train), batch_size):
            xb = X_shuff[b:b + batch_size]
            yb = Y_shuff[b:b + batch_size]
            B = len(xb)

            H = np.tanh(xb @ W1)
            pred = np.tanh(H @ W2)

            diff = pred - yb
            batch_loss = np.mean(diff[:, 0] ** 2) + beta * np.mean(diff[:, 1] ** 2)
            train_loss_acc += batch_loss
            n_batches += 1

            # Backpropagation analítica
            dY = (2.0 / B) * diff
            dY[:, 1] *= beta
            dY = dY * (1.0 - pred ** 2)

            dW2 = H.T @ dY
            dH = (dY @ W2.T) * (1.0 - H ** 2)
            dW1 = xb.T @ dH

            # Atualização estrita SGD Pur
            W1 -= learning_rate * dW1
            W2 -= learning_rate * dW2

        avg_train = train_loss_acc / max(1, n_batches)

        # Validação
        H_val = np.tanh(X_val @ W1)
        pred_val = np.tanh(H_val @ W2)
        diff_val = pred_val - Y_val
        avg_val = float(np.mean(diff_val[:, 0] ** 2) + beta * np.mean(diff_val[:, 1] ** 2))

        history["epochs"].append(epoch)
        history["train_loss"].append(float(avg_train))
        history["val_loss"].append(avg_val)
        history["learning_rate"].append(learning_rate)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Época {epoch:3d}/{epochs} | Treino: {avg_train:.6f} | Validação: {avg_val:.6f} | LR: {learning_rate:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_W1, best_W2 = W1.copy(), W2.copy()

    elapsed = time.time() - start_time
    print(f"\nTreinamento concluído em {elapsed:.2f} s. Melhor perda de validação: {best_val_loss:.6f}")

    weights_json = {
        "input_size": 3,
        "hidden_size": hidden_size,
        "output_size": 2,
        "use_bias": False,
        "input_weights": best_W1.tolist(),
        "output_weights": best_W2.tolist()
    }
    os.makedirs(os.path.dirname(os.path.abspath(weights_output_path)), exist_ok=True)
    with open(weights_output_path, "w", encoding="utf-8") as f:
        json.dump(weights_json, f, indent=2)

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
    beta: float = BETA,
    use_bias: bool = USE_BIAS,
    hidden_size: int = HIDDEN_SIZE,
) -> Dict[str, Any]:
    """
    Executa o treinamento offline completo com SGD Puro a taxa de aprendizado constante.
    """
    print("\n" + "=" * 68)
    print("  TREINAMENTO OFFLINE — PIONEER NN (SGD PURO A TAXA CONSTANTE)")
    print("=" * 68)
    print(f"Dataset              : {dataset_path}")
    print(f"Épocas               : {epochs}")
    print(f"Taxa de Aprendizado  : {learning_rate} (estritamente constante)")
    print(f"Tamanho do Batch     : {batch_size}")
    print(f"Bêta (peso angular)  : {beta}")
    print(f"Neurônios ocultos    : {hidden_size} (Tanh)")
    print(f"Termos de Bias       : {'Ativados' if use_bias else 'Desativados (v=0 estrito na doca)'}")
    print(f"Saída dos Pesos      : {weights_output_path}")
    print(f"Saída do Histórico   : {history_output_path}")
    print("=" * 68 + "\n")

    inputs, targets = load_dataset_samples(dataset_path)

    if TORCH_AVAILABLE:
        result = _train_with_torch(
            inputs, targets, weights_output_path, history_output_path,
            epochs, batch_size, learning_rate, beta, use_bias, hidden_size
        )
    elif NUMPY_AVAILABLE:
        result = _train_with_numpy(
            inputs, targets, weights_output_path, history_output_path,
            epochs, batch_size, learning_rate, beta, hidden_size
        )
    else:
        raise RuntimeError("Nem PyTorch nem NumPy foram encontrados. Execute: pip install numpy torch")

    print(f"[Sucesso] Pesos salvos em : {weights_output_path}")
    print(f"[Sucesso] Histórico salvo em : {history_output_path}")

    # Gerar automaticamente a curva de convergência
    try:
        from src.plot_curves import plot_training_history
        plot_training_history(history_output_path, output_dir="data/evaluation")
        print("[Sucesso] Curvas de treino (PNG e SVG) atualizadas em data/evaluation/")
    except Exception as e:
        print(f"[Aviso] Curvas não foram plotadas automaticamente: {e}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Treinamento Offline do LIMO (PioneerNN com SGD Puro)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Número de épocas (padrão: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Taxa de aprendizado (padrão: {LEARNING_RATE})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Tamanho do batch (padrão: {BATCH_SIZE})")
    parser.add_argument("--dataset", type=str, default=DATASET_CSV, help=f"Arquivo dataset (padrão: {DATASET_CSV})")
    parser.add_argument("--output", type=str, default=OUTPUT_WEIGHTS, help=f"Onde salvar o modelo (padrão: {OUTPUT_WEIGHTS})")
    parser.add_argument("--history", type=str, default=OUTPUT_HISTORY, help=f"Onde salvar o histórico (padrão: {OUTPUT_HISTORY})")
    args = parser.parse_args()

    train_offline(
        dataset_path=args.dataset,
        weights_output_path=args.output,
        history_output_path=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
