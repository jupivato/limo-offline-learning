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

# Auto-detecção de pacotes locais e snap (para PyTorch funcionar diretamente)
for _p in [
    "/home/juliapivato/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
    "/home/juliapivato/snap/antigravity-cli/common/local/lib/python3.12/dist-packages",
    os.path.expanduser("~/local/lib/python3.14/dist-packages"),
    os.path.expanduser("~/local/lib/python3.12/dist-packages"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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


class WeightedMSELoss(nn.Module):
    """Função de custo quadrática média ponderada: Loss = MSE(v) + beta * MSE(w)."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_lin = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
        loss_ang = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
        return loss_lin + self.beta * loss_ang


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

    # 1. Carregar o dataset e separar em Treino (80%) e Validação (20%)
    inputs, targets = load_dataset_samples(dataset_path)
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

    # 2. Inicializar o modelo e otimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PioneerNN(input_size=3, hidden_size=hidden_size, output_size=2, use_bias=use_bias)
    model.to(device)

    criterion = WeightedMSELoss(beta=beta)
    # SGD Puro: momentum=0, weight_decay=0, sem scheduler, taxa 100% constante
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")
    best_weights_json = None
    start_time = time.time()

    # 3. Loop das Épocas
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

        # Salva os melhores pesos encontrados
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights_json = model.save_weights_to_json()

    elapsed = time.time() - start_time
    print(f"\nTreinamento concluído em {elapsed:.2f} s. Melhor perda de validação: {best_val_loss:.6f}")

    # 4. Salvar arquivos finais
    if best_weights_json is None:
        best_weights_json = model.save_weights_to_json()

    os.makedirs(os.path.dirname(os.path.abspath(weights_output_path)), exist_ok=True)
    with open(weights_output_path, "w", encoding="utf-8") as f:
        json.dump(best_weights_json, f, indent=2)
    print(f"[Sucesso] Pesos salvos em : {weights_output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(history_output_path)), exist_ok=True)
    with open(history_output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Sucesso] Histórico salvo em : {history_output_path}")

    # 5. Gerar automaticamente a curva de convergência
    try:
        from src.plot_curves import plot_training_history
        plot_training_history(history_output_path, output_dir="data/evaluation")
        print("[Sucesso] Curvas de treino (PNG e SVG) atualizadas em data/evaluation/")
    except Exception as e:
        print(f"[Aviso] Curvas não foram plotadas automaticamente: {e}")

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "weights_file": weights_output_path
    }


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
