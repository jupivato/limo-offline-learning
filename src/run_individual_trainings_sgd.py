import os
import sys
import json
import time
import math
import random

sys.path.insert(0, os.path.abspath("."))
_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
snap_paths = [
    f"/home/juliapivato/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages",
    os.path.expanduser(f"~/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/.local/lib/python{_py_ver}/site-packages"),
]
for p in snap_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

from src.train_offline import load_dataset_samples, PioneerNN, WeightedMSELoss

DATASET_CSV = "data/dataset.csv"
OUT_IMG_DIR = "docs/images"
OUT_DATA_DIR = "data/evaluation/grid_study"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DATA_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUT_DATA_DIR, "history"), exist_ok=True)

print("1. Loading dataset samples...")
inputs, targets = load_dataset_samples(DATASET_CSV)
X_tensor = torch.tensor(inputs, dtype=torch.float32)
y_tensor = torch.tensor(targets, dtype=torch.float32)

N = len(X_tensor)
torch.manual_seed(42)
perm = torch.randperm(N)
split_idx = int(0.8 * N)
train_idx, val_idx = perm[:split_idx], perm[split_idx:]

train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val samples.")

LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.5]
CHECKPOINT_EPOCHS = [30, 60, 100, 150]

summary_records = []

def plot_clean_learning_curve(lr, ep_count, epochs_hist, train_loss_hist, val_loss_hist, save_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)

    # Plot lines
    ax.plot(epochs_hist, train_loss_hist, label="Perte d'Entraînement (Train MSE)",
            color="#2563eb", linewidth=2.0, zorder=3)
    ax.plot(epochs_hist, val_loss_hist, label="Perte de Validação (Val MSE)",
            color="#dc2626", linewidth=2.0, linestyle="--", zorder=3)

    # Best val epoch
    min_val = min(val_loss_hist)
    min_epoch = epochs_hist[val_loss_hist.index(min_val)]
    ax.scatter([min_epoch], [min_val], color="#16a34a", s=75, zorder=5,
               label=f"Meilleure Val: {min_val:.5f} (Ép. {min_epoch})")

    ax.set_xlabel("Époque d'Entraînement", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Erreur Quadratique Moyenne (MSE Loss)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(f"SGD Pur à Taux Constant — LR = {lr} — {ep_count} Époques\nModèle PioneerNN (1000 neurones, Tanh) | Sans Momentum, Sans Scheduler",
                 fontsize=12, fontweight="bold", pad=10, color="#0f172a")

    ax.set_xlim(1, ep_count)
    ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Plot Salvo] {save_path}")

print("2. Starting pure constant SGD individual trainings...")

for lr in LEARNING_RATES:
    print(f"\n=======================================================")
    print(f" Treinando com SGD Puro: LR = {lr} (até 150 épocas)...")
    print(f"=======================================================")
    
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Sequential(
        nn.Linear(3, 1000, bias=False),
        nn.Tanh(),
        nn.Linear(1000, 2, bias=False),
        nn.Tanh()
    )
    nn.init.xavier_uniform_(model[0].weight, gain=1.0)
    nn.init.xavier_uniform_(model[2].weight, gain=1.0)
    
    criterion = WeightedMSELoss(beta=1.0)
    # PURE SGD: momentum=0, weight_decay=0, no adaptivity, strictly constant step size
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    
    epochs_hist = []
    train_loss_hist = []
    val_loss_hist = []
    
    t_start = time.time()
    
    for epoch in range(1, 151):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_train_batches += 1
        avg_train_loss = train_loss_sum / n_train_batches

        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss_sum += loss.item()
                n_val_batches += 1
        avg_val_loss = val_loss_sum / n_val_batches
        
        epochs_hist.append(epoch)
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        
        if epoch in CHECKPOINT_EPOCHS:
            elapsed = time.time() - t_start
            print(f"  --> Checkpoint alcançado: LR={lr}, Épocas={epoch} (tempo: {elapsed:.1f}s | Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f})")
            
            # Save weights JSON
            w_dict = {
                "format": "PioneerNN_Weights_V2_SGD",
                "optimizer": "SGD_Pure_Constant",
                "learning_rate": lr,
                "epochs": epoch,
                "hidden_dim": 1000,
                "input_dim": 3,
                "output_dim": 2,
                "W1": model[0].weight.detach().cpu().numpy().tolist(),
                "W2": model[2].weight.detach().cpu().numpy().tolist(),
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss
            }
            m_path = os.path.join(OUT_DATA_DIR, "models", f"model_sgd_lr_{str(lr).replace('.', 'p')}_ep_{epoch}.json")
            with open(m_path, "w") as f:
                json.dump(w_dict, f)
                
            # Save history JSON
            h_dict = {
                "optimizer": "SGD_Pure_Constant",
                "lr": lr,
                "epochs": epochs_hist[:epoch],
                "train_loss": train_loss_hist[:epoch],
                "val_loss": val_loss_hist[:epoch],
                "final_train_loss": avg_train_loss,
                "final_val_loss": avg_val_loss,
                "best_val_loss": min(val_loss_hist[:epoch]),
                "best_epoch": epochs_hist[val_loss_hist[:epoch].index(min(val_loss_hist[:epoch]))]
            }
            h_path = os.path.join(OUT_DATA_DIR, "history", f"history_sgd_lr_{str(lr).replace('.', 'p')}_ep_{epoch}.json")
            with open(h_path, "w") as f:
                json.dump(h_dict, f)
                
            # Plot clean learning curve
            img_filename = f"curva_aprendizado_sgd_lr_{lr}_ep_{epoch}.png"
            img_path = os.path.join(OUT_IMG_DIR, img_filename)
            plot_clean_learning_curve(
                lr=lr,
                ep_count=epoch,
                epochs_hist=epochs_hist[:epoch],
                train_loss_hist=train_loss_hist[:epoch],
                val_loss_hist=val_loss_hist[:epoch],
                save_path=img_path
            )
            
            summary_records.append({
                "optimizer": "SGD_Pure",
                "lr": lr,
                "epochs": epoch,
                "final_train_loss": avg_train_loss,
                "final_val_loss": avg_val_loss,
                "best_val_loss": min(val_loss_hist[:epoch]),
                "best_epoch": epochs_hist[val_loss_hist[:epoch].index(min(val_loss_hist[:epoch]))],
                "plot_image": img_filename
            })

with open(os.path.join(OUT_DATA_DIR, "training_summary_sgd.json"), "w") as f:
    json.dump(summary_records, f, indent=2)

print("\n=======================================================")
print(f" Treinamentos SGD concluídos com sucesso! {len(summary_records)} gráficos limpos gerados em docs/images/")
print("=======================================================")
