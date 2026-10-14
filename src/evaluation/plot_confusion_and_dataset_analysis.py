"""
Geração de Matrizes de Confusão e Diagnósticos do Dataset e Modelo — LIMO
========================================================================
Avalia o modelo PioneerNN treinado sobre as 70.664 amostras do dataset,
calcula as matrizes de confusão para comandos lineares e angulares,
e gera gráficos diagnósticos de alta resolução (300 DPI) para relatórios.

Gráficos gerados:
1. Matriz de Confusão de Velocidade Linear (Ré, Parado, Frente)
2. Matriz de Confusão de Velocidade Angular (Horário, Reto, Anti-horário)
3. Distribuição Espacial 2D das 70.664 amostras coletadas
4. Histogramas Comparativos de Distribuição (Humano Real vs Rede Neural)
5. Mapa de Erro Residual no Plano (X, Y)

CERV / ENIB — Projet LIMO Offline Learning
"""

import os
import sys
import csv
import json
import math
from typing import Tuple, List, Dict, Any

# Auto-detecção de pacotes locais e snap
for _p in [
    "/home/juliapivato/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
    "/home/juliapivato/snap/antigravity-cli/common/local/lib/python3.12/dist-packages",
    os.path.expanduser("~/local/lib/python3.14/dist-packages"),
    os.path.expanduser("~/local/lib/python3.12/dist-packages"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.pioneer_nn import PioneerNN

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

# Parâmetros de normalização do LIMO
X_MAX = 5.0
Y_MAX = 5.0
V_MAX = 0.24
W_MAX = 0.50


def load_dataset_and_predict(dataset_path: str, weights_path: str):
    """Carrega dados, executa inferência e retorna alvos e predições normalizadas."""
    print(f"1. Carregando modelo : {weights_path}")
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2, use_bias=False)
    model.load_from_json_file(weights_path)
    model.eval()

    print(f"2. Carregando dataset : {dataset_path}")
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

    print(f"   Amostras válidas carregadas: {len(inputs):,}")

    print("3. Executando inferência sobre todo o dataset...")
    # Inferência em lote usando os tensores PyTorch do PioneerNN
    import torch
    with torch.no_grad():
        X_tensor = torch.tensor(inputs, dtype=torch.float32)
        preds = model(X_tensor).numpy()

    preds_v = preds[:, 0]
    preds_w = preds[:, 1]

    return np.array(inputs), np.array(targets_v), np.array(targets_w), preds_v, preds_w, np.array(coords_x), np.array(coords_y)


def discretize_action(values, thresholds=(-0.15, 0.15)):
    """Discretiza valores contínuos em 3 classes: 0=Negativo, 1=Neutro/Parado, 2=Positivo."""
    classes = np.zeros_like(values, dtype=int)
    classes[values < thresholds[0]] = 0
    classes[(values >= thresholds[0]) & (values <= thresholds[1])] = 1
    classes[values > thresholds[1]] = 2
    return classes


def compute_confusion_matrix(y_true, y_pred, n_classes=3):
    """Calcula a matriz de confusão n_classes x n_classes."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_pretty_confusion_matrix(cm, class_names, title, save_path, cmap_color="Blues"):
    """Plota matriz de confusão estilizada com valores absolutos e percentuais."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap_color, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proporção Normalizada (Recall)", rotation=-90, va="bottom", fontsize=10, fontweight="bold")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")

    # Adiciona textos com contagem e percentual dentro das células
    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > thresh else "#0f172a"
            text = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    ax.set_ylabel("Classe Real (Humano / Dataset)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Classe Predita (Rede PioneerNN)", fontsize=11, fontweight="bold", labelpad=8)

    # Métricas agregadas
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.01, f"Acurácia Geral das Classes: {accuracy*100:.2f}% | Total: {np.sum(cm):,} amostras",
                 ha="center", fontsize=10, style="italic", color="#475569")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {save_path}")


def plot_dataset_spatial_distribution(x_coords, y_coords, save_path):
    """Gera o mapa 2D com a densidade de amostras coletadas na arena 5x5m."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)

    # Hexbin para densidade
    hb = ax.hexbin(x_coords, y_coords, gridsize=35, cmap="viridis", mincnt=1, zorder=2)
    cb = fig.colorbar(hb, ax=ax, orientation="vertical", pad=0.03)
    cb.set_label("Densidade de Amostras Coletadas", fontsize=10, fontweight="bold")

    # Marca a doca de acostagem (0, 0)
    ax.scatter([0.0], [0.0], color="#dc2626", s=180, marker="*", zorder=5, label="Doca Alvo (0.0, 0.0)")
    circle_dock = plt.Circle((0, 0), 0.20, color="#dc2626", fill=False, linestyle="--", linewidth=1.5, zorder=4, label="Raio de Frenagem (0.20 m)")
    ax.add_patch(circle_dock)

    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Posição Relativa X (metros)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Posição Relativa Y (metros)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(f"Cobertura Espacial do Dataset LIMO — {len(x_coords):,} Amostras\nArena 5m x 5m (Simetria Sagital Multi-Quadrantes)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {save_path}")


def plot_prediction_histograms(targets_v, preds_v, targets_w, preds_w, save_path):
    """Plota histogramas comparativos entre alvos humanos discretos e predições contínuas da rede."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # 1. Velocidade Linear
    ax1.set_facecolor("#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax1.hist(targets_v * V_MAX, bins=30, alpha=0.55, color="#dc2626", label="Alvo Humano Real (Discreto)", zorder=2)
    ax1.hist(preds_v * V_MAX, bins=30, alpha=0.65, color="#2563eb", label="Predição PioneerNN (Contínuo Tanh)", zorder=3)
    ax1.set_xlabel("Velocidade Linear v (m/s)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Contagem de Amostras", fontsize=10, fontweight="bold")
    ax1.set_title("Distribuição: Velocidade Linear v\n(Ré -0.24, Parado 0.0, Frente +0.24)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=9)

    # 2. Velocidade Angular
    ax2.set_facecolor("#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax2.hist(targets_w * W_MAX, bins=30, alpha=0.55, color="#dc2626", label="Alvo Humano Real (Discreto)", zorder=2)
    ax2.hist(preds_w * W_MAX, bins=30, alpha=0.65, color="#16a34a", label="Predição PioneerNN (Contínuo Tanh)", zorder=3)
    ax2.set_xlabel("Velocidade Angular ω (rad/s)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Contagem de Amostras", fontsize=10, fontweight="bold")
    ax2.set_title("Distribuição: Velocidade Angular ω\n(Horário -0.50, Reto 0.0, Anti-horário +0.50)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {save_path}")


def main():
    print("\n" + "=" * 68)
    print("  GERAÇÃO DE MATRIZES DE CONFUSÃO E DIAGNÓSTICOS DO DATASET")
    print("=" * 68 + "\n")

    inputs, targets_v, targets_w, preds_v, preds_w, coords_x, coords_y = load_dataset_and_predict(
        DATASET_CSV, WEIGHTS_PATH
    )

    # Discretização
    true_v_cls = discretize_action(targets_v, thresholds=(-0.15, 0.15))
    pred_v_cls = discretize_action(preds_v, thresholds=(-0.15, 0.15))

    true_w_cls = discretize_action(targets_w, thresholds=(-0.15, 0.15))
    pred_w_cls = discretize_action(preds_w, thresholds=(-0.15, 0.15))

    # Matrizes de Confusão
    cm_v = compute_confusion_matrix(true_v_cls, pred_v_cls, 3)
    cm_w = compute_confusion_matrix(true_w_cls, pred_w_cls, 3)

    classes_v = ["Ré (-0.24 m/s)", "Parado (0.0 m/s)", "Frente (+0.24 m/s)"]
    classes_w = ["Horário (-0.5 rad/s)", "Reto (0.0 rad/s)", "Anti-horário (+0.5 rad/s)"]

    print("\n4. Gerando gráficos em alta resolução...")
    # 1. Matriz Confusão Linear
    path_cm_v = os.path.join(OUT_IMG_DIR, "matriz_confusao_velocidade_linear.png")
    plot_pretty_confusion_matrix(
        cm_v, classes_v,
        "Matriz de Confusão — Velocidade Linear (v)\nDataset V2 LIMO vs PioneerNN (SGD Puro 150 Épocas)",
        path_cm_v, cmap_color="Blues"
    )

    # 2. Matriz Confusão Angular
    path_cm_w = os.path.join(OUT_IMG_DIR, "matriz_confusao_velocidade_angular.png")
    plot_pretty_confusion_matrix(
        cm_w, classes_w,
        "Matriz de Confusão — Velocidade Angular (ω)\nDataset V2 LIMO vs PioneerNN (SGD Puro 150 Épocas)",
        path_cm_w, cmap_color="Greens"
    )

    # 3. Distribuição Espacial 2D do Dataset
    path_spatial = os.path.join(OUT_IMG_DIR, "distribuicao_espacial_dataset.png")
    plot_dataset_spatial_distribution(coords_x, coords_y, path_spatial)

    # 4. Histogramas Comparativos
    path_hist = os.path.join(OUT_IMG_DIR, "histogramas_predicao_vs_real.png")
    plot_prediction_histograms(targets_v, preds_v, targets_w, preds_w, path_hist)

    # Salva relatório de acurácias em JSON
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
    print(f"  [Salvo] {json_path}")

    print("\n" + "=" * 68)
    print(f"  Acurácia de Classificação Linear  : {acc_v*100:.2f}%")
    print(f"  Acurácia de Classificação Angular : {acc_w*100:.2f}%")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
