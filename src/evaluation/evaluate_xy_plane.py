"""
Avaliação do Modelo nos Eixos X e Y e no Plano 2D — LIMO
=========================================================
1. Executa baterias de testes em malha fechada partindo estritamente sobre os eixos cartesianos X e Y.
2. Gera o campo vetorial (quiver / vector field) da política de navegação sobre toda a arena 2D.
3. Gera mapas de calor (heatmaps) 2D das velocidades linear e angular.
4. Gera perfis 1D de resposta contínua exatamente em cima do eixo X (y=0) e do eixo Y (x=0).

Gráficos gerados em docs/images/ :
- trajetorias_testes_eixos_x_y.png
- telemetria_testes_eixos_x_y.png
- campo_vetorial_quiver_2d.png
- heatmap_velocidades_2d.png
- perfil_resposta_eixos_x_y.png

CERV / ENIB — Projet LIMO Offline Learning
"""

import os
import sys
import math
import json
from typing import List, Dict, Any

# Auto-detecção de pacotes locais e snap
_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
for _p in [
    f"/home/juliapivato/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages",
    os.path.expanduser(f"~/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/.local/lib/python{_py_ver}/site-packages"),
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

OUT_IMG_DIR = "docs/images"
OUT_EVAL_DIR = "data/evaluation/xy_tests"
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_EVAL_DIR, exist_ok=True)

WEIGHTS_PATH = "models/supervised_w_torch_diff.json"

# ==============================================================================
# ⚙️ CONFIGURAÇÕES GLOBAIS DE DESNORMALIZAÇÃO E LIMITES CINEMÁTICOS (MODIFIQUE AQUI)
# ==============================================================================
V_MAX = 0.50  # Velocidade linear máxima de desnormalização em m/s (dobrada de 0.24 para 0.50)
W_MAX = 1.00  # Velocidade angular máxima de desnormalização em rad/s (dobrada de 0.50 para 1.00)
X_MAX = 5.0   # Limite da arena em X (m)
Y_MAX = 5.0   # Limite da arena em Y (m)
# ==============================================================================


def recast_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


# ==============================================================================
# PARTE 1 : SIMULAÇÃO FECHADA DE TESTES SOBRE OS EIXOS X E Y
# ==============================================================================

AXIAL_SCENARIOS = [
    # EIXO X (y = 0)
    ("Eixo +X Frontal (3m)",   3.0,  0.0,  round(math.pi, 3),   "X+"),
    ("Eixo +X Baliza Ré (2m)",  2.0,  0.0,  0.0,                 "X+"),
    ("Eixo -X Frontal (3m)",  -3.0,  0.0,  0.0,                 "X-"),
    ("Eixo -X Inverso (2m)",  -2.0,  0.0,  round(math.pi, 3),   "X-"),
    # EIXO Y (x = 0)
    ("Eixo +Y Alinhado (3m)",   0.0,  3.0, -round(math.pi/2, 3), "Y+"),
    ("Eixo +Y Lateral (2m)",    0.0,  2.0,  0.0,                 "Y+"),
    ("Eixo -Y Alinhado (3m)",   0.0, -3.0,  round(math.pi/2, 3), "Y-"),
    ("Eixo -Y Lateral (2m)",    0.0, -2.0,  0.0,                 "Y-"),
]


def run_axial_simulation(model: PioneerNN, x0: float, y0: float, th0: float, max_time: float = 40.0):
    dt = 0.050
    steps = int(max_time / dt)
    x, y, th = x0, y0, th0

    t_hist = []
    x_hist = []
    y_hist = []
    th_hist = []
    v_hist = []
    w_hist = []

    for step in range(steps):
        t = step * dt
        dx = 0.0 - x
        dy = 0.0 - y
        dist = math.hypot(dx, dy)
        e_th = recast_angle(th - 0.0)

        xr = dx * math.cos(th) + dy * math.sin(th)
        yr = -dx * math.sin(th) + dy * math.cos(th)

        in_0 = max(-1.0, min(1.0, xr / X_MAX))
        in_1 = max(-1.0, min(1.0, yr / Y_MAX))
        in_2 = max(-1.0, min(1.0, th / math.pi))

        if step > 10 and dist < 0.15 and abs(e_th) < 0.35:
            break

        cmd_raw = model([in_0, in_1, in_2])
        if hasattr(cmd_raw, "tolist"):
            cmd = cmd_raw.tolist()
        else:
            cmd = list(cmd_raw)

        v_cmd = max(-V_MAX, min(V_MAX, float(cmd[0]) * V_MAX))
        w_cmd = max(-W_MAX, min(W_MAX, float(cmd[1]) * W_MAX))

        t_hist.append(t)
        x_hist.append(x)
        y_hist.append(y)
        th_hist.append(th)
        v_hist.append(v_cmd)
        w_hist.append(w_cmd)

        x += v_cmd * math.cos(th) * dt
        y += v_cmd * math.sin(th) * dt
        th = recast_angle(th + w_cmd * dt)

    return {
        "t": t_hist, "x": x_hist, "y": y_hist, "th": th_hist,
        "v": v_hist, "w": w_hist, "final_dist": math.hypot(x, y), "final_e_th": abs(recast_angle(th))
    }


def plot_axial_trajectories(runs_data, save_path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)

    # Destaque dos eixos X e Y
    ax.axhline(0, color="#94a3b8", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1)
    ax.axvline(0, color="#94a3b8", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1)

    colors = ["#2563eb", "#3b82f6", "#dc2626", "#ef4444", "#16a34a", "#22c55e", "#9333ea", "#a855f7"]

    for (name, x0, y0, th0, axis), data, c in zip(AXIAL_SCENARIOS, runs_data, colors):
        ax.plot(data["x"], data["y"], label=f"{name} (Ap={data['final_dist']*100:.1f}cm)",
                color=c, linewidth=2.2, zorder=3)
        ax.scatter([x0], [y0], color=c, marker="o", s=60, zorder=4)

    # Doca Alvo
    ax.scatter([0], [0], color="#dc2626", marker="*", s=220, zorder=5, label="Doca Alvo (0.0, 0.0)")
    c_dock = plt.Circle((0, 0), 0.20, color="#dc2626", fill=False, linestyle="--", linewidth=1.2, zorder=4)
    ax.add_patch(c_dock)

    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Eixo X (metros)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Eixo Y (metros)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Testes de Navegação Estritamente em Cima dos Eixos X e Y\nTrajetórias em Malha Fechada (PioneerNN — SGD Puro)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {save_path}")


def plot_axial_telemetry(runs_data, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=300, sharex=False)
    fig.patch.set_facecolor("#ffffff")

    colors = ["#2563eb", "#3b82f6", "#dc2626", "#ef4444", "#16a34a", "#22c55e", "#9333ea", "#a855f7"]

    # 1. Velocidade Linear
    ax1.set_facecolor("#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    for (name, _, _, _, _), data, c in zip(AXIAL_SCENARIOS, runs_data, colors):
        ax1.plot(data["t"], data["v"], label=name, color=c, linewidth=1.8)
    ax1.axhline(0, color="#475569", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Velocidade Linear v (m/s)", fontsize=10, fontweight="bold")
    ax1.set_title("Comandos Motores durante os Testes nos Eixos X e Y", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", ncol=2, framealpha=0.95, fontsize=8)

    # 2. Velocidade Angular
    ax2.set_facecolor("#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    for (name, _, _, _, _), data, c in zip(AXIAL_SCENARIOS, runs_data, colors):
        ax2.plot(data["t"], data["w"], label=name, color=c, linewidth=1.8)
    ax2.axhline(0, color="#475569", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Tempo (segundos)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Velocidade Angular ω (rad/s)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {save_path}")


# ==============================================================================
# PARTE 2 : MAPEAMENTO 2D DO PLANO (QUIVER / CAMPO VETORIAL E HEATMAPS)
# ==============================================================================

def generate_2d_field_maps(model: PioneerNN):
    """Gera campo vetorial e mapas de calor de velocidade no plano 2D."""
    grid_size = 41
    xs = np.linspace(-4.5, 4.5, grid_size)
    ys = np.linspace(-4.5, 4.5, grid_size)
    X, Y = np.meshgrid(xs, ys)

    # Considera orientação voltada para a doca
    TH = np.arctan2(-Y, -X)

    U = np.zeros_like(X)
    V_dir = np.zeros_like(Y)
    V_lin = np.zeros_like(X)
    W_ang = np.zeros_like(X)

    import torch
    model.eval()

    pts = []
    for i in range(grid_size):
        for j in range(grid_size):
            x_val, y_val, th_val = X[i, j], Y[i, j], TH[i, j]
            # No referencial do robô
            dx = 0.0 - x_val
            dy = 0.0 - y_val
            xr = dx * math.cos(th_val) + dy * math.sin(th_val)
            yr = -dx * math.sin(th_val) + dy * math.cos(th_val)
            in_0 = max(-1.0, min(1.0, xr / X_MAX))
            in_1 = max(-1.0, min(1.0, yr / Y_MAX))
            in_2 = max(-1.0, min(1.0, th_val / math.pi))
            pts.append([in_0, in_1, in_2])

    with torch.no_grad():
        preds = model(torch.tensor(pts, dtype=torch.float32)).numpy()

    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            v_norm = preds[idx, 0]
            w_norm = preds[idx, 1]
            idx += 1

            v_phys = float(v_norm) * V_MAX
            w_phys = float(w_norm) * W_MAX

            V_lin[i, j] = v_phys
            W_ang[i, j] = w_phys

            th_val = TH[i, j]
            U[i, j] = v_phys * math.cos(th_val)
            V_dir[i, j] = v_phys * math.sin(th_val)

    # 1. Campo Vetorial (Quiver Plot)
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)

    # Subamostragem para setas limpas
    skip = 2
    spd = np.hypot(U, V_dir)
    q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V_dir[::skip, ::skip],
                  spd[::skip, ::skip], cmap="plasma", pivot="middle", width=0.0045, headwidth=4, zorder=3)
    cb = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Magnitude da Velocidade (m/s)", fontsize=10, fontweight="bold")

    ax.scatter([0], [0], color="#dc2626", marker="*", s=220, zorder=5, label="Doca Alvo (0.0, 0.0)")
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-4.8, 4.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Posição X (metros)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Posição Y (metros)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Campo Vetorial da Política de Navegação 2D (Quiver Flow)\nDireção e Velocidade Preditas pelo Modelo PioneerNN em Cada Ponto",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

    path_quiver = os.path.join(OUT_IMG_DIR, "campo_vetorial_quiver_2d.png")
    plt.tight_layout()
    plt.savefig(path_quiver, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {path_quiver}")

    # 2. Heatmap de Velocidades (Linear e Angular)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    im1 = ax1.imshow(V_lin, extent=[-4.5, 4.5, -4.5, 4.5], origin="lower", cmap="coolwarm", vmin=-V_MAX, vmax=V_MAX)
    ax1.scatter([0], [0], color="#16a34a", marker="*", s=160, zorder=5, label="Doca")
    ax1.set_title("Heatmap 2D : Velocidade Linear v (m/s)\n(Avanço positivo vs Frenagem na Doca)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Posição X (m)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Posição Y (m)", fontsize=10, fontweight="bold")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("v (m/s)", fontsize=10, fontweight="bold")

    im2 = ax2.imshow(W_ang, extent=[-4.5, 4.5, -4.5, 4.5], origin="lower", cmap="RdYlBu_r", vmin=-W_MAX, vmax=W_MAX)
    ax2.scatter([0], [0], color="#16a34a", marker="*", s=160, zorder=5, label="Doca")
    ax2.set_title("Heatmap 2D : Velocidade Angular ω (rad/s)\n(Simetria Sagital Esquerda / Direita)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Posição X (m)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Posição Y (m)", fontsize=10, fontweight="bold")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("ω (rad/s)", fontsize=10, fontweight="bold")

    path_heatmaps = os.path.join(OUT_IMG_DIR, "heatmap_velocidades_2d.png")
    plt.tight_layout()
    plt.savefig(path_heatmaps, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {path_heatmaps}")


def plot_1d_axis_profiles(model: PioneerNN):
    """Plota cortes 1D contínuos das predições diretamente em cima do eixo X e do eixo Y."""
    pts = np.linspace(-4.5, 4.5, 200)

    # 1. Corte no Eixo X (y = 0, robô virado para o centro)
    v_x_prof = []
    w_x_prof = []
    for x_val in pts:
        th = math.pi if x_val >= 0 else 0.0
        dx = -x_val
        xr = dx * math.cos(th)
        yr = -dx * math.sin(th)
        in_0 = max(-1.0, min(1.0, xr / X_MAX))
        in_1 = max(-1.0, min(1.0, yr / Y_MAX))
        in_2 = max(-1.0, min(1.0, th / math.pi))
        cmd = model([in_0, in_1, in_2])
        v_x_prof.append(float(cmd[0]) * V_MAX)
        w_x_prof.append(float(cmd[1]) * W_MAX)

    # 2. Corte no Eixo Y (x = 0, robô virado para o centro)
    v_y_prof = []
    w_y_prof = []
    for y_val in pts:
        th = -math.pi/2 if y_val >= 0 else math.pi/2
        dy = -y_val
        xr = dy * math.sin(th)
        yr = dy * math.cos(th)
        in_0 = max(-1.0, min(1.0, xr / X_MAX))
        in_1 = max(-1.0, min(1.0, yr / Y_MAX))
        in_2 = max(-1.0, min(1.0, th / math.pi))
        cmd = model([in_0, in_1, in_2])
        v_y_prof.append(float(cmd[0]) * V_MAX)
        w_y_prof.append(float(cmd[1]) * W_MAX)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    ax1.set_facecolor("#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax1.plot(pts, v_x_prof, label="Velocidade Linear v", color="#2563eb", linewidth=2.2)
    ax1.plot(pts, w_x_prof, label="Velocidade Angular ω", color="#16a34a", linewidth=2.0, linestyle="--")
    ax1.axvline(0, color="#dc2626", linestyle=":", label="Doca (X=0)")
    ax1.axhline(0, color="#475569", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Coordenada X (m) [com Y = 0]", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Comando Motor", fontsize=10, fontweight="bold")
    ax1.set_title("Perfil de Resposta em Cima do Eixo X (y=0)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9)

    ax2.set_facecolor("#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax2.plot(pts, v_y_prof, label="Velocidade Linear v", color="#2563eb", linewidth=2.2)
    ax2.plot(pts, w_y_prof, label="Velocidade Angular ω", color="#16a34a", linewidth=2.0, linestyle="--")
    ax2.axvline(0, color="#dc2626", linestyle=":", label="Doca (Y=0)")
    ax2.axhline(0, color="#475569", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Coordenada Y (m) [com X = 0]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Comando Motor", fontsize=10, fontweight="bold")
    ax2.set_title("Perfil de Resposta em Cima do Eixo Y (x=0)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9)

    path_prof = os.path.join(OUT_IMG_DIR, "perfil_resposta_eixos_x_y.png")
    plt.tight_layout()
    plt.savefig(path_prof, dpi=300)
    plt.close(fig)
    print(f"  [Salvo] {path_prof}")


def main():
    print("\n" + "=" * 68)
    print("  AVALIAÇÃO DO MODELO NOS EIXOS X E Y E NO PLANO 2D")
    print("=" * 68 + "\n")

    print("1. Carregando modelo treinado...")
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2, use_bias=False)
    model.load_from_json_file(WEIGHTS_PATH)
    model.eval()

    print("2. Executando simulações fechadas partindo sobre os eixos X e Y...")
    runs_data = []
    for name, x0, y0, th0, axis in AXIAL_SCENARIOS:
        print(f"   -> {name} : partida ({x0}, {y0}, th={th0})")
        res = run_axial_simulation(model, x0, y0, th0)
        runs_data.append(res)
        print(f"      Concluído: Erro final dist = {res['final_dist']*100:.2f} cm")

    print("\n3. Gerando gráficos das trajetórias e telemetria axiais...")
    plot_axial_trajectories(runs_data, os.path.join(OUT_IMG_DIR, "trajetorias_testes_eixos_x_y.png"))
    plot_axial_telemetry(runs_data, os.path.join(OUT_IMG_DIR, "telemetria_testes_eixos_x_y.png"))

    print("\n4. Gerando campo vetorial 2D (Quiver) e mapas de calor...")
    generate_2d_field_maps(model)

    print("\n5. Gerando perfis 1D de resposta contínua sobre os eixos...")
    plot_1d_axis_profiles(model)

    print("\n" + "=" * 68)
    print("  Todos os gráficos gerados com sucesso em docs/images/ !")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
