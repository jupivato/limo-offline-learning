import os
import sys
import json
import csv
import math

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

OUT_DIR = "docs/images"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
with open("data/evaluation/grid_study/grid_summary.json", "r") as f:
    grid_data = json.load(f)

# Sort by lr and epochs
grid_data = sorted(grid_data, key=lambda x: (x["lr"], x["epochs"]))

# Group by LR
lrs = [0.001, 0.005, 0.01, 0.02, 0.05]
epochs_list = [30, 60, 100, 150]

lr_labels = {
    0.001: "η = 0.001 (Très Faible)",
    0.005: "η = 0.005 (Modéré)",
    0.01: "η = 0.010 (Équilibré - Record Ap)",
    0.02: "η = 0.020 (Élevé)",
    0.05: "η = 0.050 (Agressif - Record Vitesse)"
}

colors_epoch = {
    30: "#94a3b8",
    60: "#38bdf8",
    100: "#3b82f6",
    150: "#1d4ed8"
}

# -------------------------------------------------------------
# 2. INDIVIDUAL METRIC PLOTS
# -------------------------------------------------------------
print("--> Generating Individual Metric Plots...")

# 2.1 Success Rate (%)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")
ax.grid(True, linestyle="--", alpha=0.5, color="#94a3b8", zorder=0)

x = np.arange(len(lrs))
width = 0.18

for idx, ep in enumerate(epochs_list):
    vals = []
    for lr in lrs:
        row = next(r for r in grid_data if r["lr"] == lr and r["epochs"] == ep)
        vals.append(row["success_rate_pct"])
    bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=f"{ep} époques",
                  color=colors_epoch[ep], edgecolor="#1e293b", linewidth=0.8, zorder=3)
    # Add text labels on bars > 0
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color="#0f172a")

ax.set_xticks(x)
ax.set_xticklabels([f"η = {lr}" for lr in lrs], fontsize=11, fontweight="bold")
ax.set_ylabel("Taux de Succès au Docking (%)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title("Taux de Succès de l'Accostage en Boucle Fermée (Norme ISO 18646-2)\nSeuil : Distance < 0.20 m ET Erreur de Cap < 20.0°",
             fontsize=13, fontweight="bold", pad=12, color="#0f172a")
ax.set_ylim(0, 60)
ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10, loc="upper left")

# Annotation
ax.annotate("Pic de succès à 50% (4/8)\ngrâce à une vitesse franche",
            xy=(4 + 1.5 * width, 50), xytext=(3.1, 52),
            arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.8),
            fontsize=10, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#ef4444", lw=1))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metric_success_rate.png"), dpi=300)
plt.close(fig)

# 2.2 Arrival Time (s)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")
ax.grid(True, linestyle="--", alpha=0.5, color="#94a3b8", zorder=0)

ax.axhline(60.0, color="#ef4444", linestyle="--", linewidth=1.8, zorder=2,
           label="Plafond de Timeout Simulation (60.0 s)")

for idx, ep in enumerate(epochs_list):
    vals = []
    for lr in lrs:
        row = next(r for r in grid_data if r["lr"] == lr and r["epochs"] == ep)
        vals.append(row["mean_succ_time_s"])
    bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=f"{ep} époques",
                  color=colors_epoch[ep], edgecolor="#1e293b", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        if h < 60.0:
            ax.annotate(f"{h:.1f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color="#0f172a")

ax.set_xticks(x)
ax.set_xticklabels([f"η = {lr}" for lr in lrs], fontsize=11, fontweight="bold")
ax.set_ylabel("Temps Moyen d'Accostage (secondes)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title("Temps Moyen d'Accostage selon le Taux d'Apprentissage et les Époques\n(Les essais sans succès complet atteignent le timeout de 60.0 s)",
             fontsize=13, fontweight="bold", pad=12, color="#0f172a")
ax.set_ylim(40, 65)
ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10, loc="lower left")

# Annotation
ax.annotate("Temps le plus court : 46.56 s\n(13.4s de gain sur le timeout)",
            xy=(4 + 1.5 * width, 46.56), xytext=(2.6, 44),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.8),
            fontsize=10, fontweight="bold", color="#047857",
            bbox=dict(boxstyle="round,pad=0.4", fc="#d1fae5", ec="#10b981", lw=1))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metric_arrival_time.png"), dpi=300)
plt.close(fig)

# 2.3 Position Accuracy Ap (m)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")
ax.grid(True, linestyle="--", alpha=0.5, color="#94a3b8", zorder=0)

ax.axhline(0.25, color="#eab308", linestyle="--", linewidth=1.8, zorder=2,
           label="Tolérance Maximale ISO 18646-2 (Ap < 0.25 m)")

for idx, ep in enumerate(epochs_list):
    vals = []
    for lr in lrs:
        row = next(r for r in grid_data if r["lr"] == lr and r["epochs"] == ep)
        vals.append(row["Ap_m"])
    bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=f"{ep} époques",
                  color=colors_epoch[ep], edgecolor="#1e293b", linewidth=0.8, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([f"η = {lr}" for lr in lrs], fontsize=11, fontweight="bold")
ax.set_ylabel("Précision Barycentrique ISO Ap (mètres)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title("Précision de Position ISO Ap selon le Taux d'Apprentissage et les Époques\nDistance du barycentre des arrivées à la cible (0,0)",
             fontsize=13, fontweight="bold", pad=12, color="#0f172a")
ax.set_ylim(0, 0.40)
ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10, loc="upper right")

# Annotation Record
ax.annotate("RECORD ABSOLU : Ap = 0.0242 m (2.4 cm)\nBarycentre ultra-précis (η=0.01, 150 ép.)",
            xy=(2 + 1.5 * width, 0.0242), xytext=(0.5, 0.32),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=2.0),
            fontsize=10, fontweight="bold", color="#047857",
            bbox=dict(boxstyle="round,pad=0.4", fc="#d1fae5", ec="#10b981", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metric_position_accuracy.png"), dpi=300)
plt.close(fig)

# 2.4 True Heading Error |Δθ| (deg)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")
ax.grid(True, linestyle="--", alpha=0.5, color="#94a3b8", zorder=0)

ax.axhline(20.0, color="#10b981", linestyle="--", linewidth=1.8, zorder=2,
           label="Seuil d'Orientation ISO Cible (|Δθ| ≤ 20.0°)")

for idx, ep in enumerate(epochs_list):
    vals = []
    for lr in lrs:
        row = next(r for r in grid_data if r["lr"] == lr and r["epochs"] == ep)
        vals.append(row["mean_abs_angle_deg"])
    bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=f"{ep} époques",
                  color=colors_epoch[ep], edgecolor="#1e293b", linewidth=0.8, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([f"η = {lr}" for lr in lrs], fontsize=11, fontweight="bold")
ax.set_ylabel("Erreur Réelle Absolue Moyenne |Δθ| (degrés)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title("Erreur d'Orientation Angulaire Réelle Terminale |Δθ| Moyenne\n(Mesure absolue éliminant le piège d'annulation algébrique de l'Ao signé)",
             fontsize=13, fontweight="bold", pad=12, color="#0f172a")
ax.set_ylim(0, 130)
ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10, loc="upper right")

# Annotation
ax.annotate("Meilleur cap d'accostage : 56.4°\n(η=0.05, 150 ép.) — Alignement supérieur",
            xy=(4 + 1.5 * width, 56.4), xytext=(2.1, 28),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.8),
            fontsize=10, fontweight="bold", color="#1e40af",
            bbox=dict(boxstyle="round,pad=0.4", fc="#dbeafe", ec="#3b82f6", lw=1))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metric_heading_error.png"), dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# 3. INDIVIDUAL LOSS CURVES FOR EACH LEARNING RATE
# -------------------------------------------------------------
print("--> Generating Individual Loss Curves...")

hist_dir = "data/evaluation/grid_study/history"

for lr in lrs:
    hpath = os.path.join(hist_dir, f"history_lr_{str(lr).replace('.', 'p')}_ep_150.json")
    if not os.path.exists(hpath):
        continue
    with open(hpath) as f:
        h = json.load(f)
    
    eps = h["epochs"]
    tloss = h["train_loss"]
    vloss = h["val_loss"]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    
    ax.plot(eps, tloss, label="Perte d'Entraînement (Train MSE)", color="#2563eb", linewidth=2.0)
    ax.plot(eps, vloss, label="Perte de Validation (Val MSE)", color="#dc2626", linewidth=2.0, linestyle="--")
    
    min_v = min(vloss)
    min_ep = eps[vloss.index(min_v)]
    ax.scatter([min_ep], [min_v], color="#16a34a", s=90, zorder=5,
               label=f"Min Val: {min_v:.5f} (Ép. {min_ep})")
    
    ax.set_xlabel("Époque d'Entraînement", fontsize=11, fontweight="bold")
    ax.set_ylabel("Erreur Quadratique Moyenne (Loss MSE)", fontsize=11, fontweight="bold")
    ax.set_title(f"Dynamique d'Apprentissage — {lr_labels[lr]}\n(Dataset: 70 664 échantillons, AdamW, Batch 1024, Seed 42)",
                 fontsize=12, fontweight="bold", pad=12, color="#0f172a")
    ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10, loc="upper right")
    
    # Specific annotations per LR
    if lr == 0.001:
        ax.text(0.05, 0.15, "Sous-apprentissage : descente très lente,\npas d'instabilité mais puissance motrice bridée.",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f1f5f9", ec="#94a3b8"))
    elif lr == 0.01:
        ax.text(0.05, 0.15, "Convergence optimale : courbe parfaitement fluide,\nzéro pic de gradient, précision géométrique maximale.",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="#dcfce7", ec="#22c55e"))
    elif lr == 0.05:
        # annotate gradient spikes
        spike_eps = [e for e, l in zip(eps, vloss) if l > 0.5]
        if spike_eps:
            ax.annotate("Pics de gradient (Ép. 45 & 128)\nExplosion temporaire résorbée par AdamW",
                        xy=(spike_eps[0], vloss[eps.index(spike_eps[0])]), xytext=(spike_eps[0]+10, 1.2),
                        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.8),
                        fontsize=10, fontweight="bold", color="#b91c1c",
                        bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#ef4444"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"loss_curve_lr_{str(lr).replace('.', 'p')}.png"), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------
# 4. FACETED GRID LOSS CURVES WITH INTEGRATED SUMMARY TABLE
# -------------------------------------------------------------
print("--> Generating Faceted Grid Loss Curves & Table...")

fig = plt.figure(figsize=(15, 11), dpi=300)
fig.patch.set_facecolor("#ffffff")
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, 0])
]
ax_table = fig.add_subplot(gs[2, 1])

table_rows = []

for i, lr in enumerate(lrs):
    ax = axes[i]
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    
    hpath = os.path.join(hist_dir, f"history_lr_{str(lr).replace('.', 'p')}_ep_150.json")
    with open(hpath) as f:
        h = json.load(f)
    
    eps = h["epochs"]
    tloss = h["train_loss"]
    vloss = h["val_loss"]
    min_v = min(vloss)
    min_ep = eps[vloss.index(min_v)]
    final_v = vloss[-1]
    
    stability = "Fluide" if lr <= 0.01 else ("Légère oscill." if lr == 0.02 else "Sauts gradient")
    table_rows.append([f"η = {lr}", f"{min_v:.4f} (ép. {min_ep})", f"{final_v:.4f}", stability])
    
    ax.plot(eps, tloss, color="#2563eb", linewidth=1.8, label="Train MSE")
    ax.plot(eps, vloss, color="#dc2626", linewidth=1.8, linestyle="--", label="Val MSE")
    ax.scatter([min_ep], [min_v], color="#16a34a", s=60, zorder=5)
    
    ax.set_title(f"({chr(65+i)}) {lr_labels[lr]}", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Époque", fontsize=9, fontweight="bold")
    ax.set_ylabel("Perte MSE", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

# Table panel
ax_table.axis("off")
col_labels = ["Taux (η)", "Meilleure Val (Ép.)", "Val Finale (150)", "Stabilité Numérique"]
tbl = ax_table.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.9)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#0f172a")
        cell.set_text_props(color="#ffffff", fontweight="bold")
    else:
        if r % 2 == 1:
            cell.set_facecolor("#f1f5f9")
        else:
            cell.set_facecolor("#ffffff")
        if r == 3: # eta=0.01
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(fontweight="bold")
        elif r == 5: # eta=0.05
            cell.set_facecolor("#fee2e2")

ax_table.set_title("(F) Tableau Récapitulatif de Convergence", fontsize=11, fontweight="bold", pad=12)

fig.suptitle("Convergence de l'Apprentissage par Sous-Graphique Individuel (150 Époques, Sans Chevauchement)",
             fontsize=14, fontweight="bold", y=0.98, color="#0f172a")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, "grid_loss_faceted.png"), dpi=300)
# Overwrite grid_loss_curves.png as well so markdown links automatically point to this clean version
plt.savefig(os.path.join(OUT_DIR, "grid_loss_curves.png"), dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# 5. KINEMATIC & DOCKING TIME ANALYSIS (THE USER'S PHYSICAL QUERY)
# -------------------------------------------------------------
print("--> Generating Kinematic Docking Time Analysis Plots...")

# Let's extract telemetry for representative runs:
# Compare lr_0p001_ep_150 (run 1), lr_0p01_ep_150 (run 6), lr_0p05_ep_150 (run 2)
def load_traj(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    ts = [float(x['t']) for x in r]
    xs = [float(x['x']) for x in r]
    ys = [float(x['y']) for x in r]
    ths = [float(x['th']) for x in r]
    dists = [float(x['dist']) for x in r]
    eths = [float(x['e_theta']) for x in r]
    
    # linear velocities
    vs = [0.0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        dl = math.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
        vs.append(dl/dt if dt > 0 else 0.0)
    return {
        "t": ts, "x": xs, "y": ys, "th": ths, "dist": dists, "eth": eths, "v": vs
    }

traj_001 = load_traj("data/evaluation/grid_study/trajectories/lr_0p001_ep_150/run_1_q1.csv")
traj_01 = load_traj("data/evaluation/grid_study/trajectories/lr_0p01_ep_150/run_6_q3.csv")
traj_05 = load_traj("data/evaluation/grid_study/trajectories/lr_0p05_ep_150/run_2_q1.csv")

# 5.1 Distance vs Time d(t)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")
ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

ax.axhline(0.50, color="#f59e0b", linestyle=":", linewidth=1.5, label="Zone d'approche terminale (d = 0.50 m)")
ax.axhline(0.20, color="#10b981", linestyle="--", linewidth=1.8, label="Seuil ISO de position validée (d < 0.20 m)")

ax.plot(traj_05["t"], traj_05["dist"], color="#dc2626", linewidth=2.4,
        label="η = 0.050 (Run 2) — Atteint 0.5m en 10.8s (Transit Rapide)")
ax.plot(traj_01["t"], traj_01["dist"], color="#16a34a", linewidth=2.4,
        label="η = 0.010 (Run 6) — Atteint 0.5m en 41.0s (Vitesse Lente)")
ax.plot(traj_001["t"], traj_001["dist"], color="#2563eb", linewidth=2.4,
        label="η = 0.001 (Run 1) — Atteint 0.5m en 43.4s (Vitesse Lente)")

ax.axvline(10.8, color="#dc2626", linestyle=":", linewidth=1.2)
ax.axvline(32.1, color="#dc2626", linestyle="--", linewidth=1.5)

ax.annotate("Fin du transit rectiligne (t = 10.8 s)\nDistance < 0.5 m atteinte !",
            xy=(10.8, 0.49), xytext=(14, 1.2),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8),
            fontsize=9.5, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#ef4444"))

ax.annotate("Accostage complet validé (t = 32.1 s)\n(21.3 s passées à aligner l'orientation)",
            xy=(32.1, 0.198), xytext=(35, 0.8),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8),
            fontsize=9.5, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#ef4444"))

ax.set_xlabel("Temps Écoulé (secondes)", fontsize=11, fontweight="bold")
ax.set_ylabel("Distance Euclidienne à la Cible (mètres)", fontsize=11, fontweight="bold")
ax.set_title("Évolution Temporelle de la Distance à la Cible d(t) : Décomposition en 2 Phases\nPhase 1 (Transit Rapide vs Lent) et Phase 2 (Amarrage Terminal)",
             fontsize=12, fontweight="bold", pad=12, color="#0f172a")
ax.set_xlim(0, 65)
ax.set_ylim(0, 3.2)
ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kinematic_distance_evolution.png"), dpi=300)
plt.close(fig)

# 5.2 Linear Velocity vs Time v(t)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")
ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

ax.axhline(0.24, color="#0f172a", linestyle="--", linewidth=1.5,
           label="Vitesse Maximale Matérielle AgileX LIMO (Vmax = 0.24 m/s)")

# Moving average filter for display smoothness
def smooth(y, box_pts=7):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

ax.plot(traj_05["t"], smooth(traj_05["v"]), color="#dc2626", linewidth=2.2,
        label="η = 0.050 — Exploite 100% de Vmax (v ≈ 0.24 m/s en transit)")
ax.plot(traj_01["t"], smooth(traj_01["v"]), color="#16a34a", linewidth=2.2,
        label="η = 0.010 — Vitesse bridée (v ≈ 0.05 m/s) -> 2.8m / 0.05m/s = 56s !")
ax.plot(traj_001["t"], smooth(traj_001["v"]), color="#2563eb", linewidth=2.2,
        label="η = 0.001 — Vitesse bridée (v ≈ 0.047 m/s)")

ax.set_xlabel("Temps Écoulé (secondes)", fontsize=11, fontweight="bold")
ax.set_ylabel("Vitesse Linéaire Réelle v(t) (m/s)", fontsize=11, fontweight="bold")
ax.set_title("Profil de Vitesse Linéaire v(t) : Preuve Mathématique de la Vitesse Bridée\nPourquoi les modèles à faible taux mettent 50 à 60s pour parcourir 2.8 mètres",
             fontsize=12, fontweight="bold", pad=12, color="#0f172a")
ax.set_xlim(0, 65)
ax.set_ylim(0, 0.30)
ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5, loc="upper right")

ax.text(0.04, 0.45,
        "Démonstration Cinématique :\n"
        "• À v = 0.05 m/s, franchir d = 2.83 m exige t = 2.83 / 0.05 = 56.6 secondes !\n"
        "• À v = 0.24 m/s (η=0.05), le robot couvre 2.83 m en t = 11.8 secondes,\n"
        "  puis consacre 20 secondes supplémentaires aux manœuvres de rotation finale.",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#94a3b8", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kinematic_velocity_profile.png"), dpi=300)
plt.close(fig)

# 5.3 Orientation Error vs Time e_theta(t)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f8fafc")
ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

ax.axhline(20.0, color="#10b981", linestyle="--", linewidth=1.8,
           label="Seuil d'Orientation ISO Validé (|e_theta| ≤ 20.0°)")

deg_05 = [abs(e) * 180 / math.pi for e in traj_05["eth"]]
deg_01 = [abs(e) * 180 / math.pi for e in traj_01["eth"]]
deg_001 = [abs(e) * 180 / math.pi for e in traj_001["eth"]]

ax.plot(traj_05["t"], deg_05, color="#dc2626", linewidth=2.2,
        label="η = 0.050 (Run 2) — Passe de 113° à 19.9° à t = 32.1s (Succès)")
ax.plot(traj_01["t"], deg_01, color="#16a34a", linewidth=2.2,
        label="η = 0.010 (Run 6) — Passe sous 20° à t = 56.9s (Succès)")
ax.plot(traj_001["t"], deg_001, color="#2563eb", linewidth=2.2,
        label="η = 0.001 (Run 1) — Bloqué à 131° (Orientation orthogonale)")

ax.set_xlabel("Temps Écoulé (secondes)", fontsize=11, fontweight="bold")
ax.set_ylabel("Erreur d'Orientation Absolue |e_theta| (degrés)", fontsize=11, fontweight="bold")
ax.set_title("Évolution Temporelle de l'Erreur de Cap |e_theta(t)| : La Phase de Rotation\nExpliquant pourquoi l'accostage complet prend du temps même après arrivée sur place",
             fontsize=12, fontweight="bold", pad=12, color="#0f172a")
ax.set_xlim(0, 65)
ax.set_ylim(0, 180)
ax.legend(framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kinematic_heading_evolution.png"), dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# 6. INDIVIDUAL TRAJECTORY PLOTS WITH HEADING ARROWS
# -------------------------------------------------------------
print("--> Generating Individual 2D Trajectory Plots...")

import glob

def plot_single_model_trajectory(model_name, title_str, save_name, note_text):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    
    # Target station at (0,0)
    circle_iso = plt.Circle((0, 0), 0.20, color="#10b981", alpha=0.2, label="Tolérance ISO (0.20 m)")
    ax.add_patch(circle_iso)
    circle_iso_bound = plt.Circle((0, 0), 0.20, color="#059669", fill=False, linestyle="--", linewidth=1.5)
    ax.add_patch(circle_iso_bound)
    ax.scatter([0], [0], color="#059669", s=150, marker="*", zorder=10, label="Cible Cerveau (0,0)")
    
    # Target heading arrow: pointing to (1, 0)
    ax.arrow(0, 0, 0.40, 0, head_width=0.12, head_length=0.12, fc="#059669", ec="#047857",
             linewidth=2.5, zorder=12, label="Cap Cible Désiré (θ* = 0°)")
    
    traj_files = sorted(glob.glob(f"data/evaluation/grid_study/trajectories/{model_name}/run_*.csv"))
    colors = plt.cm.tab10(np.linspace(0, 1, len(traj_files)))
    
    for i, tf in enumerate(traj_files):
        with open(tf) as f:
            r = list(csv.DictReader(f))
        xs = [float(x['x']) for x in r]
        ys = [float(x['y']) for x in r]
        ths = [float(x['th']) for x in r]
        
        c = colors[i]
        label = f"Essai {i+1} ({os.path.basename(tf).split('.')[0]})"
        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.85, label=label)
        
        # Start marker and arrow
        ax.scatter([xs[0]], [ys[0]], color=c, s=50, edgecolors="#0f172a", zorder=6)
        arrow_len = 0.25
        ax.arrow(xs[0], ys[0], arrow_len*math.cos(ths[0]), arrow_len*math.sin(ths[0]),
                 head_width=0.08, head_length=0.08, fc=c, ec="#0f172a", linewidth=1.0, zorder=7)
        
        # End marker and arrow
        ax.scatter([xs[-1]], [ys[-1]], color=c, s=70, marker="s", edgecolors="#0f172a", zorder=8)
        arrow_len_end = 0.32
        ax.arrow(xs[-1], ys[-1], arrow_len_end*math.cos(ths[-1]), arrow_len_end*math.sin(ths[-1]),
                 head_width=0.10, head_length=0.10, fc="#ef4444", ec="#991b1b", linewidth=1.5, zorder=9)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Coordonnée X (mètres)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Coordonnée Y (mètres)", fontsize=11, fontweight="bold")
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=12, color="#0f172a")
    ax.legend(framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8, loc="upper right")
    
    # Custom note box
    ax.text(0.03, 0.03, note_text, transform=ax.transAxes, fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec="#94a3b8", lw=1.2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, save_name), dpi=300)
    plt.close(fig)

# Trajectory 1: lr_0.001
plot_single_model_trajectory(
    "lr_0p001_ep_150",
    "Trajectoires 2D et Vecteurs d'Orientation — η = 0.001 (150 Époques)\nFlèches rouges : Orientation finale du robot à l'arrêt",
    "trajectory_lr_0p001.png",
    "Observation Cinématique (η = 0.001) :\n"
    "• Vitesse linéaire bridée à 0.047 m/s (sous-apprentissage).\n"
    "• Trajectoires quasi-rectilignes mais arrêt à 0.18-0.25 m par timeout.\n"
    "• Flèches rouges : Les robots restent orientés orthogonalement (~84°)."
)

# Trajectory 2: lr_0.01
plot_single_model_trajectory(
    "lr_0p01_ep_150",
    "Trajectoires 2D et Vecteurs d'Orientation — η = 0.010 (150 Époques)\nRECORD DE PRÉCISION GÉOMÉTRIQUE : Ap = 2.4 cm de la cible",
    "trajectory_lr_0p01.png",
    "Observation Cinématique (η = 0.010) :\n"
    "• Barycentre d'arrivée record : Ap = 0.0242 m (2.4 cm de la cible) !\n"
    "• Convergence spatiale parfaite vers l'étoile verte centrale.\n"
    "• 2/8 essais valident le cap en < 60s ; les autres manquent de temps pour finir la rotation."
)

# Trajectory 3: lr_0.05
plot_single_model_trajectory(
    "lr_0p05_ep_150",
    "Trajectoires 2D et Vecteurs d'Orientation — η = 0.050 (150 Époques)\nRECORD DE TAUX DE SUCCÈS : 50% (4/8) et Arrivée Rapide (46.6s)",
    "trajectory_lr_0p05.png",
    "Observation Cinématique (η = 0.050) :\n"
    "• Vitesse nominale maximale (0.24 m/s) atteinte dès le départ.\n"
    "• 4 essais sur 8 concluent l'amarrage complet (position < 0.2m ET cap < 20°).\n"
    "• Arrivée en zone cible en seulement 10.8s, suivie de manœuvres d'orientation."
)

# -------------------------------------------------------------
# 7. CLEAN 3-PANEL SIDE-BY-SIDE TRAJECTORY COMPARISON
# -------------------------------------------------------------
print("--> Generating Side-by-Side 3-Panel Trajectory Comparison...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.2), dpi=300)
fig.patch.set_facecolor("#ffffff")

panel_cfgs = [
    (ax1, "lr_0p001_ep_150", "(A) η = 0.001 (150 ép.) — Sous-apprentissage", "#2563eb"),
    (ax2, "lr_0p01_ep_150", "(B) η = 0.010 (150 ép.) — Record Ap (2.4 cm)", "#16a34a"),
    (ax3, "lr_0p05_ep_150", "(C) η = 0.050 (150 ép.) — Record Succès (50%)", "#dc2626")
]

for ax, mname, title, col in panel_cfgs:
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    
    circle_iso = plt.Circle((0, 0), 0.20, color="#10b981", alpha=0.2)
    ax.add_patch(circle_iso)
    circle_bound = plt.Circle((0, 0), 0.20, color="#059669", fill=False, linestyle="--", linewidth=1.5)
    ax.add_patch(circle_bound)
    ax.scatter([0], [0], color="#059669", s=130, marker="*", zorder=10)
    ax.arrow(0, 0, 0.40, 0, head_width=0.12, head_length=0.12, fc="#059669", ec="#047857",
             linewidth=2.2, zorder=12)
    
    tfiles = sorted(glob.glob(f"data/evaluation/grid_study/trajectories/{mname}/run_*.csv"))
    c_list = plt.cm.tab10(np.linspace(0, 1, len(tfiles)))
    
    for i, tf in enumerate(tfiles):
        with open(tf) as f:
            r = list(csv.DictReader(f))
        xs = [float(x['x']) for x in r]
        ys = [float(x['y']) for x in r]
        ths = [float(x['th']) for x in r]
        
        ax.plot(xs, ys, color=c_list[i], linewidth=1.8, alpha=0.85)
        ax.scatter([xs[0]], [ys[0]], color=c_list[i], s=35, edgecolors="#0f172a", zorder=6)
        ax.scatter([xs[-1]], [ys[-1]], color=c_list[i], s=55, marker="s", edgecolors="#0f172a", zorder=8)
        
        # End orientation arrow
        ax.arrow(xs[-1], ys[-1], 0.30*math.cos(ths[-1]), 0.30*math.sin(ths[-1]),
                 head_width=0.09, head_length=0.09, fc="#ef4444", ec="#991b1b", linewidth=1.3, zorder=9)
        
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#0f172a")

fig.suptitle("Comparaison Impartiale des Trajectoires 2D et Caps Terminaux (Norme ISO 18646-2)\nÉtoile verte : Cible (0,0) | Cercle vert : Tolérance 0.20m | Flèches rouges : Cap terminal du robot",
             fontsize=13, fontweight="bold", y=0.98, color="#0f172a")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, "trajectory_comparison_key_models.png"), dpi=300)
# Overwrite lr_trajectories_comparison.png as well
plt.savefig(os.path.join(OUT_DIR, "lr_trajectories_comparison.png"), dpi=300)
plt.close(fig)

print("--> All plots generated successfully!")
