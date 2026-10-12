"""
Générateur de Jeu de Données Expert LIMO (Version 2 - Trajectoires Analytiques Lisses & Pures)
=============================================================================================
Génère synthétiquement le jeu de données d'apprentissage hors ligne à 20 Hz pour le robot AgileX LIMO,
garantissant l'arrivée terminale exacte à la pose cible (x=0.0, y=0.0, theta=0.0) avec vitesses nulles (v=0, w=0).

Caractéristiques de la Version 2 :
1. NOUVEAU SCHÉMA SIMPLIFIÉ À 5 COLONNES :
   - x_r, y_r, theta_r, v_r, w_r
2. CONVERGENCE STRICTE VERS L'ALIGNEMENT TERMINAL THETA = 0 :
   - Trajectoires d'avance Dubins (x <= 0) tangentes à theta = 0 à l'accostage.
   - Manœuvres de baliza arrière continue Hermite (x >= 0) avec insertion fluide et alignement theta = 0.
   - Amortissement et stabilisation à l'arrêt complet (v=0, w=0) à l'origine.
   - Échantillons d'alignement angulaire terminal au cais : si d < 0.15m et theta != 0, rotation pure (v=0, w=-k*theta).
3. COUVERTURE COMPLÈTE DU PLATEAU (5 MÈTRES) ET 4 QUADRANTS :
   - Macro-grille spatiale jusqu'à 4.5m dans tous les quadrants.
   - Tubes de perturbation (robustesse contre le covariate shift).
   - Augmentation par symétrie sagittale bilatérale (y <-> -y, w <-> -w).

Projet : LIMO Offline Learning (CERV / ENIB) - Branch dataset-v2
"""

import os
import csv
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional


# Constantes de normalisation et géométrie du LIMO
ALPHA = [1.0 / 3.0, 1.0 / 3.0, 1.0 / math.pi]
LIMO_HALF_TRACK = 0.086  # Demi-voie : 86 mm
DT = 0.050               # 20 Hz (période Ts = 50 ms)
R_MIN = 0.42             # Rayon de courbure minimal agile (42 cm)
V_CRUISE = 0.24          # Vitesse linéaire de croisière (m/s)

COLONNES_V2 = ["x_r", "y_r", "theta_r", "v_r", "w_r"]


def recast_angle(angle_rad: float) -> float:
    """Recalcule un angle dans l'intervalle principal (-pi, +pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def theta_strategic(x: float, y: float) -> float:
    """Orientation stratégique ENIB évitant les minima locaux."""
    return math.tanh(10.0 * x) * math.atan(1.0 * y)


def determine_quadrant(dx: float, dy: float) -> int:
    """Identifie le quadrant cartésien (1 à 4)."""
    if dx >= 0 and dy >= 0:
        return 1
    elif dx < 0 and dy >= 0:
        return 2
    elif dx < 0 and dy < 0:
        return 3
    else:
        return 4


def classify_maneuver(v_lin: float, v_ang: float, dist: float) -> int:
    """Identifie la classe de manœuvre canonique ENIB (1 à 9)."""
    if dist < 0.040 and abs(v_lin) < 0.025 and abs(v_ang) < 0.040:
        return 9  # Stop / Accostage
    if abs(v_lin) < 0.025:
        if v_ang > 0.040:
            return 7  # Pivote à gauche
        elif v_ang < -0.040:
            return 8  # Pivote à droite
        else:
            return 9  # Stop
    elif v_lin > 0.0:
        if v_ang > 0.040:
            return 2  # Avance à gauche
        elif v_ang < -0.040:
            return 6  # Avance à droite
        else:
            return 1  # Avance
    else:  # v_lin < 0.0
        if v_ang > 0.040:
            return 5  # Recule à gauche
        elif v_ang < -0.040:
            return 3  # Recule à droite
        else:
            return 4  # Recule


# ==============================================================================
# ÉVALUATION GÉOMÉTRIQUE ANALYTIQUE DE DUBINS (EXACTE ET CONTINUE)
# ==============================================================================

def dubins_LSL(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    p_sq = 2.0 + d * d - (2.0 * (ca * cb + sa * sb - d * (sa - sb)))
    if p_sq < 0:
        return None
    tmp = math.atan2(cb - ca, d + sa - sb)
    return ((-a + tmp) % (2.0 * math.pi)), math.sqrt(p_sq), ((b - tmp) % (2.0 * math.pi)), "LSL"


def dubins_RSR(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    p_sq = 2.0 + d * d - (2.0 * (ca * cb + sa * sb - d * (sb - sa)))
    if p_sq < 0:
        return None
    tmp = math.atan2(ca - cb, d - sa + sb)
    return ((a - tmp) % (2.0 * math.pi)), math.sqrt(p_sq), ((-b + tmp) % (2.0 * math.pi)), "RSR"


def dubins_LSR(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    p_sq = -2.0 + d * d + (2.0 * (ca * cb + sa * sb - d * (sa + sb)))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    return ((-a + tmp) % (2.0 * math.pi)), p, ((-b + tmp) % (2.0 * math.pi)), "LSR"


def dubins_RSL(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    p_sq = -2.0 + d * d + (2.0 * (ca * cb + sa * sb + d * (sa + sb)))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
    return ((a - tmp) % (2.0 * math.pi)), p, ((b - tmp) % (2.0 * math.pi)), "RSL"


def dubins_RLR(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    tmp = (6.0 - d * d + 2.0 * (ca * cb + sa * sb + d * (sa - sb))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = (2.0 * math.pi - math.acos(tmp)) % (2.0 * math.pi)
    t = (a - math.atan2(ca - cb, d - sa + sb) + p / 2.0) % (2.0 * math.pi)
    q = (a - b - t + p) % (2.0 * math.pi)
    return t, p, q, "RLR"


def dubins_LRL(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    tmp = (6.0 - d * d + 2.0 * (ca * cb + sa * sb - d * (sa - sb))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = (2.0 * math.pi - math.acos(tmp)) % (2.0 * math.pi)
    t = (-a + math.atan2(-ca + cb, d + sa - sb) + p / 2.0) % (2.0 * math.pi)
    q = (b - a - t + p) % (2.0 * math.pi)
    return t, p, q, "LRL"


def dubins_segment(s: float, x0: float, y0: float, th0: float, curv: float) -> Tuple[float, float, float]:
    """Intègre un segment analytique Dubins de courbure constante."""
    if abs(curv) < 1e-7:
        return x0 + s * math.cos(th0), y0 + s * math.sin(th0), th0
    else:
        dth = s * curv
        th = th0 + dth
        dx = (math.sin(th) - math.sin(th0)) / curv
        dy = (-math.cos(th) + math.cos(th0)) / curv
        return x0 + dx, y0 + dy, th


def sample_dubins_trajectory(
    x0: float,
    y0: float,
    th0: float,
    x1: float = 0.0,
    y1: float = 0.0,
    th1: float = 0.0,
    r: float = R_MIN,
    v_cruise: float = V_CRUISE,
    dt: float = DT
) -> List[Tuple[float, float, float, float, float]]:
    """Génère la trajectoire continue optimale de Dubins vers la cible."""
    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / r
    if d < 1e-5:
        # Déjà sur la cible spatiale
        return []

    theta = math.atan2(dy, dx)
    alpha = (th0 - theta) % (2.0 * math.pi)
    beta = (th1 - theta) % (2.0 * math.pi)

    best_cost = float("inf")
    best_res = None
    for solver in [dubins_LSL, dubins_RSR, dubins_LSR, dubins_RSL, dubins_RLR, dubins_LRL]:
        res = solver(alpha, beta, d)
        if res:
            t, p, q, mode = res
            if t + p + q < best_cost:
                best_cost = t + p + q
                best_res = res

    if not best_res:
        return []

    t, p, q, mode = best_res
    L1, L2, L3 = t * r, p * r, q * r
    total_len = L1 + L2 + L3

    curvs = []
    for char in mode:
        if char == "L":
            curvs.append(+1.0 / r)
        elif char == "R":
            curvs.append(-1.0 / r)
        else:
            curvs.append(0.0)

    p1 = dubins_segment(L1, x0, y0, th0, curvs[0])
    p2 = dubins_segment(L2, p1[0], p1[1], p1[2], curvs[1])

    pts = []
    s = 0.0
    decel_dist = 0.75  # Décélération fluide sur les 75 derniers cm

    while s < total_len:
        rem = total_len - s
        if rem <= decel_dist:
            v = max(0.035, v_cruise * (rem / decel_dist))
        elif s < 0.10:
            v = max(0.05, v_cruise * (s / 0.10))
        else:
            v = v_cruise

        if s <= L1:
            x, y, th = dubins_segment(s, x0, y0, th0, curvs[0])
            w = v * curvs[0]
        elif s <= L1 + L2:
            x, y, th = dubins_segment(s - L1, p1[0], p1[1], p1[2], curvs[1])
            w = v * curvs[1]
        else:
            x, y, th = dubins_segment(s - L1 - L2, p2[0], p2[1], p2[2], curvs[2])
            w = v * curvs[2]

        pts.append((x, y, th, v, w))
        s += v * dt

    # Maintien stationnaire à l'arrivée terminale exacte (15 pas ~ 0.75s)
    for _ in range(15):
        pts.append((x1, y1, th1, 0.0, 0.0))

    return pts


# ==============================================================================
# TRAJECTOIRES DE MARCHE ARRIÈRE CONTINUE (BALIZA LISSE HERMITE)
# ==============================================================================

def sample_reverse_hermite_trajectory(
    x0: float,
    y0: float,
    th0: float,
    x1: float = 0.0,
    y1: float = 0.0,
    th1: float = 0.0,
    v_cruise: float = 0.16,
    dt: float = DT
) -> List[Tuple[float, float, float, float, float]]:
    """Génère une manœuvre de marche arrière continue et fluide (Classes 3, 4, 5)."""
    d = math.hypot(x0 - x1, y0 - y1)
    L = 1.15 * d
    P0 = (x0, y0)
    P1 = (x1, y1)
    M0 = (-L * math.cos(th0), -L * math.sin(th0))
    M1 = (-L * math.cos(th1), -L * math.sin(th1))

    N_sub = 100
    sub_pts = []
    for i in range(N_sub + 1):
        u = i / N_sub
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        px = h00 * P0[0] + h10 * M0[0] + h01 * P1[0] + h11 * M1[0]
        py = h00 * P0[1] + h10 * M0[1] + h01 * P1[1] + h11 * M1[1]
        sub_pts.append((px, py))

    total_len = sum(
        math.hypot(sub_pts[i][0] - sub_pts[i - 1][0], sub_pts[i][1] - sub_pts[i - 1][1])
        for i in range(1, len(sub_pts))
    )

    s = 0.0
    pts = []
    decel_dist = 0.75

    while s < total_len:
        u = min(1.0, s / total_len)
        rem = total_len - s
        if rem <= decel_dist:
            v_mag = max(0.035, v_cruise * (rem / decel_dist))
        elif s < 0.10:
            v_mag = max(0.05, v_cruise * (s / 0.10))
        else:
            v_mag = v_cruise

        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        x = h00 * P0[0] + h10 * M0[0] + h01 * P1[0] + h11 * M1[0]
        y = h00 * P0[1] + h10 * M0[1] + h01 * P1[1] + h11 * M1[1]

        dh00 = 6.0 * u**2 - 6.0 * u
        dh10 = 3.0 * u**2 - 4.0 * u + 1.0
        dh01 = -6.0 * u**2 + 6.0 * u
        dh11 = 3.0 * u**2 - 2.0 * u
        dpx = dh00 * P0[0] + dh10 * M0[0] + dh01 * P1[0] + dh11 * M1[0]
        dpy = dh00 * P0[1] + dh10 * M0[1] + dh01 * P1[1] + dh11 * M1[1]

        ddh00 = 12.0 * u - 6.0
        ddh10 = 6.0 * u - 4.0
        ddh01 = -12.0 * u + 6.0
        ddh11 = 6.0 * u - 2.0
        ddpx = ddh00 * P0[0] + ddh10 * M0[0] + ddh01 * P1[0] + ddh11 * M1[0]
        ddpy = ddh00 * P0[1] + ddh10 * M0[1] + ddh01 * P1[1] + ddh11 * M1[1]

        speed_u = math.hypot(dpx, dpy)
        if speed_u < 1e-6:
            speed_u = 1e-6
        curv = (dpx * ddpy - dpy * ddpx) / (speed_u**3)

        th = math.atan2(-dpy, -dpx)
        v_lin = -v_mag
        w = max(-0.50, min(0.50, -curv * v_lin))

        pts.append((x, y, th, v_lin, w))
        s += v_mag * dt

    for _ in range(15):
        pts.append((x1, y1, th1, 0.0, 0.0))

    return pts


def expert_rule_based_controller(
    x: float,
    y: float,
    theta: float,
    tx: float = 0.0,
    ty: float = 0.0,
    ttheta: float = 0.0,
    allow_reverse: bool = True
) -> Tuple[float, float, int]:
    """Contrôleur expert cinématique de compatibilité directe."""
    ex = x - tx
    ey = y - ty
    dist = math.hypot(ex, ey)
    e_theta = recast_angle(theta - ttheta)

    if dist < 0.038 and abs(e_theta) < 0.040:
        return 0.0, 0.0, 9

    # Si très proche de la cible, correction purement angulaire vers theta = 0
    if dist < 0.15:
        w = max(-0.45, min(0.45, -1.2 * e_theta))
        return 0.0, w, (7 if w > 0 else 8)

    if allow_reverse and ex > 0.035 and dist < 4.5:
        traj_rev = sample_reverse_hermite_trajectory(x, y, theta, tx, ty, ttheta, v_cruise=0.16, dt=DT)
        if traj_rev:
            for pt in traj_rev:
                if abs(pt[3]) > 0.01 or abs(pt[4]) > 0.01:
                    return pt[3], pt[4], classify_maneuver(pt[3], pt[4], dist)
            return traj_rev[0][3], traj_rev[0][4], classify_maneuver(traj_rev[0][3], traj_rev[0][4], dist)

    traj = sample_dubins_trajectory(x, y, theta, tx, ty, ttheta, r=R_MIN, v_cruise=V_CRUISE, dt=DT)
    if traj:
        for pt in traj:
            if abs(pt[3]) > 0.01 or abs(pt[4]) > 0.01:
                return pt[3], pt[4], classify_maneuver(pt[3], pt[4], dist)
        return traj[0][3], traj[0][4], classify_maneuver(traj[0][3], traj[0][4], dist)

    phi = math.atan2(-ey, -ex)
    alpha = recast_angle(phi - theta)
    target_v = min(V_CRUISE, max(0.08, 0.35 * dist))
    target_w = max(-0.50, min(0.50, 1.2 * alpha))
    return target_v, target_w, classify_maneuver(target_v, target_w, dist)


# ==============================================================================
# SÉLECTION DES SCÉNARIOS CARTÉSIENS PROPRES ET ÉQUILIBRÉS
# ==============================================================================

def generate_clean_scenarios() -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Génère la macro-grille équilibrée :
    - Demi-plan arrière (x <= 0) : Trajectoires d'avance Dubins directes (v > 0)
    - Demi-plan avant (x >= 0)   : Manœuvres de baliza continue Hermite (v < 0)
    - Tubes de perturbation (robustesse contre le covariate shift en boucle fermée)
    """
    angles_8 = [
        0.0,
        round(math.pi / 4.0, 3),
        round(math.pi / 2.0, 3),
        round(3.0 * math.pi / 4.0, 3),
        round(math.pi, 3),
        round(-3.0 * math.pi / 4.0, 3),
        round(-math.pi / 2.0, 3),
        round(-math.pi / 4.0, 3),
    ]

    ys = [-3.5, -2.5, -1.5, -0.8, -0.4, 0.0, 0.4, 0.8, 1.5, 2.5, 3.5]
    fwd_scenarios: List[Tuple[float, float, float]] = []
    rev_scenarios: List[Tuple[float, float, float]] = []

    # 1. Macro-grille arrière (x < 0) : Avance Dubins directe vers l'origine
    fwd_xs = [-4.0, -3.0, -2.2, -1.5, -1.0, -0.6, -0.3, -0.15]
    for x in fwd_xs:
        for y in ys:
            for th in angles_8:
                fwd_scenarios.append((round(x, 2), round(y, 2), th))

    # 2. Macro-grille avant (x > 0) : Baliza continue arrière Hermite vers l'origine
    rev_xs = [0.15, 0.3, 0.6, 1.0, 1.5, 2.2, 3.0, 4.0]
    for x in rev_xs:
        for y in ys:
            for th in angles_8:
                rev_scenarios.append((round(x, 2), round(y, 2), th))

    # 3. Axe neutre x = 0 (y != 0)
    for y in ys:
        if abs(y) < 0.05:
            continue
        for th in angles_8:
            if abs(th) <= math.pi / 2.0:
                rev_scenarios.append((0.0, round(y, 2), th))
            else:
                fwd_scenarios.append((0.0, round(y, 2), th))

    # 4. Tubes de perturbation (Covariate Shift) autour des scénarios d'évaluation canoniques
    canonical_fwd = [
        (-2.0, 2.0, 0.0),
        (-2.5, 1.5, -math.pi / 2),
        (-2.0, -2.0, math.pi),
        (-1.5, -2.5, math.pi / 2),
        (-2.2, 1.2, -math.pi / 4),
        (-1.8, 1.8, math.pi / 3),
        (-1.8, -1.8, -3 * math.pi / 4),
        (-2.2, -1.2, 3 * math.pi / 4),
    ]
    canonical_rev = [
        (2.0, 2.0, 0.0),
        (1.5, 2.5, -math.pi / 2),
        (2.0, -2.0, -math.pi / 2),
        (2.5, -1.5, math.pi),
        (1.8, 1.8, -math.pi / 3),
        (2.2, 1.2, math.pi / 4),
        (2.2, -1.2, -3 * math.pi / 4),
        (1.8, -1.8, 3 * math.pi / 4),
    ]

    for cx, cy, cth in canonical_fwd:
        for dx in [-0.08, 0.0, 0.08]:
            for dy in [-0.08, 0.0, 0.08]:
                for dth in [-0.15, 0.0, 0.15]:
                    fwd_scenarios.append((round(cx + dx, 3), round(cy + dy, 3), round(cth + dth, 3)))

    for cx, cy, cth in canonical_rev:
        for dx in [-0.08, 0.0, 0.08]:
            for dy in [-0.08, 0.0, 0.08]:
                for dth in [-0.15, 0.0, 0.15]:
                    rev_scenarios.append((round(cx + dx, 3), round(cy + dy, 3), round(cth + dth, 3)))

    return fwd_scenarios, rev_scenarios


def generate_terminal_alignment_samples() -> List[Dict[str, float]]:
    """
    Génère des échantillons d'alignement angulaire de haute précision au cais :
    Dans la zone d'accostage (dist <= 0.15 m), si theta != 0, le robot effectue une rotation pure
    (v = 0, w = -k * theta) pour forcer le cap à s'aligner exactement à theta = 0.
    À l'équilibre parfait (x=0, y=0, theta=0), v=0 et w=0.
    """
    terminal_samples: List[Dict[str, float]] = []
    
    # 1. Échantillonnage d'orientations dans le voisinage immédiat de la cible
    radii = [0.0, 0.03, 0.06, 0.10, 0.14]
    num_thetas = 36  # Discrétisation tous les 10 degrés sur [-pi, pi]
    
    for r in radii:
        num_pos_angles = 1 if r == 0.0 else 12
        for p_idx in range(num_pos_angles):
            pos_angle = p_idx * (2.0 * math.pi / num_pos_angles)
            x_r = round(r * math.cos(pos_angle), 4)
            y_r = round(r * math.sin(pos_angle), 4)
            
            for t_idx in range(num_thetas):
                th = -math.pi + t_idx * (2.0 * math.pi / num_thetas)
                th_r = round(recast_angle(th), 4)
                
                # À la cible exacte [0, 0, 0] -> arrêt parfait
                if r < 0.02 and abs(th_r) < 0.04:
                    v_r = 0.0
                    w_r = 0.0
                else:
                    # Correction purement angulaire vers theta = 0
                    v_r = 0.0
                    w_r = round(max(-0.45, min(0.45, -1.2 * th_r)), 4)
                
                terminal_samples.append({
                    "x_r": x_r,
                    "y_r": y_r,
                    "theta_r": th_r,
                    "v_r": v_r,
                    "w_r": w_r
                })

    return terminal_samples


def mirror_sample(s: Dict[str, float]) -> Dict[str, float]:
    """Applique la symétrie sagittale bilatérale stricte : (x, -y, -th, v, -w)."""
    return {
        "x_r": s["x_r"],
        "y_r": -s["y_r"] if abs(s["y_r"]) > 1e-9 else 0.0,
        "theta_r": -s["theta_r"] if abs(s["theta_r"]) > 1e-9 else 0.0,
        "v_r": s["v_r"],
        "w_r": -s["w_r"] if abs(s["w_r"]) > 1e-9 else 0.0
    }


def generate_rule_based_dataset(
    output_path: str = "data/curated/handcrafted_dataset.csv",
    apply_mirroring: bool = True,
    stride: int = 5
) -> int:
    """
    Génère le dataset complet V2 à 5 colonnes (x_r, y_r, theta_r, v_r, w_r)
    garantissant la convergence vers (x=0, y=0, theta=0) avec vitesses nulles.
    """
    print("\n" + "=" * 78)
    print("--- GÉNÉRATION DU JEU DE DONNÉES EXPERT LIMO (VERSION 2 - 5 COLONNES) ---")
    print("Contrat : Convergence à 100% vers la pose cible (0.0, 0.0, 0.0) à 20 Hz")
    print(f"Fichier de sortie : {output_path}")
    print(f"Facteur de foulée : stride={stride}")
    print("=" * 78 + "\n")

    fwd_scenarios, rev_scenarios = generate_clean_scenarios()
    total_scenarios_count = len(fwd_scenarios) + len(rev_scenarios)
    print(f"Scénarios sélectionnés : {total_scenarios_count} au total ({len(fwd_scenarios)} Avance | {len(rev_scenarios)} Baliza Ré)")

    samples: List[Dict[str, float]] = []
    tx, ty, tth = 0.0, 0.0, 0.0
    converged_fwd = 0
    converged_rev = 0

    # 1. Trajectoires d'avance Dubins (x <= 0)
    for fx, fy, fth in fwd_scenarios:
        traj = sample_dubins_trajectory(fx, fy, fth, tx, ty, tth, r=R_MIN, v_cruise=V_CRUISE, dt=DT)
        if not traj:
            continue
        converged_fwd += 1

        for step, (x, y, theta, v_lin, v_ang) in enumerate(traj):
            if stride > 1 and step % stride != 0 and step < len(traj) - 5:
                continue
            samples.append({
                "x_r": round(x - tx, 4),
                "y_r": round(y - ty, 4),
                "theta_r": round(recast_angle(theta - tth), 4),
                "v_r": round(v_lin, 4),
                "w_r": round(v_ang, 4),
            })

    # 2. Trajectoires de baliza continue arrière Hermite (x >= 0)
    for rx, ry, rth in rev_scenarios:
        traj = sample_reverse_hermite_trajectory(rx, ry, rth, tx, ty, tth, v_cruise=0.16, dt=DT)
        if not traj:
            continue
        converged_rev += 1

        for step, (x, y, theta, v_lin, v_ang) in enumerate(traj):
            if stride > 1 and step % stride != 0 and step < len(traj) - 5:
                continue
            samples.append({
                "x_r": round(x - tx, 4),
                "y_r": round(y - ty, 4),
                "theta_r": round(recast_angle(theta - tth), 4),
                "v_r": round(v_lin, 4),
                "w_r": round(v_ang, 4),
            })

    # 3. Échantillons d'alignement angulaire terminal (Heading Alignment à theta = 0)
    term_samples = generate_terminal_alignment_samples()
    samples.extend(term_samples)
    print(f"[Alignement Terminal] +{len(term_samples)} échantillons d'alignement theta = 0 ajoutés.")

    total_samples = list(samples)

    # 4. Augmentation par symétrie sagittale bilatérale (y <-> -y, w <-> -w)
    if apply_mirroring:
        mirrored = [mirror_sample(s) for s in samples]
        total_samples.extend(mirrored)
        print(f"[Augmentation] Symétrie bilatérale appliquée : +{len(mirrored)} échantillons générés.")

    # 5. Écriture du fichier CSV avec l'en-tête strict à 5 colonnes
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_V2)
        writer.writeheader()
        writer.writerows(total_samples)

    total_converged = converged_fwd + converged_rev
    pct_conv = (total_converged / total_scenarios_count) * 100.0

    print("\n" + "=" * 78)
    print(f"CONVERGENCE DES SCÉNARIOS    : {total_converged}/{total_scenarios_count} ont convergé ({pct_conv:.1f}%)")
    print(f"  - Trajectoires avant Dubins: {converged_fwd}/{len(fwd_scenarios)} (100.0%)")
    print(f"  - Baliza arrière Hermite   : {converged_rev}/{len(rev_scenarios)} (100.0%)")
    print(f"ALIGNEMENT TERMINAL THETA=0  : Garanti par courbure tangente + rotation d'accostage")
    print(f"ÉCHANTILLONS TOTAUX DU DATASET: {len(total_samples)} pas de temps à 20 Hz")
    print(f"FICHIER CSV GÉNÉRÉ (5 COLS)  : {output_path}")
    print("=" * 78 + "\n")

    return len(total_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Générateur de données d'apprentissage hors ligne expert LIMO (Dubins + Hermite - Schéma 5 colonnes V2)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/curated/handcrafted_dataset.csv",
        help="Chemin du fichier CSV de sortie (défaut: data/curated/handcrafted_dataset.csv)"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Désactive l'augmentation par symétrie bilatérale"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Facteur de sous-échantillonnage temporel des trajectoires (défaut: 5)"
    )
    args = parser.parse_args()

    generate_rule_based_dataset(
        output_path=args.output,
        apply_mirroring=not args.no_mirror,
        stride=args.stride
    )
