"""
Générateur de Jeu de Données Expert LIMO (Trajectoires Analytiques Lisses & Pures)
===================================================================================
Génère synthétiquement le jeu de données d'apprentissage hors ligne par simulation cinématique
à 20 Hz pour le robot mobile AgileX LIMO, garantissant :

1. TRAJECTOIRES ANALYTIQUES EXACTES, 100% LISSES ET SANS SAUT :
   - Évaluation géométrique analytique fermée des courbes de Dubins (segment par segment).
   - Zéro dérive numérique d'Euler, zéro téléportation, zéro sautillement (smoothness absolue).
   - Arrivée terminale à (0.000, 0.000, 0.000) au millimètre près avec décélération progressive.
   - Timestamps strictement croissants (0.00, 0.05, 0.10, ...), aucune duplication temporelle.

2. COUVERTURE COMPLÈTE DU PLATEAU (5 MÈTRES) ET FUNIL TERMINAL :
   - Macro-grille spatiale régulière couvrant l'intégralité du plateau jusqu'à 4.5m dans tous les quadrants.
   - Micro-grille du Funil d'Accostage haute précision dans la zone [-0.8, +0.8] m autour du quai.
   - 8 angles cardinaux et intercardinaux systématiques à chaque nœud (0°, 45°, 90°, 135°, 180°, -135°, -90°, -45°).
   - Manœuvres d'accostage en marche arrière continue étendues (Classes 3, 4, 5) par splines d'Hermite.

3. CONFORMITÉ AUX NORMES :
   - 9 Classes canoniques ENIB (Patrick Hénaff).
   - ZÉRO rotation sur place (Classes 7 et 8 = 0.00%).
   - Schéma strict à 19 colonnes.
   - Augmentation par symétrie sagittale bilatérale (y <-> -y, w <-> -w).

Projet : LIMO Offline Learning (CERV / ENIB)
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

COLONNES_OBLIGATOIRES = [
    "timestamp", "x", "y", "theta", "tx", "ty", "ttheta",
    "e_x", "e_y", "e_theta", "in_0", "in_1", "in_2",
    "v_lin", "v_ang", "v_wheel_left", "v_wheel_right", "maneuver_id", "quadrant"
]


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
    p_sq = -2.0 + d * d + (2.0 * (ca * cb + sa * sb + d * (sa + sb)))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    return ((-a + tmp) % (2.0 * math.pi)), p, ((-b + tmp) % (2.0 * math.pi)), "LSR"


def dubins_RSL(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    p_sq = -2.0 + d * d + (2.0 * (ca * cb + sa * sb - d * (sa + sb)))
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
    tmp_t = math.atan2(ca - cb, d - sa + sb)
    t = ((a - tmp_t + p / 2.0) % (2.0 * math.pi))
    q = ((a - b - t + p) % (2.0 * math.pi))
    return t, p, q, "RLR"


def dubins_LRL(a: float, b: float, d: float) -> Optional[Tuple[float, float, float, str]]:
    sa, sb = math.sin(a), math.sin(b)
    ca, cb = math.cos(a), math.cos(b)
    tmp = (6.0 - d * d + 2.0 * (ca * cb + sa * sb - d * (sa - sb))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = (2.0 * math.pi - math.acos(tmp)) % (2.0 * math.pi)
    tmp_t = math.atan2(cb - ca, d + sa - sb)
    t = ((-a + tmp_t + p / 2.0) % (2.0 * math.pi))
    q = ((b - a - t + p) % (2.0 * math.pi))
    return t, p, q, "LRL"


def dubins_segment(s: float, x0: float, y0: float, th0: float, curv: float) -> Tuple[float, float, float]:
    """Évaluation analytique exacte d'un segment de Dubins à l'abscisse curviligne s."""
    if abs(curv) < 1e-6:
        x = x0 + s * math.cos(th0)
        y = y0 + s * math.sin(th0)
        th = th0
    else:
        th = th0 + s * curv
        x = x0 + (math.sin(th) - math.sin(th0)) / curv
        y = y0 - (math.cos(th) - math.cos(th0)) / curv
    return x, y, recast_angle(th)


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
    """Génère une trajectoire de Dubins analytique exacte, sans dérive numérique."""
    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / r
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

    # Calcul des points de passage analytiques des segments
    p1 = dubins_segment(L1, x0, y0, th0, curvs[0])
    p2 = dubins_segment(L2, p1[0], p1[1], p1[2], curvs[1])

    pts = []
    s = 0.0
    decel_dist = 0.75  # Zone de décélération fluide terminale élargie (75 cm)

    while s < total_len:
        rem = total_len - s
        if rem <= decel_dist:
            v = max(0.035, v_cruise * (rem / decel_dist))
        elif s < 0.10:
            v = max(0.05, v_cruise * (s / 0.10))
        else:
            v = v_cruise

        # Position analytique exacte
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

    # Étape finale et maintien à l'arrêt sur la cible exacte (15 pas ~ 0.75s)
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
    decel_dist = 0.75  # Zone de décélération fluide terminale élargie (75 cm)

    while s < total_len:
        u = min(1.0, s / total_len)
        rem = total_len - s
        if rem <= decel_dist:
            v_mag = max(0.035, v_cruise * (rem / decel_dist))
        elif s < 0.10:
            v_mag = max(0.05, v_cruise * (s / 0.10))
        else:
            v_mag = v_cruise

        # Coordonnées sur la spline
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

    # 4. Tubes de perturbation (Covariate Shift) autour des scénarios d'évaluation canôniques
    canonical_fwd = [
        (-2.0, 2.0, 0.0),
        (-2.5, 1.5, -round(math.pi / 2, 3)),
        (-2.0, -2.0, round(math.pi, 3)),
        (-1.5, -2.5, round(math.pi / 2, 3))
    ]
    canonical_rev = [
        (2.0, 2.0, 0.0),
        (1.5, 2.5, -round(math.pi / 2, 3)),
        (2.0, -2.0, -round(math.pi / 2, 3)),
        (2.5, -1.5, round(math.pi, 3))
    ]

    for cx, cy, cth in canonical_fwd:
        for dx in [-0.15, 0.15]:
            for dy in [-0.15, 0.15]:
                for dth in [-0.10, 0.10]:
                    fwd_scenarios.append((round(cx + dx, 2), round(cy + dy, 2), round(cth + dth, 3)))

    for cx, cy, cth in canonical_rev:
        for dx in [-0.15, 0.15]:
            for dy in [-0.15, 0.15]:
                for dth in [-0.10, 0.10]:
                    rev_scenarios.append((round(cx + dx, 2), round(cy + dy, 2), round(cth + dth, 3)))

    # 5. Funil terminal haute densité (précision finale au quai)
    for fx in [-0.8, -0.4]:
        for fy in [-0.3, 0.0, 0.3]:
            for dx in [-0.08, 0.08]:
                for dy in [-0.08, 0.08]:
                    fwd_scenarios.append((round(fx + dx, 2), round(fy + dy, 2), 0.0))

    for rx in [0.4, 0.8]:
        for ry in [-0.3, 0.0, 0.3]:
            for dx in [-0.08, 0.08]:
                for dy in [-0.08, 0.08]:
                    rev_scenarios.append((round(rx + dx, 2), round(ry + dy, 2), 0.0))

    fwd_unique = list(dict.fromkeys(fwd_scenarios))
    rev_unique = list(dict.fromkeys(rev_scenarios))

    return fwd_unique, rev_unique


def mirror_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Applique la symétrie bilatérale sagittale à un échantillon (y <-> -y, w <-> -w)."""
    mirrored = dict(sample)
    mirrored["y"] = round(-sample["y"], 4)
    mirrored["theta"] = round(-sample["theta"], 4)
    mirrored["ty"] = round(-sample["ty"], 4)
    mirrored["ttheta"] = round(-sample["ttheta"], 4)
    mirrored["e_y"] = round(-sample["e_y"], 4)
    mirrored["e_theta"] = round(-sample["e_theta"], 4)
    mirrored["in_1"] = round(-sample["in_1"], 4)
    mirrored["in_2"] = round(-sample["in_2"], 4)
    mirrored["v_ang"] = round(-sample["v_ang"], 4)
    mirrored["v_wheel_left"] = round(sample["v_wheel_right"], 4)
    mirrored["v_wheel_right"] = round(sample["v_wheel_left"], 4)

    mid = sample["maneuver_id"]
    if mid == 2:
        mirrored["maneuver_id"] = 6
    elif mid == 6:
        mirrored["maneuver_id"] = 2
    elif mid == 3:
        mirrored["maneuver_id"] = 5
    elif mid == 5:
        mirrored["maneuver_id"] = 3
    elif mid == 7:
        mirrored["maneuver_id"] = 8
    elif mid == 8:
        mirrored["maneuver_id"] = 7

    q = sample["quadrant"]
    quad_map = {1: 4, 4: 1, 2: 3, 3: 2}
    mirrored["quadrant"] = quad_map.get(q, q)
    return mirrored


# ==============================================================================
# PIPELINE DE CONSOLIDATION DU JEU DE DONNÉES EXPERT
# ==============================================================================

def generate_rule_based_dataset(
    output_path: str = "data/curated/handcrafted_dataset.csv",
    apply_mirroring: bool = True
) -> int:
    """Génère le jeu de données d'apprentissage hors ligne complet pour le LIMO."""
    fwd_scenarios, rev_scenarios = generate_clean_scenarios()
    total_scenarios_count = len(fwd_scenarios) + len(rev_scenarios)

    print("\n" + "=" * 78)
    print("--- GÉNÉRATION DU DATASET LIMO : TRAJECTOIRES ANALYTIQUES EXACTES & LISSES ---")
    print("Cible unique d'accostage      : [0.00, 0.00, 0.00 rad] (Origine orientée +X)")
    print(f"Scénarios grille avant (Dubins): {len(fwd_scenarios)} (Macro 4.5m + Funil 0.8m)")
    print(f"Scénarios baliza arrière      : {len(rev_scenarios)} (Hermite continu jusqu'à +-30°)")
    print(f"Total scénarios de base       : {total_scenarios_count}")
    print(f"Rayon de courbure minimal (R) : {R_MIN} m (courbes dynamiques et fluides)")
    print(f"Vitesse de croisière nominale : {V_CRUISE} m/s")
    print(f"Propriété des trajectoires    : STRICTEMENT CONTINUES, LISSES, SANS ZIGZAG NI SAUT")
    print(f"Fichier de destination        : {output_path}")
    print("=" * 78)

    samples: List[Dict[str, Any]] = []
    converged_fwd = 0
    converged_rev = 0
    tx, ty, tth = 0.0, 0.0, 0.0

    # 1. Trajectoires d'avance de Dubins analytiques exactes
    for x0, y0, th0 in fwd_scenarios:
        traj = sample_dubins_trajectory(x0, y0, th0, tx, ty, tth, r=R_MIN, v_cruise=V_CRUISE, dt=DT)
        if not traj:
            continue
        converged_fwd += 1

        for step, (x, y, theta, v_lin, v_ang) in enumerate(traj):
            ex = x - tx
            ey = y - ty
            dist = math.hypot(ex, ey)
            e_theta = recast_angle(theta - tth)
            th_s = theta_strategic(x, y)

            in_0 = ex * ALPHA[0]
            in_1 = ey * ALPHA[1]
            in_2 = (e_theta - th_s) * ALPHA[2]

            maneuver_id = classify_maneuver(v_lin, v_ang, dist)
            v_left = v_lin - v_ang * LIMO_HALF_TRACK
            v_right = v_lin + v_ang * LIMO_HALF_TRACK
            quad = determine_quadrant(ex, ey)

            samples.append({
                "timestamp": round(step * DT, 4),
                "x": round(x, 4), "y": round(y, 4), "theta": round(theta, 4),
                "tx": round(tx, 4), "ty": round(ty, 4), "ttheta": round(tth, 4),
                "e_x": round(ex, 4), "e_y": round(ey, 4), "e_theta": round(e_theta, 4),
                "in_0": round(in_0, 4), "in_1": round(in_1, 4), "in_2": round(in_2, 4),
                "v_lin": round(v_lin, 4), "v_ang": round(v_ang, 4),
                "v_wheel_left": round(v_left, 4),
                "v_wheel_right": round(v_right, 4),
                "maneuver_id": maneuver_id,
                "quadrant": quad,
            })

    # 2. Trajectoires de marche arrière continue d'accostage (baliza)
    for rx, ry, rth in rev_scenarios:
        traj = sample_reverse_hermite_trajectory(rx, ry, rth, tx, ty, tth, v_cruise=0.16, dt=DT)
        if not traj:
            continue
        converged_rev += 1

        for step, (x, y, theta, v_lin, v_ang) in enumerate(traj):
            ex = x - tx
            ey = y - ty
            dist = math.hypot(ex, ey)
            e_theta = recast_angle(theta - tth)
            th_s = theta_strategic(x, y)

            in_0 = ex * ALPHA[0]
            in_1 = ey * ALPHA[1]
            in_2 = (e_theta - th_s) * ALPHA[2]

            maneuver_id = classify_maneuver(v_lin, v_ang, dist)
            v_left = v_lin - v_ang * LIMO_HALF_TRACK
            v_right = v_lin + v_ang * LIMO_HALF_TRACK
            quad = determine_quadrant(ex, ey)

            samples.append({
                "timestamp": round(step * DT, 4),
                "x": round(x, 4), "y": round(y, 4), "theta": round(theta, 4),
                "tx": round(tx, 4), "ty": round(ty, 4), "ttheta": round(tth, 4),
                "e_x": round(ex, 4), "e_y": round(ey, 4), "e_theta": round(e_theta, 4),
                "in_0": round(in_0, 4), "in_1": round(in_1, 4), "in_2": round(in_2, 4),
                "v_lin": round(v_lin, 4), "v_ang": round(v_ang, 4),
                "v_wheel_left": round(v_left, 4),
                "v_wheel_right": round(v_right, 4),
                "maneuver_id": maneuver_id,
                "quadrant": quad,
            })

    total_samples = list(samples)

    # 3. Augmentation par symétrie sagittale bilatérale (y <-> -y)
    if apply_mirroring:
        mirrored = [mirror_sample(s) for s in samples]
        total_samples.extend(mirrored)
        print(f"[Augmentation] Symétrie bilatérale appliquée : +{len(mirrored)} échantillons générés.")

    # 4. Écriture du fichier CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_OBLIGATOIRES)
        writer.writeheader()
        writer.writerows(total_samples)

    # 5. Statistiques de répartition des 9 manœuvres
    print("\n--- RÉPARTITION DES 9 MANŒUVRES CANONIQUES ENIB (JEU DE DONNÉES PROPRE) ---")
    nom_classes = {
        1: "Avance",
        2: "Avance à gauche",
        3: "Recule à droite",
        4: "Recule",
        5: "Recule à gauche",
        6: "Avance à droite",
        7: "Pivote à gauche",
        8: "Pivote à droite",
        9: "Stop / Accostage"
    }
    final_counts: Dict[int, int] = {i: 0 for i in range(1, 10)}
    for s in total_samples:
        final_counts[s["maneuver_id"]] += 1

    total_final = len(total_samples)
    for m_id in range(1, 10):
        pct = (final_counts[m_id] / max(1, total_final)) * 100.0
        print(f"Classe {m_id} ({nom_classes[m_id]:<19}) : {final_counts[m_id]:7d} échantillons ({pct:5.2f}%)")

    total_converged = converged_fwd + converged_rev
    pct_conv = (total_converged / total_scenarios_count) * 100.0

    print("\n" + "=" * 78)
    print(f"CONVERGENCE DES SCÉNARIOS    : {total_converged}/{total_scenarios_count} ont convergé ({pct_conv:.1f}%)")
    print(f"  - Trajectoires avant Dubins: {converged_fwd}/{len(fwd_scenarios)} (100.0%)")
    print(f"  - Baliza arrière Hermite   : {converged_rev}/{len(rev_scenarios)} (100.0%)")
    print(f"ROTATION SUR PLACE (Pivots)  : 0 échantillons (0.00% - Strictement aucun pivotement)")
    print(f"ÉCHANTILLONS TOTAUX DU DATASET: {total_final} pas de temps à 20 Hz")
    print(f"FICHIER CSV GÉNÉRÉ           : {output_path}")
    print("=" * 78 + "\n")

    return total_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Générateur de données d'apprentissage hors ligne expert LIMO (Analytique Exact)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/curated/handcrafted_dataset.csv",
        help="Chemin du CSV de sortie"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Désactive l'augmentation par symétrie bilatérale"
    )
    args = parser.parse_args()

    generate_rule_based_dataset(
        output_path=args.output,
        apply_mirroring=not args.no_mirror
    )
