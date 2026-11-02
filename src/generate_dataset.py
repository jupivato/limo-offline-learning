"""
Gerador de Dataset de Aprendizado Offline Perfeitamente Equilibrado — AgileX LIMO (Norma ENIB)
=============================================================================================
Gera o conjunto de dados canônico com estritamente 5 COLUNAS para o aprendizado
supervisionado offline (Behavioral Cloning) da rede PioneerNN.

Esquema de 5 Colunas (Robot-Centric Relative Mapping):
1. x_target_robot : Coordenada X da doca no referencial do robô (m)
2. y_target_robot : Coordenada Y da doca no referencial do robô (m)
3. theta_robot    : Orientação absoluta do robô no referencial global (rad) [-pi, +pi]
4. v_robot        : Consigne de velocidade linear {-V_MAX, 0.0, +V_MAX} (m/s)
5. w_robot        : Consigne de velocidade angular {-W_MAX, 0.0, +W_MAX} (rad/s)

Estrutura das 9 Manobras Canônicas da ENIB (Prof. Patrick Hénaff):
- Classe 1: Avanço Retilíneo Puro (v > 0, w = 0)
- Classe 2: Ré Retilínea Pura (v < 0, w = 0)
- Classe 3: Rotação Pura Anti-horária / Pivô Esquerda (v = 0, w > 0)
- Classe 4: Rotação Pura Horária / Pivô Direita (v = 0, w < 0)
- Classe 5: Curva Suave Frente-Esquerda (v > 0, w > 0)
- Classe 6: Curva Suave Frente-Direita (v > 0, w < 0)
- Classe 7: Curva Ré-Esquerda / Baliza (v < 0, w < 0)
- Classe 8: Curva Ré-Direita / Baliza (v < 0, w > 0)
- Classe 9: Atracação e Estabilização Final / Stop (v = 0, w = 0)

Equilíbrio Perfeito e Justo (11,11% para cada uma das 9 manobras):
- Amostragem com passo espacial frequente e constante: STEP_XY = 0.20 m (-5.0m a +5.0m).
- Amostragem angular frequente e constante: 24 orientações discretas a cada 15° (passo pi/12 rad).
- Réplicas exatas de ancoragem terminal na doca: (0.0, 0.0, 0.0) -> (0.0, 0.0) [Classe 9].
- Balanceamento estratificado determinístico: exatamente 5.000 amostras para cada uma das 9 classes
  (Total: 45.000 amostras, 11,11% cada, com simetria bilateral exata).

Projeto: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
"""

from __future__ import annotations

import os
import sys
import csv
import math
import random
import argparse
from typing import List, Dict, Any, Tuple

# Resolução de caminho da raiz do projeto
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ==============================================================================
# ⚙️ PARÂMETROS CINEMÁTICOS E ESPACIAIS DA GRADE REGULAR
# ==============================================================================
V_MAX = 0.50                   # Velocidade linear máxima em m/s {-V_MAX, 0.0, +V_MAX}
W_MAX = 1.00                   # Velocidade angular máxima em rad/s {-W_MAX, 0.0, +W_MAX}
X_MAX = 5.0                    # Limite da arena em X [-5.0, +5.0] m
Y_MAX = 5.0                    # Limite da arena em Y [-5.0, +5.0] m
STEP_XY = 0.20                 # Passo frequente e constante de amostragem espacial (m)
NUM_ANGLES = 24                # Passo frequente de amostragem angular: 24 direções a cada 15° (pi/12 rad)

STEER_DEADBAND = 0.26          # Margem angular para avanço retilíneo puro (~14.9°)
REVERSE_DEADBAND = 0.28        # Margem de visada angular para ré retilínea pura (~16.0°)
D_STOP = 0.15                  # Raio de doca para atracação e parada (m)
D_STOP_ANGLE_TOL = 0.35        # Tolerância angular para atracação final (~20°, compatível com run_autonomous)
D_BRAKE = 0.30                 # Distância de frenagem e desaceleração antes do cais (m)
MAX_PIVOT_PHI = 1.35           # Ângulo lateral a partir do qual executa pivô no próprio eixo (~77.3°)

SAMPLES_PER_CLASS = 5000       # Cota exata e justa por classe (5.000 x 9 = 45.000 amostras, 11,11% cada)

# ESQUEMA RESTRITO A EXATAMENTE 5 COLUNAS (Sem colunas extras)
COLUMNS = ["x_target_robot", "y_target_robot", "theta_robot", "v_robot", "w_robot"]

# Nomenclatura das 9 manobras da ENIB para auditoria e relatório estatístico
ENIB_MANEUVER_NAMES = {
    1: "Classe 1 : Avanço Retilíneo Puro",
    2: "Classe 2 : Ré Retilínea Pura",
    3: "Classe 3 : Rotação Pura Anti-horária (Pivô Esquerda)",
    4: "Classe 4 : Rotação Pura Horária (Pivô Direita)",
    5: "Classe 5 : Curva Suave Frente-Esquerda",
    6: "Classe 6 : Curva Suave Frente-Direita",
    7: "Classe 7 : Curva Ré-Esquerda (Baliza)",
    8: "Classe 8 : Curva Ré-Direita (Baliza)",
    9: "Classe 9 : Atracação e Estabilização Final (Stop)"
}


def recast_angle(angle_rad: float) -> float:
    """Normaliza o ângulo rigorosamente no intervalo (-pi, +pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def classify_enib_maneuver(v: float, w: float) -> Tuple[int, str]:
    """
    Classifica a ação nas 9 classes canônicas da ENIB (Prof. Patrick Hénaff).
    """
    V_EPS = 0.025
    W_EPS = 0.040

    if abs(v) <= V_EPS and abs(w) <= W_EPS:
        m_id = 9
    elif abs(v) <= V_EPS:
        m_id = 3 if w > 0.0 else 4
    elif v > V_EPS:
        if abs(w) <= W_EPS:
            m_id = 1
        elif w > 0.0:
            m_id = 5
        else:
            m_id = 6
    else:  # v < -V_EPS
        if abs(w) <= W_EPS:
            m_id = 2
        elif w < 0.0:
            m_id = 7
        else:
            m_id = 8

    return m_id, ENIB_MANEUVER_NAMES[m_id]


def expert_rule_based_controller(
    x_tr: float,
    y_tr: float,
    theta_r: float,
    v_max: float = V_MAX,
    w_max: float = W_MAX,
    d_brake: float = D_BRAKE,
    d_stop: float = D_STOP
) -> Tuple[float, float, int]:
    """
    Controlador especialista determinístico baseado nas 9 manobras da ENIB.

    Entradas:
        x_tr, y_tr: Posição do alvo expressa no referencial local do robô (m).
        theta_r   : Orientação absoluta do robô em relação ao cais (rad).

    Saídas:
        v_robot, w_robot: Velocidades discretizadas {-v_max, 0, v_max} e {-w_max, 0, w_max}.
        maneuver_id: Classe canônica da manobra (1 a 9).
    """
    dist = math.hypot(x_tr, y_tr)
    e_theta = recast_angle(theta_r)
    phi = math.atan2(y_tr, x_tr)

    # 1. Zona Morta de Atracação Final na Doca (d <= d_stop e ângulo alinhado)
    if dist <= d_stop:
        if abs(e_theta) <= D_STOP_ANGLE_TOL:
            v, w = 0.0, 0.0   # Parada perfeita alinhada
        elif e_theta > 0.0:
            v, w = 0.0, -w_max  # Pivô horário
        else:
            v, w = 0.0, w_max   # Pivô anti-horário
        m_id, _ = classify_enib_maneuver(v, w)
        return v, w, m_id

    # 2. Zona de Frenagem e Alinhamento no Cais (d <= d_brake)
    if dist <= d_brake:
        v = 0.0
        if abs(e_theta) <= 0.20:
            w = 0.0
        elif e_theta > 0.0:
            w = -w_max
        else:
            w = w_max
        m_id, _ = classify_enib_maneuver(v, w)
        return v, w, m_id

    # 3. Zona de Baliza de Ré (Alvo diretamente na traseira: x_tr < -d_brake e |y_tr| <= 1.5 m)
    if x_tr < -d_brake and abs(y_tr) <= 1.5:
        v = -v_max
        phi_rear = math.atan2(y_tr, -x_tr)
        if phi_rear > REVERSE_DEADBAND:
            w = -w_max  # Leva a traseira à esquerda (girando horário)
        elif phi_rear < -REVERSE_DEADBAND:
            w = w_max   # Leva a traseira à direita (girando anti-horário)
        else:
            w = 0.0     # Ré retilínea pura
        m_id, _ = classify_enib_maneuver(v, w)
        return v, w, m_id

    # 4. Alvo Muito Lateral ou Fora do Campo Frontal (|phi| > 77.3° = ~1.35 rad)
    if abs(phi) > MAX_PIVOT_PHI:
        v = 0.0
        w = w_max if phi > 0.0 else -w_max
        m_id, _ = classify_enib_maneuver(v, w)
        return v, w, m_id

    # 5. Aproximação Frontal Suave (|phi| <= 77.3°)
    v = v_max
    if dist < 1.2:
        # Perto da vaga (< 1.2m), combina a visada do alvo com o alinhamento terminal (theta -> 0)
        steer_err = phi + 0.4 * (-e_theta)
    else:
        # Longe da vaga, mira de forma direta na linha de visada do alvo
        steer_err = phi

    if steer_err > STEER_DEADBAND:
        w = w_max   # Curva para a esquerda
    elif steer_err < -STEER_DEADBAND:
        w = -w_max  # Curva para a direita
    else:
        w = 0.0     # Avanço retilíneo puro

    m_id, _ = classify_enib_maneuver(v, w)
    return v, w, m_id


def generate_balanced_grid(
    x_max: float = X_MAX,
    y_max: float = Y_MAX,
    step_xy: float = STEP_XY,
    num_angles: int = NUM_ANGLES,
    v_max: float = V_MAX,
    w_max: float = W_MAX,
    samples_per_class: int = SAMPLES_PER_CLASS
) -> Tuple[List[List[float]], Dict[int, int]]:
    """
    Gera a amostragem em grade com passo frequente e uniforme, e equilibra perfeitamente
    as 9 classes canônicas da ENIB para que cada uma tenha exatamente 'samples_per_class' amostras.
    Garante também a inclusão das amostras exatas de parada na doca (0.0, 0.0, 0.0) -> (0.0, 0.0).
    """
    samples_by_class: Dict[int, List[List[float]]] = {i: [] for i in range(1, 10)}

    # Posições de -x_max a +x_max com passo frequente e uniforme step_xy (ex: 0.20 m)
    n_steps_x = int(round(2.0 * x_max / step_xy))
    xs = [round(-x_max + i * step_xy, 2) for i in range(n_steps_x + 1)]

    n_steps_y = int(round(2.0 * y_max / step_xy))
    ys = [round(-y_max + i * step_xy, 2) for i in range(n_steps_y + 1)]

    # 24 orientações com passo angular frequente e uniforme a cada 15° (2*pi / num_angles)
    thetas = [round(-math.pi + (i + 1) * (2.0 * math.pi / num_angles), 4) for i in range(num_angles)]

    # 1. Amostragem da grade uniforme (2601 posições x 24 orientações = 62.424 amostras)
    for x in xs:
        for y in ys:
            for th in thetas:
                v, w, m_id = expert_rule_based_controller(
                    x, y, th,
                    v_max=v_max, w_max=w_max,
                    d_brake=D_BRAKE, d_stop=D_STOP
                )
                samples_by_class[m_id].append([x, y, th, v, w])

    # 2. Amostras exatas de ancoragem e repouso na doca (0.0, 0.0, 0.0) -> (v=0.0, w=0.0) [Classe 9]
    for _ in range(samples_per_class):
        samples_by_class[9].append([0.0, 0.0, 0.0, 0.0, 0.0])

    # 3. Balanceamento estratificado perfeito (exatamente samples_per_class para cada classe)
    balanced_samples: List[List[float]] = []
    final_counts: Dict[int, int] = {}

    for m_id in range(1, 10):
        pool = samples_by_class[m_id]
        if not pool:
            continue

        if len(pool) >= samples_per_class:
            # Sub-amostragem com passo constante e regular ao longo da grade ordenada
            step_idx = len(pool) / samples_per_class
            selected = [pool[int(i * step_idx)] for i in range(samples_per_class)]
        else:
            # Re-amostragem periódica uniforme das classes minoritárias
            repeats = samples_per_class // len(pool)
            remainder = samples_per_class % len(pool)
            selected = pool * repeats + pool[:remainder]

        balanced_samples.extend(selected)
        final_counts[m_id] = len(selected)

    # 4. Embaralhamento determinístico com semente fixa para reprodutibilidade científica
    random.seed(42)
    random.shuffle(balanced_samples)

    return balanced_samples, final_counts


def build_dataset(
    output_path: str = "data/dataset.csv",
    step_xy: float = STEP_XY,
    num_angles: int = NUM_ANGLES,
    v_max: float = V_MAX,
    w_max: float = W_MAX,
    samples_per_class: int = SAMPLES_PER_CLASS
) -> int:
    """
    Função principal que constrói e grava o CSV oficial de 5 colunas perfeitamente justo.
    """
    total_expected = samples_per_class * 9
    print("\n" + "=" * 75)
    print("   LIMO OFFLINE LEARNING : GERADOR DE DATASET PERFEITAMENTE JUSTO")
    print("=" * 75)
    print(f"Arena Espacial       : [-{X_MAX:.1f}, +{X_MAX:.1f}] x [-{Y_MAX:.1f}, +{Y_MAX:.1f}] m")
    print(f"Passo Espacial (Δxy) : {step_xy:.2f} m (grade uniforme frequente)")
    print(f"Passo Angular (Δθ)   : {360.0/num_angles:.1f}° ({num_angles} direções por posição)")
    print(f"Amostras por Classe  : {samples_per_class:,} amostras para cada uma das 9 manobras")
    print(f"Total Justo Esperado : {total_expected:,} amostras (exatamente {100.0/9:.2f}% por classe)")
    print(f"Amostras (0,0,0)     : Presentes em repouso perfeito na doca (Classe 9)")
    print(f"Formato de Saída     : Estritamente 5 colunas {COLUMNS}")
    print(f"Arquivo de Destino   : {output_path}")
    print("=" * 75 + "\n")

    print("1. Gerando grade de amostragem regular e aplicando balanceamento justo 9x...")
    samples, maneuver_counts = generate_balanced_grid(
        x_max=X_MAX, y_max=Y_MAX,
        step_xy=step_xy, num_angles=num_angles,
        v_max=v_max, w_max=w_max,
        samples_per_class=samples_per_class
    )
    total_samples = len(samples)
    print(f"   -> Total de amostras geradas: {total_samples:,}")

    # Relatório de distribuição estatística das 9 manobras da ENIB
    print("\n" + "-" * 75)
    print("  DISTRIBUIÇÃO DAS 9 MANOBRAS CANÔNICAS DA ENIB (Patrick Hénaff)")
    print("-" * 75)
    for m_id in range(1, 10):
        cnt = maneuver_counts[m_id]
        pct = (cnt / max(1, total_samples)) * 100.0
        print(f"{ENIB_MANEUVER_NAMES[m_id]:<48} : {cnt:6d} amostras ({pct:5.2f}%)")
    print("-" * 75)

    # Gravação do arquivo CSV com estritamente 5 colunas
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in samples:
            writer.writerow(row)

    print(f"\n[Sucesso] Dataset justo de 5 colunas ({total_samples:,} amostras) salvo em: {output_path}\n")
    return total_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gerador de Dataset de 5 Colunas LIMO Justo (9 Manobras ENIB com 11,11% cada)"
    )
    parser.add_argument("--output", type=str, default="data/dataset.csv", help="Caminho do CSV de saída")
    parser.add_argument("--step", type=float, default=STEP_XY, help="Passo da grade em metros (padrão: 0.20)")
    parser.add_argument("--angles", type=int, default=NUM_ANGLES, help="Quantidade de orientações regulares (padrão: 24)")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS, help="Amostras por classe (padrão: 5000)")
    parser.add_argument("--v-max", type=float, default=V_MAX, help="Velocidade linear máxima (padrão: 0.50)")
    parser.add_argument("--w-max", type=float, default=W_MAX, help="Velocidade angular máxima (padrão: 1.00)")
    args = parser.parse_args()

    build_dataset(
        output_path=args.output,
        step_xy=args.step,
        num_angles=args.angles,
        samples_per_class=args.samples_per_class,
        v_max=args.v_max,
        w_max=args.w_max
    )
