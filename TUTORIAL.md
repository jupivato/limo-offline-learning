# 📘 Guia Operacional Padronizado: LIMO Offline Learning

Guia unificado e definitivo para executar o projeto **AgileX LIMO**: desde a geração do dataset e treinamento da rede neural offline até os testes no simulador Gazebo 3D e no robô físico real.

---

## 🏗️ Estrutura Simplificada do Projeto

```
limo-offline-learning/
├── src/
│   ├── core/              # Arquitetura da rede neural PioneerNN (Tanh, sem viés)
│   ├── ros2/              # Interface com hardware e tópicos (/odom, /cmd_vel)
│   ├── evaluation/        # Scripts de execução autônoma e benchmarks ISO
│   │   ├── run_autonomous.py       # Navegação individual (robô real e simulação)
│   │   ├── run_batch_benchmark.py  # Bateria estendida de 20 ensaios ISO 18646-2
│   │   └── data_logger.py          # Gravação de telemetria em CSV
│   ├── visualization/     # Central de geração de gráficos e dashboards
│   │   ├── plot_curves.py          # Curvas de convergência do treinamento
│   │   ├── benchmark_visualizer.py # Painéis ISO, trajetórias e dashboard HTML
│   │   ├── evaluate_xy_plane.py    # Campo vetorial 2D (Quiver) e heatmaps
│   │   └── plot_confusion_and_dataset_analysis.py # Matrizes de confusão
│   ├── generate_dataset.py # Gerador do dataset canônico de 45.000 amostras
│   └── train_offline.py    # Treinamento offline supervisionado da PioneerNN
├── models/                # Pesos ativos da rede neural e histórico de treino
│   ├── pioneer_model_weights.json
│   └── supervised_w_torch_diff.json
├── data/
│   ├── dataset.csv        # Dataset balanceado (5.000 amostras por classe ENIB)
│   └── evaluation/        # Telemetria e imagens de ensaios salvos
└── TUTORIAL.md            # Este manual operacional unificado
```

---

## ⚡ Passo 0: Preparação do Ambiente ROS 2

Sempre que abrir um novo terminal para executar comandos do projeto, execute:

```bash
# 1. Carrega o ROS 2 Humble
source /opt/ros/humble/setup.bash

# 2. Carrega os pacotes do robô AgileX LIMO
source ~/ros2_ws/install/setup.bash

# 3. Entra na pasta do projeto
cd ~/Documentos/PPE/limo-offline-learning
```

> [!NOTE]
> **Padrões de Unidades Físicas (Norma ROS REP-103)**:
> - **Velocidade Linear ($v$)**: $\text{m/s}$ (máximo $0.50\text{ m/s}$).
> - **Velocidade Angular ($\omega$)**: $\text{rad/s}$ (máximo $1.00\text{ rad/s} \approx 57.3^\circ/\text{s}$).
> - **Orientação e Ângulos ($\theta$)**: radianos em $[-\pi, +\pi]$.

---

## 🗄️ Passo 1: Dataset & Treinamento da Rede Neural

*(Nota: O dataset canônico com 45.000 amostras e os pesos já estão prontos. Use estes comandos se desejar recriar ou retreinar).*

### 1.1 Gerar o Dataset Canônico (45.000 amostras balanceadas)
Gera o dataset cobrindo os 4 quadrantes, espelhamento sagital e costura de fronteira periódica ($-\pi \equiv +\pi$):
```bash
python3 src/generate_dataset.py
```
* **Arquivo gerado**: `data/dataset.csv` (exatamente 5.000 amostras em cada uma das 9 classes da ENIB).

### 1.2 Treinar a Rede Neural PioneerNN
Treina a arquitetura MLP sem viés (`3 -> 1000 -> 2`, ativação $\tanh$):
```bash
python3 src/train_offline.py
```
* **Pesos salvos**: `models/pioneer_model_weights.json` e `models/supervised_w_torch_diff.json`.

---

## 🏎️ Passo 2: Simulação no Gazebo 3D (Suite Estendida de 20 Ensaios)

### 🖥️ Terminal 1 — Abrir o Simulador Gazebo
```bash
# Limpeza prévia de processos antigos
killall -9 gzserver gzclient 2>/dev/null

# Inicializar o mundo Gazebo com o modelo LIMO diferencial
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch limo_description gazebo_models_diff.launch.py
```
> Espere a janela 3D abrir com o chão cinza e o robô no centro `(0, 0, 0)`. Verifique se o botão no canto inferior esquerdo está em **Play**.

### 🖥️ Terminal 2 — Executar o Benchmark Automatizado de 20 Ensaios ISO 18646-2
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/Documentos/PPE/limo-offline-learning

python3 src/evaluation/run_batch_benchmark.py --suite extended --gazebo
```
* O robô navega automaticamente pelos 20 cenários cobrindo todos os quadrantes.
* Ao final, calcula $A_p$ (precisão posicional), $A_o$ (precisão angular) e verifica os 30 segundos contínuos de imobilização (*dwell time*).

---

## 🤖 Passo 3: Execução no Robô Físico Real LIMO

No robô real, você **não precisa digitar nenhuma coordenada manual**. O robô lê a odometria acumulada das rodas automaticamente!

### 3.1 Terminal 1 no Robô — Inicializar o Driver dos Motores
Coloque o robô parado exatamente no local onde você quer que seja a doca/meta $(0, 0, 0)$ e execute:
```bash
# Para LIMO COBOT (Foxy):
source /opt/ros/foxy/setup.bash
source ~/limo_ros2_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py

# (Ou se for LIMO Standard Humble: source /opt/ros/humble/setup.bash && source ~/agilex_ws/install/setup.bash && ros2 launch limo_base limo_base.launch.py)
```
*(O nó deve continuar rodando em segundo plano. Ele zera a odometria como $(0, 0, 0)$ e publica `/odom`).*

### 3.2 Terminal 2 no Robô — Dirigir com Teclado (Teleop)
Abra um segundo terminal no robô para afastar o robô da doca até o ponto inicial do teste:
```bash
source /opt/ros/foxy/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
**Controles no teclado:**
- `i` : Anda para frente
- `,` : Anda para trás
- `j` : Gira para a esquerda no próprio eixo
- `l` : Gira para a direita no próprio eixo
- `k` : Parar imediatamente (Stop)

Dirija o robô até onde quiser (ex: $1.5\text{ m}$ ou $2.0\text{ m}$ de distância, reto ou virado em ângulo), aperte `k` para parar e finalize com `Ctrl + C`.

### 3.3 Terminal 3 no Robô — Lançar a Navegação Autônoma (PioneerNN)
```bash
# Para LIMO COBOT (Foxy):
source /opt/ros/foxy/setup.bash
cd ~/limo-offline-learning

python3 src/evaluation/run_autonomous.py --real
```
1. O script lerá a pose atual do `/odom` automaticamente e mostrará na tela:
   ```text
   [Odometry] Current robot pose: x=1.84 m, y=0.92 m, theta=34.2°
   [Ready] Press Enter to start autonomous docking...
   ```
2. Aperte **Enter**! O PioneerNN assumirá o controle em `/cmd_vel`, fará a aproximação suave e estacionará no $(0, 0, 0)$ com imobilização de 30 segundos.

---

## 📊 Passo 4: Geração de Gráficos e Diagnósticos (`src/visualization/`)

Todos os códigos de plotagem foram centralizados na pasta `src/visualization/`:

### 4.1 Gráfico de Convergência do Treino (Perda MSE e Acurácia)
```bash
python3 src/visualization/plot_curves.py
```
* **Salva em**: `data/evaluation/training_curves.png` e `.svg`.

### 4.2 Painel Consolidado de Trajetórias e Dashboard HTML
Gera o dashboard interativo e a prancha de trajetórias a partir de qualquer pasta de testes:
```bash
python3 src/visualization/benchmark_visualizer.py --dir data/evaluation/test_2026_11_04_14_24_14/data
```
* **Gera**: `benchmark_trajectory.png`, `benchmark_all.png` e `benchmark_dashboard.html`.

### 4.3 Campo Vetorial 2D (Quiver) e Perfis Axiais
Gera o mapa de vetores da arena e a resposta de velocidade no plano $XY$:
```bash
python3 src/visualization/evaluate_xy_plane.py
```

### 4.4 Matrizes de Confusão das 9 Manobras Canônicas
Avalia as previsões discretas da rede contra o supervisor especialista nas 45.000 amostras:
```bash
python3 src/visualization/plot_confusion_and_dataset_analysis.py
```
* **Gera**: Matriz de confusão linear 3x3, matriz de confusão angular 3x3 e distribuição espacial em `docs/images/`.

---

## 🆘 Resumo de Comandos Rápidos e Troubleshooting

| Ação | Comando |
| :--- | :--- |
| **Matar processos travados do Gazebo** | `killall -9 gzserver gzclient 2>/dev/null` |
| **Parada de emergência imediata do robô** | `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" --once` |
| **Verificar frequência da odometria** | `ros2 topic hz /odom` |
| **Copiar resultados do robô para o PC** | `scp -r agilex@<IP_ROBO>:~/limo-offline-learning/data/evaluation/test_* ./` |
