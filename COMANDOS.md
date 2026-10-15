# 📜 Guia Geral de Comandos em Ordem Cronológica (LIMO Offline Learning)

Este documento contém **TODOS os comandos do projeto organizados na ordem exata de execução**, desde a preparação do ambiente até a simulação no Gazebo 3D e a geração de todos os gráficos para o relatório.

---

## 📋 Sumário Rápido dos Passos

1. [Passo 0: Carregar o Ambiente](#passo-0-carregar-o-ambiente-em-qualquer-terminal)
2. [Passo 1: Dataset de Treinamento (62.524 amostras)](#passo-1-dataset-de-treinamento)
3. [Passo 2: Treinar a Rede Neural (150 épocas, lr=0.001, SGD)](#passo-2-treinar-a-rede-neural-offline)
4. [Passo 3: Gráfico da Curva de Perda (Loss MSE)](#passo-3-gerar-gráfico-da-curva-de-perda)
5. [Passo 4: Matrizes de Confusão e Análise do Dataset](#passo-4-matrizes-de-confusão-e-análise-do-dataset)
6. [Passo 5: Teste nos Eixos X e Y + Campo Vetorial 2D (Quiver)](#passo-5-teste-nos-eixos-cartesianos-x-e-y--campo-vetorial-2d)
7. [Passo 6: Benchmark ISO 18646-2 Rápido (Sem abrir Gazebo)](#passo-6-benchmark-iso-18646-2-rápido-em-malha-fechada)
8. [Passo 7: Simulação Completa no Gazebo 3D com Teletransporte (20 Posições)](#passo-7-simulação-3d-no-gazebo-com-teletransporte)
9. [Passo 8: Gerar o Painel Consolidado das 20 Trajetórias ISO](#passo-8-gerar-o-painel-consolidado-das-20-trajetórias)
10. [Tabela Mestra: Onde Fica Salvo Cada Arquivo e Gráfico](#-tabela-mestra-onde-fica-salvo-cada-arquivo-e-gráfico)
11. [Comandos Úteis de Apoio e Socorro (Troubleshooting)](#-comandos-úteis-de-apoio-e-socorro-troubleshooting)

---

## Passo 0: Carregar o Ambiente (Em qualquer terminal)

Sempre que abrir um novo terminal para rodar coisas do ROS 2 ou do robô, execute:

```bash
# Carrega o ROS 2 Humble e o Workspace do robô LIMO
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Entra na pasta do projeto
cd ~/Documentos/PPE/limo-offline-learning
```

> [!NOTE]
> **Padrão de Unidades do Projeto (Norma ROS REP-103)**:
> - **Velocidade Angular ($\omega$)**: sempre em **$\text{rad/s}$ (radianos por segundo)**. $\Omega_{\max} = 1.00\text{ rad/s} \approx 57.30^\circ/\text{s}$.
> - **Velocidade Linear ($v$)**: em **$\text{m/s}$ (metros por segundo)**. $V_{\max} = 0.50\text{ m/s}$.
> - **Orientação e Ângulos ($\theta$)**: em **$\text{rad}$ (radianos)** no intervalo $[-\pi, +\pi]$.
> - **Coordenadas e Distâncias ($x, y, d$)**: em **$\text{m}$ (metros)**.

---

## Passo 1: Dataset de Treinamento

> *Nota: O dataset canônico com **62.524 amostras** já está salvo e balanceado em `data/dataset.csv`. Só rode este passo se você quiser regerá-lo do zero.*

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/generate_dataset.py
```

* **Entrada**: Regras canônicas de pilotagem (9 manobras, espelhamento sagital, subgrade de accostage fino).
* **Arquivo gerado**: `data/dataset.csv`

---

## Passo 2: Treinar a Rede Neural Offline

Executa o treinamento supervisionado (*Behavioral Cloning*) com **150 épocas**, **lr = 0.001**, **SGD puro** (sem AdamW) e função de ativação $\tanh$.

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/train_offline.py
```

> **Dica**: Para mudar hiperparâmetros (épocas, lr, batch), basta editar o topo do arquivo [`src/train_offline.py`](file:///home/juliapivato/Documentos/PPE/limo-offline-learning/src/train_offline.py#L56-L65) nas variáveis globais `EPOCHS`, `LEARNING_RATE`, etc.

* **Arquivos gerados**:
  * `models/pioneer_model_weights.json` (Pesos salvos em JSON)
  * `models/supervised_w_torch_diff.json` (Cópia sincronizada)
  * `models/training_history.json` (Histórico numérico da perda época por época)

---

## Passo 3: Gerar Gráfico da Curva de Perda

Gera os gráficos de alta resolução da evolução da função de perda (MSE):

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/plot_curves.py
```

* **Arquivos gerados**:
  * `docs/images/loss_curve_150_epochs.png` (Curva PNG em 300 DPI destacando o início e a convergência)
  * `data/evaluation/training_curves.png`
  * `data/evaluation/training_curves.svg` (Vetor SVG para relatórios LaTeX/web)

---

## Passo 4: Matrizes de Confusão e Análise do Dataset

Avalia a rede contra todas as **62.524 amostras** do dataset e gera diagnósticos estatísticos completos:

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/plot_confusion_and_dataset_analysis.py
```

* **Arquivos gerados em `docs/images/`**:
  * `matriz_confusao_velocidade_linear.png` (Matriz 3x3: Ré, Parado, Frente)
  * `matriz_confusao_velocidade_angular.png` (Matriz 3x3: Horário, Reto, Anti-horário)
  * `distribuicao_espacial_dataset.png` (Mapa de densidade 2D hexbin do dataset na arena de $5\times 5\text{ m}$)
  * `histogramas_predicao_vs_real.png` (Distribuições discretas reais vs saídas contínuas da rede)
  * `data/evaluation/confusion_metrics_summary.json` (Acurácia, precisão e recall)

---

## Passo 5: Teste nos Eixos Cartesianos X e Y + Campo Vetorial 2D

Testa como o modelo reage quando o robô é colocado estritamente em cima do eixo X ($y=0$) e em cima do eixo Y ($x=0$), além de gerar o campo de vetores 2D (*quiver*):

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/evaluate_xy_plane.py
```

* **Arquivos gerados em `docs/images/`**:
  * `trajetorias_testes_eixos_x_y.png` (Trajetórias em malha fechada partindo de $\pm 3\text{ m}$ e $\pm 2\text{ m}$ em X e Y)
  * `telemetria_testes_eixos_x_y.png` (Evolução temporal das velocidades $v(t)$ e $\omega(t)$ nos eixos)
  * `campo_vetorial_quiver_2d.png` (**Campo de fluxo 2D com setas direcionais** cobrindo toda a arena)
  * `heatmap_velocidades_2d.png` (Mapas de calor da velocidade linear e angular)
  * `perfil_resposta_eixos_x_y.png` (Corte 1D de resposta de velocidade ao longo de $y=0$ e $x=0$)

---

## Passo 6: Benchmark ISO 18646-2 Rápido (Em Malha Fechada)

Executa os **20 cenários de teste sistemáticos** cobrindo os 4 quadrantes em modo simulação matemática a 20 Hz (termina em ~2 segundos, sem precisar abrir o Gazebo):

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/run_batch_benchmark.py --sim --suite extended --outdir data/evaluation/batch_run
```

* **Arquivos gerados**:
  * `data/evaluation/batch_run/iso_18646_metrics.csv` (Métricas normativas: $A_p, A_o, R_p, R_o$)
  * `data/evaluation/batch_run/run_1_q1.csv` a `run_20_q4.csv` (Telemetria individual de cada corrida)
  * `data/evaluation/batch_run/run_1_q1.png` a `run_20_q4.png` (Gráficos individuais de cada corrida)

---

## Passo 7: Simulação 3D no Gazebo com Teletransporte

Para rodar com a física real do robô no simulador 3D (atrito das rodas, deslizamento, inércia):

### 🖥️ Aba 1: Iniciar o Simulador Gazebo
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch limo_description gazebo_models_diff.launch.py
```
*(Aguarde o Gazebo 3D abrir com o robô parado no centro na posição `0,0,0`)*

---

### 🧪 Aba 2 (Opção Principal): Rodar as 20 Posições com Teletransporte Automático
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/run_batch_benchmark.py --gazebo --suite extended --outdir data/evaluation/batch_run
```
*(O robô vai se teletransportar automaticamente para cada uma das 20 posições nos 4 quadrantes, navegar até a vaga, estacionar, salvar os dados e ir para a próxima sozinho).*

---

### 🎯 Aba 2 (Opção Alternativa): Testar Apenas 1 Posição Específica (Single Run)
Se quiser testar apenas 1 corrida partindo de uma pose customizada (ex: $X=2.0, Y=2.0, \theta=0.0$):
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/run_autonomous.py --teleport-to 2.0 2.0 0.0 --target 0.0 0.0 0.0
```

---

## Passo 8: Gerar o Painel Consolidado das 20 Trajetórias

Após rodar o benchmark (no Gazebo ou em simulação), execute o visualizador para gerar o painel de sobreposição:

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/benchmark_visualizer.py
```

* **Arquivos gerados em `docs/images/`**:
  * `iso_benchmark_20_runs.png` (As 20 trajetórias sobrepostas convergindo para a doca com pinos de início coloridos)
  * `benchmark_quadrants.png` (Trajetórias separadas por quadrantes 1, 2, 3 e 4)
  * `benchmark_all.png` (Painel com 7 subplots de telemetria completa)

---

## 📁 Tabela Mestra: Onde Fica Salvo Cada Arquivo e Gráfico

| O que é | Arquivo / Caminho no seu Computador |
| :--- | :--- |
| **Pesos da Rede Neural (JSON)** | `models/pioneer_model_weights.json` |
| **Histórico da Perda (JSON)** | `models/training_history.json` |
| **Tabela Normativa ISO 18646-2** | `data/evaluation/batch_run/iso_18646_metrics.csv` |
| **Curva de Perda (Loss)** | `docs/images/loss_curve_150_epochs.png` |
| **Matriz Confusão Linear (v)** | `docs/images/matriz_confusao_velocidade_linear.png` |
| **Matriz Confusão Angular (w)** | `docs/images/matriz_confusao_velocidade_angular.png` |
| **Densidade 2D do Dataset** | `docs/images/distribuicao_espacial_dataset.png` |
| **Histogramas de Predição** | `docs/images/histogramas_predicao_vs_real.png` |
| **Trajetórias nos Eixos X e Y** | `docs/images/trajetorias_testes_eixos_x_y.png` |
| **Campo Vetorial 2D (Quiver)** | `docs/images/campo_vetorial_quiver_2d.png` |
| **Painel de Velocidades 2D** | `docs/images/heatmap_velocidades_2d.png` |
| **Perfil 1D nos Eixos X e Y** | `docs/images/perfil_resposta_eixos_x_y.png` |
| **Painel das 20 Corridas ISO** | `docs/images/iso_benchmark_20_runs.png` |
| **Telemetria Individual (1 a 20)** | `data/evaluation/batch_run/run_1_q1.png` até `run_20_q4.png` |

---

## 🛠️ Comandos Úteis de Apoio e Socorro (Troubleshooting)

### Matar o Gazebo se ele travar ou não abrir:
```bash
killall -9 gzserver gzclient
```

### Recompilar o pacote do robô caso dê erro de launch:
```bash
cd ~/ros2_ws
colcon build --packages-select limo_description
source install/setup.bash
```

### Inspecionar os tópicos do ROS 2 durante a simulação:
```bash
source /opt/ros/humble/setup.bash

# Ver a taxa da odometria:
ros2 topic hz /odom

# Ver coordenadas atuais:
ros2 topic echo /odom --once

# Ver ordens de velocidade saindo da rede:
ros2 topic echo /cmd_vel

# Parada de emergência imediata:
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" --once
```
