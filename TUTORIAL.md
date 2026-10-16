# 📘 Manual Operacional Completo: LIMO Offline Learning & Gazebo 3D

Este documento é o guia definitivo, completo e 100% prático para executar todas as etapas do projeto **AgileX LIMO**: desde a geração do dataset e treinamento da rede neural offline até os testes no simulador Gazebo 3D e no robô físico real.

Tudo está **organizado em blocos sequenciais**, com comandos exatos prontos para copiar e colar.

---

## 📑 Sumário dos Blocos

* [Bloco 0: Preparação do Ambiente e Regra de Ouro](#-bloco-0-preparação-do-ambiente-e-regra-de-ouro)
* [Bloco 1: Geração do Dataset Canônico (com Baliza/Ré)](#-bloco-1-geração-do-dataset-canônico-com-balizaré)
* [Bloco 2: Treinamento Offline da Rede Neural (PioneerNN)](#-bloco-2-treinamento-offline-da-rede-neural-pioneernn)
* [Bloco 3: Gráficos de Aprendizado e Diagnósticos Estatísticos](#-bloco-3-gráficos-de-aprendizado-e-diagnósticos-estatísticos)
* [Bloco 4: Gazebo 3D — Teste de Navegação Individual (Single Run)](#-bloco-4-gazebo-3d--teste-de-navegação-individual-single-run)
* [Bloco 5: Gazebo 3D — Benchmark Automatizado de 20 Posições (ISO 18646-2)](#-bloco-5-gazebo-3d--benchmark-automatizado-de-20-posições-iso-18646-2)
* [Bloco 6: Terminal de Monitoramento e Diagnóstico em Tempo Real](#-bloco-6-terminal-de-monitoramento-e-diagnóstico-em-tempo-real)
* [Bloco 7: Execução no Robô Físico Real LIMO](#-bloco-7-execução-no-robô-físico-real-limo)
* [Bloco 8: Solução Imediata de Problemas (Troubleshooting)](#-bloco-8-solução-imediata-de-problemas-troubleshooting)

---

## ⚡ Bloco 0: Preparação do Ambiente e Regra de Ouro

### 0.1 Limpeza Prévia (Evita 99% das falhas no Gazebo)
Se você já abriu o simulador antes ou ele fechou inesperadamente, sempre rode este comando para matar processos fantasmas que travam as portas do ROS 2:

```bash
killall -9 gzserver gzclient
```

### 0.2 Carregamento Básico (Necessário em qualquer terminal novo)
Antes de rodar qualquer comando que envolva ROS 2, Gazebo ou o robô:

```bash
# 1. Carrega o ROS 2 Humble
source /opt/ros/humble/setup.bash

# 2. Carrega os pacotes compilados do robô LIMO
source ~/ros2_ws/install/setup.bash

# 3. Entra na pasta do projeto
cd ~/Documentos/PPE/limo-offline-learning
```

> [!NOTE]
> **Padrão de Unidades do Projeto (Norma ROS REP-103)**:
> - **Velocidade Linear ($v$)**: $\text{m/s}$ (máximo $0.50\text{ m/s}$).
> - **Velocidade Angular ($\omega$)**: sempre em $\text{rad/s}$ (máximo $1.00\text{ rad/s} \approx 57.3^\circ/\text{s}$, nunca em graus/s).
> - **Orientação e Ângulos ($\theta$)**: radianos em $[-\pi, +\pi]$.
> - **Coordenadas ($x, y$)**: metros no intervalo $[-5.0, +5.0]\text{ m}$.

---

## 🗄️ Bloco 1: Geração do Dataset Canônico (com Baliza/Ré)

*(Nota: O dataset canônico com **62.524 amostras** já vem pronto e balanceado em `data/dataset.csv`. Só rode este bloco se quiser recriá-lo do zero).*

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/generate_dataset.py
```

### O que este script faz:
1. Amostra uma grade uniforme de $-5.0\text{ m}$ a $+5.0\text{ m}$ a cada $0.5\text{ m}$.
2. Gera uma subgrade fina de atracação ao redor da doca ($|x|, |y| \le 0.6\text{ m}$).
3. **Lógica de Baliza Inteligente (Parallel Parking)**:
   - Se a doca estiver atrás ($x_{tr} < 0$) e desalinhada, gira no próprio eixo apontando a traseira para o alvo.
   - Engata marcha à ré ($v = -0.50\text{ m/s}$) e esterça coordenado até o estacionamento.
4. Aplica espelhamento bilateral sagital ($(x, y, \theta, v, \omega) \implies (x, -y, -\theta, v, -\omega)$) garantindo simetria perfeita sem viés direcional.
* **Arquivo gerado**: `data/dataset.csv` (62.524 linhas, 5 colunas).

---

## 🧠 Bloco 2: Treinamento Offline da Rede Neural (PioneerNN)

Treina a rede neural MLP bias-free (`3 -> 1000 -> 2`, ativação $\tanh$, sem bias para garantir $\text{MLP}(0,0,0) = [0, 0]$):

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/train_offline.py
```

### Configurações Ativas no Código:
* **Otimizador**: SGD Puro Constante (`USE_ADAMW = False`, `LEARNING_RATE = 0.001`).
* **Épocas**: 150 épocas completas.
* **Função de Perda**: MSE ponderada com foco na precisão de ângulo.

### Arquivos Salvos e Organização por Timestamp:
A cada treino, o script cria automaticamente uma pasta rastreável com a data e hora exata:
`models/training_yyyy_mm_dd_sec_min_hr/`
* `training_params.json` — Arquivo JSON completo contendo: taxa de aprendizado (`lr`), otimizador (`AdamW`/`SGD`), número de épocas, perdas finais, `weight_decay`, agendador e caminhos dos artefatos.
* `pioneer_model_weights.json` — Pesos da rede neural salvos e organizados.
* `supervised_w_torch_diff.json` — Cópia espelhada dos pesos para integração direta.
* `training_history.json` — Histórico detalhado de perdas de treino e validação a cada época.
* `training_curves.png` e `training_curves.svg` — Gráficos vetoriais de perda e convergência daquele treinamento.

*(Nota: Para total compatibilidade com comandos diretos, os pesos também são sincronizados na raiz `models/`).*

---

## 📈 Bloco 3: Gráficos de Aprendizado e Diagnósticos Estatísticos

Execute estes comandos a partir da raiz do projeto (`cd ~/Documentos/PPE/limo-offline-learning`):

### 3.1 Gerar Curva de Convergência (Perda MSE ao longo das 150 épocas)
```bash
python3 src/plot_curves.py
```
* **Salva em**: `data/evaluation/training_curves.png` e `training_curves.svg`.

### 3.2 Matrizes de Confusão e Análise 2D do Dataset
Verifica a acurácia de classificação discreta nas 62.524 amostras:
```bash
python3 src/evaluation/plot_confusion_and_dataset_analysis.py
```
* **Gera**:
  * Matriz de confusão linear 3x3 (Ré, Parado, Frente).
  * Matriz de confusão angular 3x3 (Horário, Reto, Anti-horário).
  * Densidade 2D do dataset e histogramas de predição.

### 3.3 Campo Vetorial 2D (Quiver) e Perfis Axiais
Mapeia o campo contínuo de setas de direção em toda a arena de $5\times 5\text{ m}$:
```bash
python3 src/evaluation/evaluate_xy_plane.py
```

---

## 🏎️ Bloco 4: Gazebo 3D — Teste de Navegação Individual (Single Run)

Use quando quiser ver o robô saindo de uma posição específica e manobrando no Gazebo 3D até a doca `(0, 0, 0)`.

### 🖥️ Terminal 1 — Abrir o Simulador Gazebo 3D
```bash
# 1. Carregar ROS 2 e o LIMO
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 2. Lançar a simulação com o modelo diferencial
ros2 launch limo_description gazebo_models_diff.launch.py
```

> ⚠️ **Atenção no Gazebo**:
> Espere a janela 3D abrir com o chão cinza e o robô no centro `(0,0,0)`.
> Verifique se o botão no canto inferior esquerdo está em **Play** (se estiver pausado, clique nele ou aperte a **barra de espaço**).

---

### 🖥️ Terminal 2 — Executar o Controlador da Rede Neural
Abra um **segundo terminal** (mantenha o Terminal 1 aberto rodando o Gazebo):

```bash
# 1. Carregar ambiente
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 2. Entrar na pasta do projeto
cd ~/Documentos/PPE/limo-offline-learning

# 3. Teletransportar o robô para o ponto inicial e navegar até a doca (0,0,0)
python3 src/evaluation/run_autonomous.py \
    --weights models/pioneer_model_weights.json \
    --teleport-to 2.0 2.0 0.0 \
    --target 0.0 0.0 0.0
```

#### 💡 Exemplos de Outras Posições para Testar (Só mudar o `--teleport-to`):
* **Frente longa:** `--teleport-to 3.0 0.0 0.0`
* **Lateral oblíqua:** `--teleport-to 1.5 2.0 0.0`
* **Baliza / Ré (Parallel Parking):** `--teleport-to 1.5 0.8 0.0` *(o robô gira, dá ré e estaciona)*
* **Ré pura atrás da doca:** `--teleport-to -2.0 0.0 0.0`

Ao terminar a manobra ou ser detectada a parada do robô (corte imediato), uma pasta com a data e hora exata do teste é criada:
`data/evaluation/test_yyyy_mm_dd_sec_min_hr/`
* `params.json` — Arquivo com todos os parâmetros do teste (ponto de partida, pose final, erro de distância, erro angular, limites cinemáticos, tempo decorrido, etc.).
* Subpasta `data/` — Contém o arquivo CSV de telemetria completa (`eval_success.csv` ou `eval_stop.csv`).
* Subpasta `images/` — Contém o gráfico individual da trajetória gerado (`eval_success.png` ou `eval_stop.png`).

---

## 🏆 Bloco 5: Gazebo 3D — Benchmark Automatizado de 20 Posições (ISO 18646-2)

Neste modo, o robô é teletransportado **automaticamente** para 20 posições diferentes cobrindo os 4 quadrantes (distâncias curta, média e longa, com vários ângulos), executa as manobras no Gazebo e calcula as métricas oficiais da norma ISO 18646-2.

### 🖥️ Terminal 1 — Abrir o Gazebo 3D
Se já estiver aberto do Bloco 4, mantenha-o aberto. Se não:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch limo_description gazebo_models_diff.launch.py
```

---

### 🖥️ Terminal 2 — Iniciar o Benchmark Automatizado de 20 Posições
```bash
# 1. Carregar ambiente e ir ao projeto
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/Documentos/PPE/limo-offline-learning

# 2. Executar o benchmark com a flag --gazebo OBRIGATÓRIA
python3 src/evaluation/run_batch_benchmark.py \
    --gazebo \
    --suite extended \
    --weights models/pioneer_model_weights.json
```

> [!IMPORTANT]
> **A flag `--gazebo` é OBRIGATÓRIA para rodar no simulador 3D!**
> Se você omitir `--gazebo` (ou usar `--sim`), o script rodará em simulação cinemática matemática instantânea a 20 Hz.
>
> **Corte Antecipado (Stall Detection)**:
> Assim que o carrinho para de se mover na arena, a simulação daquele ensaio é cortada imediatamente (não espera 60 segundos à toa).

#### 📊 Estrutura Automática do Teste (`data/evaluation/test_yyyy_mm_dd_sec_min_hr/`):
A cada bateria de testes, uma pasta com o timestamp exato é criada:
* `params.json` — JSON completo com: modelo testado, limites cinemáticos ($V_{\max}$, $\Omega_{\max}$), tolerâncias, configuração do stall detection, métricas da norma ISO 18646-2 ($A_p, A_o, R_p, R_o$) e o resumo detalhado de cada um dos ensaios.
* Subpasta `data/` — Contém todos os CSVs:
  * `run_1_q1.csv` ... `run_20_q4.csv` — Telemetria ponto a ponto de cada ensaio.
  * `iso_18646_metrics.csv` — Tabela oficial das métricas consolidadas.
* Subpasta `images/` — Contém todos os gráficos e relatórios visuais:
  * `run_1_q1.png` ... `run_20_q4.png` — Gráficos individuais 3x3 de cada corrida.
  * `benchmark_trajectory.png` — Imagem com todas as trajetórias sobrepostas na arena.
  * `benchmark_all.png` — Painel com subgráficos de telemetria completa.
  * `benchmark_quadrants.png` — Trajetórias divididas pelos Quadrantes 1, 2, 3 e 4.
  * `benchmark_trajectory.svg` — Versão vetorial pura SVG da arena.
  * `benchmark_dashboard.html` — Painel interativo para abrir no navegador.

*(Nota: Para máxima compatibilidade com ferramentas legadas, os resultados também são espelhados em `data/evaluation/batch_run/`).*

---

## 🔍 Bloco 6: Terminal de Monitoramento e Diagnóstico em Tempo Real (Terminal 3 - Opcional)

Se você quiser abrir um terceiro terminal para espionar os tópicos do robô enquanto ele se move no Gazebo:

```bash
source /opt/ros/humble/setup.bash

# 1. Conferir a frequência da odometria (deve marcar ~30-50 Hz)
ros2 topic hz /odom

# 2. Ver as coordenadas (X, Y) e ângulo do robô em tempo real
ros2 topic echo /odom --once

# 3. Ver os comandos de velocidade enviados pela rede neural
ros2 topic echo /cmd_vel

# 4. Parada de emergência manual (forçar parada imediata dos motores)
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" --once
```

---

## 🤖 Bloco 7: Execução no Robô Físico Real LIMO

Para rodar a rede neural embarcada no próprio computador do robô AgileX LIMO real (Intel NUC ou Jetson Nano):

```bash
# 1. Carregar ambiente do ROS 2 no robô
source /opt/ros/humble/setup.bash
cd ~/Documentos/PPE/limo-offline-learning

# 2. Executar com a flag de proteção de hardware (--real)
python3 src/evaluation/run_autonomous.py \
    --weights models/pioneer_model_weights.json \
    --target 0.0 0.0 0.0 \
    --real
```

> A flag `--real` ativa as travas de segurança de torque, impedindo comandos bruscos que possam danificar as caixas de redução das rodas.

---

## 🚨 Bloco 8: Solução Imediata de Problemas (Troubleshooting)

| Sintoma Observado | Causa Mais Provável | O que fazer para resolver |
| :--- | :--- | :--- |
| **O robô não se mexe no Gazebo** | O simulador abriu com a física pausada | Aperte a **barra de espaço** ou clique no botão **Play** no canto inferior esquerdo do Gazebo. |
| **O teste rodou em 2 segundos e nada aconteceu no Gazebo** | Faltou a flag `--gazebo` no comando | Sempre inclua `--gazebo`: `python3 src/evaluation/run_batch_benchmark.py --gazebo ...` |
| **`Package 'limo_description' not found`** | O workspace do robô não foi carregado | Rode `source ~/ros2_ws/install/setup.bash` no terminal atual. |
| **`ModuleNotFoundError: No module named 'rclpy'`** | O ROS 2 Humble não foi carregado | Rode `source /opt/ros/humble/setup.bash` no terminal atual. |
| **Gazebo trava na inicialização ou tela fica cinza** | Processos antigos órfãos estão presos na memória | Rode `killall -9 gzserver gzclient` e tente abrir o Terminal 1 novamente. |
| **`unrecognized arguments` no `run_autonomous.py`** | Argumento antigo digitado | O comando aceita `--teleport-to X Y TH` e `--target X Y TH` (nunca use `--x-target`). |
| **`Failed to import torch`** | Ambiente Python sem acesso aos pacotes locais | Todos os scripts possuem detecção automática de caminhos locais e snap, mas se necessário rode no venv correto: `source ~/ros2_ws/venv/bin/activate`. |
