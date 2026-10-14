# End-to-End Operational Tutorial: AgileX LIMO Offline Learning & ISO 18646-2 Benchmarking

**Project**: LIMO Offline Learning — Autonomous Docking via Behavioral Cloning  
**Academic Context**: Master's Capstone / PPE — CERV (European Centre for Virtual Reality) / ENIB (National Engineering School of Brest)  
**Author**: Julia Pivato  
**Platform**: AgileX LIMO Mobile Robot (Four-Wheel Differential Mode) under ROS 2 & PyTorch  

---

## Table of Contents

1. [System Prerequisites & Environment Setup](#1-system-prerequisites--environment-setup)
2. [Workspace Directory Structure & Navigation (`cd`)](#2-workspace-directory-structure--navigation-cd)
3. [Step 1: Dataset Generation (Canonical 70,664 Samples)](#3-step-1-dataset-generation-canonical-70664-samples)
4. [Step 2: Neural Network Offline Training (Pure Constant SGD)](#4-step-2-neural-network-offline-training-pure-constant-sgd)
5. [Step 3: Training Convergence Curve Visualization](#5-step-3-training-convergence-curve-visualization)
6. [Step 4: Confusion Matrices & Dataset Diagnostics](#6-step-4-confusion-matrices--dataset-diagnostics)
7. [Step 5: Cartesian Axes (X & Y) and 2D Plane Field Evaluation](#7-step-5-cartesian-axes-x--y-and-2d-plane-field-evaluation)
8. [Step 6: Closed-Loop Benchmarking (20 Initial Poses under ISO 18646-2)](#8-step-6-closed-loop-benchmarking-20-initial-poses-under-iso-18646-2)
   - [Mode A: High-Speed Pure Simulation (Standalone 20 Hz)](#mode-a-high-speed-pure-simulation-standalone-20-hz)
   - [Mode B: Gazebo 3D Simulation with Automated Teleportation](#mode-b-gazebo-3d-simulation-with-automated-teleportation)
9. [Step 7: Physical Hardware Deployment (AgileX LIMO)](#9-step-7-physical-hardware-deployment-agilex-limo)
10. [Master Artifacts & Output File Location Reference](#10-master-artifacts--output-file-location-reference)

---

## 1. System Prerequisites & Environment Setup

### 1.1. Python Dependencies
Ensure Python 3.10+ is installed with the required tensor and plotting libraries:

```bash
# Navigate to your home directory or workspace
cd ~/Documentos/PPE/limo-offline-learning

# Install Python dependencies
pip install torch numpy matplotlib
```

*(Note: If you run within an environment where PyTorch is installed under snap or local paths, the codebase automatically detects and prioritizes those paths).*

### 1.2. ROS 2 Environment (Required for Gazebo and Physical Robot)
Ensure ROS 2 (Humble or Foxy) and the AgileX LIMO Gazebo packages are sourced in your bash shell:

```bash
# Source standard ROS 2 setup
source /opt/ros/humble/setup.bash

# Source LIMO simulation workspace (adjust path if your workspace is named differently)
source ~/limo_ws/install/setup.bash
```

---

## 2. Workspace Directory Structure & Navigation (`cd`)

All Python scripts in this pipeline are designed to be executed from the **root directory of the project**:

```bash
cd ~/Documentos/PPE/limo-offline-learning
```

### Key Folders and File Destinations:

| Directory | Content Description |
| :--- | :--- |
| `src/` | Python source code (training, dataset generation, evaluation nodes) |
| `src/core/` | PioneerNN neural architecture definition (`pioneer_nn.py`) |
| `src/evaluation/` | Benchmarking scripts, data loggers, visualizers |
| `src/ros2/` | ROS 2 interface for `/odom`, `/cmd_vel`, and Gazebo teleportation |
| `data/` | Canonical dataset (`dataset.csv`) |
| `data/evaluation/` | CSV telemetry logs, benchmark runs, ISO metrics reports |
| `models/` | Trained model weights (`.json`) and training history (`.json`) |
| `docs/images/` | All publication-ready high-resolution plots (300 DPI PNGs) |

---

## 3. Step 1: Dataset Generation (Canonical 70,664 Samples)

*(Note: A validated dataset is already provided at `data/dataset.csv`. You only need to run this step if you want to regenerate it from scratch).*

### Terminal Command:
```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/generate_dataset.py --output data/dataset.csv --x-max 5.0 --y-max 5.0 --step-xy 0.5
```

### What this script does:
1. Samples an operational grid from $-5.0\text{ m}$ to $+5.0\text{ m}$ at $0.5\text{ m}$ intervals.
2. Generates a fine-density docking subgrid around the dock origin ($|x|, |y| \le 0.6\text{ m}$).
3. Executes 20 Hz closed-loop trajectories using pure unicycle differential kinematics.
4. Enforces sagittal bilateral mirroring ($(x, y, \theta, v, \omega) \implies (x, -y, -\theta, v, -\omega)$) ensuring zero directional bias.

### Saved Output:
- `data/dataset.csv` (70,664 rows, 5 columns: `x_target_robot`, `y_target_robot`, `theta_robot`, `v_robot`, `w_robot`).

---

## 4. Step 2: Neural Network Offline Training (Pure Constant SGD)

The training script [`src/train_offline.py`](src/train_offline.py) has been streamlined into a single, clean PyTorch pipeline.

### 4.1. Changing Hyperparameters in the Code
You do not need to pass long arguments in the terminal. Open [`src/train_offline.py`](src/train_offline.py) and modify the global variables at the top of the file:

```python
# ==============================================================================
# ⚙️ GLOBAL TRAINING CONFIGURATION (MODIFY DIRECTLY HERE)
# ==============================================================================
EPOCHS = 150              # Training epochs (e.g. 30, 60, 100, 150)
LEARNING_RATE = 0.001     # Learning rate (Pure constant SGD, no scheduler)
BATCH_SIZE = 256          # Mini-batch size
BETA = 1.0                # Angular loss weighting: MSE(v) + beta * MSE(w)
HIDDEN_SIZE = 1000        # Hidden layer width (Tanh activation)
USE_BIAS = False          # False = Strict mathematical guarantee of v=0 at dock
# ==============================================================================
```

### 4.2. Running the Training:
```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/train_offline.py
```

*(Alternatively, you can override any parameter from the CLI)*:
```bash
python3 src/train_offline.py --epochs 150 --lr 0.001 --batch-size 256
```

### Saved Outputs:
- Model weights (JSON format): `models/supervised_w_torch_diff.json`
- Loss history (JSON format): `models/training_history.json`
- Automatic learning curve plot: `data/evaluation/training_curves.png` & `.svg`

---

## 5. Step 3: Training Convergence Curve Visualization

To manually regenerate or inspect the training and validation loss curves at any time:

### Terminal Command:
```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/plot_curves.py --history models/training_history.json --outdir data/evaluation
```

### Saved Outputs:
- High-resolution plot (300 DPI): `data/evaluation/training_curves.png`
- Vector plot for papers/web: `data/evaluation/training_curves.svg`

---

## 6. Step 4: Confusion Matrices & Dataset Diagnostics

To evaluate the neural network against all 70,664 expert actions and verify discrete classification accuracy ($v \in \{-0.24, 0.0, 0.24\}$, $\omega \in \{-0.50, 0.0, 0.50\}$):

### Terminal Command:
```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/plot_confusion_and_dataset_analysis.py
```

### Saved Outputs in `docs/images/`:
1. `docs/images/matriz_confusao_velocidade_linear.png` (3x3 Confusion Matrix: Reverse, Stop, Forward).
2. `docs/images/matriz_confusao_velocidade_angular.png` (3x3 Confusion Matrix: Clockwise, Straight, Counter-clockwise).
3. `docs/images/distribuicao_espacial_dataset.png` (2D hexbin density map of 70,664 samples across the $5\times 5\text{ m}$ arena).
4. `docs/images/histogramas_predicao_vs_real.png` (Side-by-side distribution histograms: discrete targets vs continuous $\tanh$ outputs).
5. `data/evaluation/confusion_metrics_summary.json` (Numerical accuracy, recall, and precision summary).

---

## 7. Step 5: Cartesian Axes (X & Y) and 2D Plane Field Evaluation

To evaluate the policy when starting strictly along the principal axes ($y=0$ for X, $x=0$ for Y) and map the entire continuous vector field:

### Terminal Command:
```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/evaluate_xy_plane.py
```

### Saved Outputs in `docs/images/`:
1. `docs/images/trajetorias_testes_eixos_x_y.png` (Closed-loop trajectory overlays starting at $\pm 3\text{ m}$ and $\pm 2\text{ m}$ on X and Y).
2. `docs/images/telemetria_testes_eixos_x_y.png` (Control commands $v(t)$ and $\omega(t)$ over time during axial approaches).
3. `docs/images/campo_vetorial_quiver_2d.png` (**2D Quiver Vector Field Flow**, showing directional navigation arrows across the arena).
4. `docs/images/heatmap_velocidades_2d.png` (2D heatmaps of linear velocity $v(x, y)$ and angular velocity $\omega(x, y)$).
5. `docs/images/perfil_resposta_eixos_x_y.png` (1D slice profiles of $v$ and $\omega$ along $y=0$ and $x=0$).

---

## 8. Step 6: Closed-Loop Benchmarking (20 Initial Poses under ISO 18646-2)

The benchmark evaluates 20 systematic test poses covering all 4 quadrants, 3 distance scales (Short $\sim 1.2\text{ m}$, Medium $\sim 2.5\text{ m}$, Long $\sim 3.5\text{ m}$), and multiple orientations ($0^\circ, \pm 45^\circ, \pm 90^\circ, \pm 135^\circ, 180^\circ$).

### Mode A: High-Speed Pure Simulation (Standalone 20 Hz)
Runs directly without requiring Gazebo or active ROS 2 masters:

```bash
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/run_batch_benchmark.py --weights models/supervised_w_torch_diff.json --suite extended --sim --outdir data/evaluation/benchmark_20_extended
```

### Mode B: Gazebo 3D Simulation with Automated Teleportation
Runs the 20 test scenarios inside Gazebo using ROS 2 `/cmd_vel` and `/odom`. The robot is automatically teleported to each initial pose at the start of each run.

#### Terminal 1 — Start Gazebo Simulation:
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash
source ~/limo_ws/install/setup.bash

# Launch the AgileX LIMO 4WD world
ros2 launch limo_gazebo limo_4wd.launch.py
```
*(Wait until Gazebo is fully loaded and `/odom` is publishing).*

#### Terminal 2 — Launch Automated Benchmark:
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash
source ~/limo_ws/install/setup.bash

# Navigate to project and launch
cd ~/Documentos/PPE/limo-offline-learning
python3 src/evaluation/run_batch_benchmark.py --weights models/supervised_w_torch_diff.json --suite extended --gazebo --outdir data/evaluation/gazebo_20_benchmark
```

### What Happens Automatically in Mode B:
1. The script calls the Gazebo entity state reset (`gz model -m limo -x ... -y ... -Y ...`) to teleport the vehicle to Scenario 1.
2. It pauses $350\text{ ms}$ for physics stabilization and recalibrates odometry.
3. PioneerNN drives the robot in closed loop at 20 Hz via `/cmd_vel`.
4. When the docking tolerance is reached (or 60s timeout), the robot stops, logs telemetry, and repeats for Scenarios 2 through 20.
5. At the end of the 20 runs, ISO 18646-2 metrics and all comparison plots are generated automatically.

### Saved Outputs (in the specified `--outdir` folder):
- `benchmark_trajectory.png`: Overlay of all 20 trajectories on an 8x8 square plane with start pins and dock star.
- `benchmark_all.png`: Comprehensive 7-subplot dashboard (trajectories, cumulative cost, linear/angular velocity, position errors $X, Y$, heading error $\theta$).
- `benchmark_quadrants.png`: Trajectories grouped by Quadrants 1, 2, 3, and 4.
- `iso_18646_metrics.csv`: Formal metrological table with $A_p$ (Position Accuracy), $A_o$ (Orientation Accuracy), $R_p$ (Repeatability), and $R_o$.
- `run_1_q1.png` to `run_20_q4.png`: Detailed 3x3 telemetry breakdown for each individual run.
- `benchmark_dashboard.html`: Interactive web dashboard.

---

## 9. Step 7: Physical Hardware Deployment (AgileX LIMO)

To deploy the trained model directly on the physical AgileX LIMO robot (onboard Intel NUC / Jetson Nano):

### Terminal Command on the Physical Robot:
```bash
cd ~/Documentos/PPE/limo-offline-learning

# Launch single-goal autonomous navigation (Target: x=0.0, y=0.0, theta=0.0)
python3 src/evaluation/run_autonomous.py \
    --weights models/supervised_w_torch_diff.json \
    --is-real \
    --target-x 0.0 --target-y 0.0 --target-theta 0.0
```

The `--is-real` flag enforces hardware safety velocity limits ($V_{\max} = 0.24\text{ m/s}$, $\Omega_{\max} = 0.50\text{ rad/s}$) to protect the gearbox and wheel hub motors.

---

## 10. Master Artifacts & Output File Location Reference

| Category | File Path | Format | Description |
| :--- | :--- | :---: | :--- |
| **Model Weights** | `models/supervised_w_torch_diff.json` | JSON | Deployed neural network weights |
| **Training History** | `models/training_history.json` | JSON | Per-epoch train/val loss logs |
| **Learning Curve** | `data/evaluation/training_curves.png` | PNG | Training vs validation MSE curve |
| **Linear Confusion** | `docs/images/matriz_confusao_velocidade_linear.png` | PNG | 3x3 classification matrix for $v$ |
| **Angular Confusion**| `docs/images/matriz_confusao_velocidade_angular.png`| PNG | 3x3 classification matrix for $\omega$ |
| **Dataset Space** | `docs/images/distribuicao_espacial_dataset.png` | PNG | 2D spatial sampling coverage |
| **Prediction Hists** | `docs/images/histogramas_predicao_vs_real.png` | PNG | Target vs output distributions |
| **Vector Field** | `docs/images/campo_vetorial_quiver_2d.png` | PNG | 2D navigation quiver flow |
| **Heatmaps** | `docs/images/heatmap_velocidades_2d.png` | PNG | 2D velocity heatmaps |
| **Axis Profiles** | `docs/images/perfil_resposta_eixos_x_y.png` | PNG | 1D response slices on $y=0$ and $x=0$ |
| **Axial Trajectories**| `docs/images/trajetorias_testes_eixos_x_y.png` | PNG | Closed-loop runs on X & Y axes |
| **20-Run Trajectory**| `data/evaluation/benchmark_20_extended/benchmark_trajectory.png` | PNG | 20 benchmark trajectories overlay |
| **20-Run Dashboard** | `data/evaluation/benchmark_20_extended/benchmark_all.png` | PNG | Full 7-subplot telemetry dashboard |
| **ISO 18646-2 Report**| `data/evaluation/benchmark_20_extended/iso_18646_metrics.csv` | CSV | $A_p, A_o, R_p, R_o$ metrology metrics |
