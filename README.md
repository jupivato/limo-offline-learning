# LIMO Offline Learning: Behavioral Cloning & ISO 18646-2 Metrological Benchmarking

Autonomous closed-loop neural controller for the **AgileX LIMO** differential mobile robot, implementing offline behavioral cloning, robot-centric coordinate frames, discrete ternary actions, and metrological benchmarking under **ISO 18646-2**.

---

## 🏗️ Architecture & Pipeline Overview

```mermaid
flowchart TD
    subgraph Offline Dataset Generation [src/generate_dataset.py]
        Grid["Spatial Grid: -5m ≤ x,y ≤ +5m (step 0.5m) & -π ≤ θ ≤ +π"]
        Dock["Terminal Docking Subgrid (|x|, |y| ≤ 0.6m, fine steps)"]
        Rollouts["Closed-Loop 20 Hz Rollouts (Dynamic Transitions)"]
        Sym["Bilateral Sagittal Mirroring (Zero Left/Right Bias)"]
        Grid --> Sym
        Dock --> Sym
        Rollouts --> Sym
        Sym --> CSV["data/dataset.csv\n(x_target_robot, y_target_robot, theta_robot, v_robot, w_robot)"]
    end

    subgraph Neural Network Training [PioneerNN / src/train_offline.py]
        NormIn["Input Normalization into [-1, +1]:\n[x/5.0, y/5.0, θ/π]"]
        MLP["Dense Layer (3 -> 1000) [Tanh]\nOutput Layer (1000 -> 2) [Tanh]\nuse_bias=False (Strict MLP(0)=0 Guarantee)"]
        NormOut["Target Normalization:\nv/0.50 ∈ {-1, 0, 1}, w/1.00 ∈ {-1, 0, 1}"]
        Loss["SGD / AdamW Optimizer + Weighted MSE Loss"]
        CSV --> NormIn --> MLP --> Loss
        CSV --> NormOut --> Loss
        Loss --> Weights["models/pioneer_model_weights.json\nmodels/supervised_w_torch_diff.json"]
    end

    subgraph Closed-Loop Autonomous Control [ROS 2 / src/evaluation/run_autonomous.py]
        Odom["/odom Pose: (x_R, y_R, θ_R)"]
        Transform["Transform Target to Robot Frame:\nx_tr = Δx·cos(θ) + Δy·sin(θ)\ny_tr = -Δx·sin(θ) + Δy·cos(θ)"]
        Inference["PioneerNN Inference: cmd ∈ [-1, +1]²"]
        Denorm["Denormalization:\nv = cmd[0] · 0.50 m/s\nω = cmd[1] · 1.00 rad/s"]
        CmdVel["/cmd_vel (AgileX LIMO Chassis)"]
        
        Odom --> Transform --> Inference --> Denorm --> CmdVel
    end

    subgraph Metrology & Benchmarking [calculus/calculate_iso_metrics.py]
        DataLogger["20 Hz Telemetry Log\n(x, y, θ, v, ω, timestamp)"]
        ISOCalc["ISO 18646-2 Calculations:\nAp (Position Accuracy), Rp (Repeatability)\nAo (Orientation Accuracy), Ro (Repeatability)"]
        CmdVel --> DataLogger --> ISOCalc
    end
```

---

## 📂 Repository Structure

```text
limo-offline-learning/
├── calculus/
│   ├── __init__.py
│   └── calculate_iso_metrics.py         # ISO 18646-2 metrological calculation engine
├── config/
│   └── config.json                      # Spatial bounds, kinematic limits, & normalization
├── data/
│   └── dataset.csv                      # Canonical offline dataset (62,524 discrete samples)
├── docs/
│   ├── KINEMATICS_AND_NORMALIZATION_SPECIFICATION.md # Mathematical derivations & limits
│   └── TUTORIAL_GAZEBO.md               # Complete Gazebo 3D simulation guide
├── models/
│   ├── pioneer_model_weights.json       # Trained neural weights (JSON format)
│   └── supervised_w_torch_diff.json     # Synced weights mirror
├── src/
│   ├── generate_dataset.py              # Robot-centric discrete dataset generator
│   ├── train_offline.py                 # Offline training pipeline (PioneerNN)
│   ├── plot_curves.py                   # Training/validation loss curve plotter
│   ├── core/
│   │   ├── __init__.py
│   │   └── pioneer_nn.py                # PioneerNN architecture (bias-free MLP)
│   ├── ros2/
│   │   ├── __init__.py
│   │   └── limo_interface.py            # ROS 2 odometry transform & command publisher
│   └── evaluation/
│       ├── __init__.py
│       ├── data_logger.py               # Real-time 20 Hz state recorder
│       ├── run_autonomous.py            # Closed-loop autonomous controller
│       ├── run_batch_benchmark.py       # Automated multi-run testbench
│       ├── evaluate_xy_plane.py         # XY plane model evaluation
│       ├── plot_confusion_and_dataset_analysis.py # Confusion matrices and diagnostics
│       └── benchmark_visualizer.py      # Trajectory visualizer and HTML dashboards
├── requirements.txt
├── COMANDOS.md                          # Sequential terminal execution cheatsheet
├── TUTORIAL.md                          # Comprehensive end-to-end user tutorial
└── README.md
```

---

## 🔬 Kinematic Specifications & Normalization

For complete mathematical derivations, stability proofs, and friction stopping distances, see [KINEMATICS_AND_NORMALIZATION_SPECIFICATION.md](docs/KINEMATICS_AND_NORMALIZATION_SPECIFICATION.md).

> [!NOTE]
> **Units of Measure & ROS REP-103 Standard Compliance**:
> All kinematic quantities in this repository strictly adhere to [ROS REP 103](https://www.ros.org/reps/rep-0103.html) and SI units:
> - **Linear velocity ($v$)**: $\text{m/s}$ ($V_{\max} = 0.50\text{ m/s}$)
> - **Angular velocity ($\omega$)**: $\text{rad/s}$ ($\Omega_{\max} = 1.00\text{ rad/s} \approx 57.30^\circ/\text{s}$, never degrees/s)
> - **Angle / Orientation ($\theta$)**: $\text{rad}$ ($[-\pi, +\pi]$)
> - **Position / Coordinates ($x, y$)**: $\text{m}$ ($[-5.0, +5.0]\text{ m}$)

### Summary Table of Parameters

| Parameter | Symbol | Value | Physical & Mathematical Rationale |
| :--- | :---: | :---: | :--- |
| Spatial Arena Bounds | $X_{\max}, Y_{\max}$ | **$\pm 5.0\text{ m}$** | Covers the entire laboratory testing arena |
| Grid Sampling Step | $\Delta x, \Delta y$ | **$0.5\text{ m}$** | Discrete space interval specification |
| Absolute Robot Heading | $\theta_R$ | **$[-\pi, +\pi]\text{ rad}$** | Full $SO(2)$ circle coverage |
| Maximum Linear Velocity | $V_{\max}$ | **$0.50\text{ m/s}$** | Maximum linear command limit |
| Maximum Angular Velocity | $\Omega_{\max}$ | **$1.00\text{ rad/s}$** | Maximum safe yaw rate preventing wheel slip |
| Linear Velocity Set | $v_{\text{robot}}$ | **$\{-0.50, 0.0, +0.50\}\text{ m/s}$** | Strictly discrete ternary output |
| Angular Velocity Set | $w_{\text{robot}}$ | **$\{-1.00, 0.0, +1.00\}\text{ rad/s}$** | Strictly discrete ternary output |
| Deceleration Funnel | $D_{\text{brake}}$ | **$0.20\text{ m}$** | Allows the MLP to learn a smooth deceleration ramp |
| Terminal Stop Deadband | $D_{\text{stop}}$ | **$0.05\text{ m}$** | Prevents motor chattering and limit cycles at origin |

### Coordinate Transformation

The neural network navigates strictly in the **robot's local body frame**:

$$\begin{bmatrix} x_{\text{target}}^{\text{robot}} \\ y_{\text{target}}^{\text{robot}} \end{bmatrix} = \begin{bmatrix} \cos\theta_R & \sin\theta_R \\ -\sin\theta_R & \cos\theta_R \end{bmatrix} \begin{bmatrix} x_{\text{target}} - x_{\text{robot}} \\ y_{\text{target}} - y_{\text{robot}} \end{bmatrix}$$

### Input & Output Normalization

$$\text{in}_0 = \frac{x_{\text{target}}^{\text{robot}}}{5.0} \in [-1.0, 1.0], \quad \text{in}_1 = \frac{y_{\text{target}}^{\text{robot}}}{5.0} \in [-1.0, 1.0], \quad \text{in}_2 = \frac{\theta_{\text{robot}}}{\pi} \in [-1.0, 1.0]$$

During inference on the robot:
$$v_{\text{cmd}} = \hat{y}_0 \times 0.50\text{ m/s}, \quad \omega_{\text{cmd}} = \hat{y}_1 \times 1.00\text{ rad/s}$$

---

## 🚀 Quickstart & Execution Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jupivato/limo-offline-learning.git
cd limo-offline-learning

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate the Dataset

Generates the 5-column dataset across $[-5, +5]\text{ m}$ every $0.5\text{ m}$ with high-density docking and 20 Hz rollouts:

```bash
python3 src/generate_dataset.py
```

### 3. Train the Bias-Free Neural Controller

Trains the MLP (`3 -> 1000 -> 2`) using pure constant SGD (strictly constant learning rate, zero momentum, zero weight decay) for 150 epochs:

```bash
python3 src/train_offline.py
```

*(For complete operational instructions, directory structures, and Gazebo tutorials, see [TUTORIAL.md](TUTORIAL.md)).*

### 4. Plot Training Convergence Curves

Visualizes training and validation MSE curves:

```bash
python3 src/plot_curves.py
```

### 5. Closed-Loop Evaluation & Benchmarking

#### Single Scenario Execution:
```bash
python3 src/evaluation/run_autonomous.py \
    --weights models/supervised_w_torch_diff.json \
    --x-target 0.0 --y-target 0.0 --theta-target 0.0
```

#### Batch Benchmarking (Canonical Scenarios under ISO 18646-2):
```bash
python3 src/evaluation/run_batch_benchmark.py \
    --weights models/supervised_w_torch_diff.json \
    --outdir data/evaluation/batch_run
```

---

## 👥 Authors & Affiliation

- **Framework Author**: Julia Pivato
- **Project**: Projet Pluridisciplinaire en Équipe (PPE)
- **Institution**: Centre Européen de Réalité Virtuelle (CERV) / École Nationale d'Ingénieurs de Brest (ENIB)
- **Target Hardware**: AgileX Robotics LIMO (Four-Wheel Differential / Ackermann ROS 2 Platform)
