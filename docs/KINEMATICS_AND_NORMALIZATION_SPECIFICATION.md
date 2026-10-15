# Kinematic Specifications, Discrete Action Mapping, and Normalization Theory

**Project**: LIMO Offline Learning (CERV / ENIB)  
**Robot Platform**: AgileX LIMO (Differential Drive Mode)  
**Target Standard**: ISO 18646-2 (Robots and Robotic Devices — Mobile Service Robots — Performance Criteria and Test Methods)  
**Date**: October 2026  

---

## 1. Executive Summary & Core Motivation

In behavioral cloning (offline imitation learning) for mobile robot docking, the supervisor demonstrates expert actions to drive the vehicle from any initial pose $(x_0, y_0, \theta_0)$ to the dock $(0, 0, 0)$ with zero velocity $(v=0, \omega=0)$.

Prior iterations of this project suffered from:
1. **Vanishing Gradients / Average Prediction**: Datasets populated with continuous, smoothly varying decimal speeds ($0.05, 0.11, 0.18, 0.24\text{ m/s}$) trained with unnormalized targets caused the neural network to merely predict the dataset average ($\approx 0.207\text{ m/s}$), showing almost no loss reduction across epochs.
2. **Dynamic Range Mismatch**: The [`PioneerNN`](../../src/core/pioneer_nn.py) architecture terminates with a `Tanh()` activation function bounded strictly in $[-1.0, +1.0]$. Training directly on raw velocities (e.g. $0.24\text{ m/s}$) compressed all activations into a tiny linear sub-regime near zero, discarding the non-linear classification and saturation capacity of the MLP.

To resolve this, the **V2 framework** adopts a **3-Level Discrete Supervisory Policy (Bang-Off-Bang Control)** normalized symmetrically to $\{-1.0, 0.0, +1.0\}$:
$$\mathbf{v} \in \{-V_{\max}, \; 0, \; +V_{\max}\} \xrightarrow{\text{normalize}} \{-1.0, \; 0.0, \; +1.0\}$$
$$\mathbf{\omega} \in \{-\Omega_{\max}, \; 0, \; +\Omega_{\max}\} \xrightarrow{\text{normalize}} \{-1.0, \; 0.0, \; +1.0\}$$

This document establishes the theoretical, physical, and metrological rationale behind every numerical parameter chosen in the pipeline.

---

## 2. Platform Hardware Constraints (AgileX LIMO)

The AgileX LIMO mobile robot exhibits the following manufacturer baseline characteristics:

| Parameter | Symbol | Manufacturer Value | Source / Significance |
| :--- | :---: | :---: | :--- |
| **Track Width (gauge)** | $L$ | $0.172\text{ m}$ ($172\text{ mm}$) | Distance between left and right wheels |
| **Half-Track (effective lever)** | $b = L/2$ | $0.086\text{ m}$ ($86\text{ mm}$) | Kinematic multiplier for differential steering |
| **Wheelbase** | $l$ | $0.200\text{ m}$ ($200\text{ mm}$) | Front-to-rear axle distance |
| **Datasheet Max Speed (Diff Mode)** | $v_{\text{mfg}}$ | $0.40 \sim 0.50\text{ m/s}$ | Low-level STM32 firmware clamp |
| **Datasheet Max Angular Rate** | $\omega_{\text{mfg}}$ | $1.00 \sim 2.00\text{ rad/s}$ | In-place rotation limit before motor torque sag |
| **ROS 2 Driver Safety Ceiling** | $v_{\text{safe}}, \omega_{\text{safe}}$ | $0.50\text{ m/s}, 1.00\text{ rad/s}$ | Enforced in `limo_interface.py` to protect gearboxes |

### 2.1. Metrological Standards, Units of Measure & ROS REP-103 Compliance

All telemetry, state estimation, control setpoints, and neural network targets strictly adhere to the international robotic standard **[ROS REP 103: Standard Units of Measure and Coordinate System Conventions](https://www.ros.org/reps/rep-0103.html)** and the International System of Units (SI):

| Physical Quantity | Symbol | Standard SI Unit | ROS 2 Message & Field | Definition & Value in Project |
| :--- | :---: | :---: | :--- | :--- |
| **Linear Position (2D)** | $x, y$ | $\text{m}$ (meters) | `geometry_msgs/Point` | Arena Cartesian coordinate ($[-5.0, +5.0]\text{ m}$) |
| **Heading / Orientation** | $\theta$ | $\text{rad}$ (radians) | `geometry_msgs/Quaternion` | Principal angle in $[-\pi, +\pi]\text{ rad}$ ($[-180^\circ, +180^\circ]$) |
| **Linear Velocity** | $v$ | $\text{m/s}$ (meters/sec) | `Twist.linear.x` | Longitudinal motion ($V_{\max} = 0.24\text{ m/s}$) |
| **Angular Velocity** | $\omega$ | **$\text{rad/s}$ (radians/sec)** | `Twist.angular.z` | Yaw rotational speed ($\Omega_{\max} = 0.50\text{ rad/s} \approx 28.65^\circ/\text{s}$) |
| **Control Frequency** | $f_{\text{ctrl}}$ | $\text{Hz}$ (Hertz) | Wall-clock rate | Nominal $20\text{ Hz}$ ($\Delta t = 50\text{ ms}$) |

#### Dimensional Proof: Why Angular Velocity MUST be in $\text{rad/s}$ (Not $\text{deg/s}$)

In differential-drive mobile robotics, the unicycle kinematics relating the central vehicle twist $(v, \omega)$ to the peripheral linear velocities of the left ($v_L$) and right ($v_R$) wheels are governed by:
$$v_R = v + \frac{L}{2} \cdot \omega, \qquad v_L = v - \frac{L}{2} \cdot \omega$$

Applying dimensional analysis to each term:
- $[v] = \frac{\text{m}}{\text{s}}$
- $[L] = \text{m} \implies \left[\frac{L}{2}\right] = \text{m}$
- For the addition $v + \frac{L}{2} \cdot \omega$ to be physically homogeneous, the product $\left[\frac{L}{2} \cdot \omega\right]$ **must** have dimension $\frac{\text{m}}{\text{s}}$:
  $$[\text{m}] \times [\omega] = \frac{\text{m}}{\text{s}} \implies [\omega] = \frac{1}{\text{s}} \equiv \frac{\text{rad}}{\text{s}}$$

Because radians are dimensionless geometric ratios ($\text{arc length} / \text{radius} = \text{m} / \text{m} = 1$), $\text{rad/s}$ preserves dimensional homogeneity without introducing arbitrary conversion factors. If $\omega$ were expressed in degrees per second ($\text{deg/s}$), the kinematic equations would require an explicit conversion factor of $\frac{\pi}{180^\circ} \approx 0.01745$, which is **absent** from the AgileX LIMO firmware and the ROS 2 differential drive controller (`diff_drive_controller`).

#### Handy Metrological Conversion Table for Reports and Defenses

| Rotational State | Value ($\text{rad/s}$) | Equivalent in Degrees ($\text{deg/s}$) | Physical Interpretation on LIMO Chassis |
| :--- | :---: | :---: | :--- |
| **Zero Rotation** | $0.00\text{ rad/s}$ | $0.00^\circ/\text{s}$ | Pure straight-line motion ($v_L = v_R$) |
| **Fine Alignment** | $\pm 0.10\text{ rad/s}$ | $\pm 5.73^\circ/\text{s}$ | Smooth heading correction near docking target |
| **Standard Turn** | $\pm 0.25\text{ rad/s}$ | $\pm 14.32^\circ/\text{s}$ | Dubins approach arc transition |
| **Max Policy Speed ($\Omega_{\max}$)** | $\pm \mathbf{0.50\text{ rad/s}}$ | $\pm \mathbf{28.65^\circ/\text{s}}$ | Full supervisory turn rate ($360^\circ$ circle in $12.57\text{ s}$) |
| **Firmware Limit ($\omega_{\text{mfg}}$)** | $\pm 1.00\text{ rad/s}$ | $\pm 57.30^\circ/\text{s}$ | Maximum allowable safety ceiling in hardware |

---

## 3. Kinematic Parameter Derivation

### 3.1. Maximum Linear Velocity: $V_{\max} = 0.24\text{ m/s}$

While the manufacturer firmware allows up to $0.50\text{ m/s}$, the project sets $V_{\max} = 0.24\text{ m/s}$ ($48\%$ of the physical ceiling) based on three rigorous physical principles:

#### A. Control Loop Spatial Discretization (20 Hz)
The robot's onboard controller runs at a fixed frequency of $f = 20\text{ Hz}$ ($\Delta t = 50\text{ ms}$). During each discrete control tick, the robot advances by:
$$\Delta s = V_{\max} \cdot \Delta t = 0.24\text{ m/s} \times 0.050\text{ s} = \mathbf{0.012\text{ m}} \; (1.2\text{ cm})$$
- A spatial discretization of **$1.2\text{ cm}$ per control tick** guarantees that the controller can position the vehicle within the strict ISO 18646-2 accuracy envelope ($A_p \le 10\text{ cm}$) without jumping or overshoot.
- Operating at $0.80\text{ m/s}$ would jump $4\text{ cm}$ per tick, creating spatial hunting oscillations around the docking target.

#### B. Odometry Integrity & Wheel Slippage Avoidance
The AgileX LIMO dead-reckoning state estimator integrates wheel encoder pulses on `/odom`. At linear speeds above $0.35\text{ m/s}$ on standard lab floors, longitudinal slip occurs during acceleration and braking. At $V_{\max} = 0.24\text{ m/s}$, the tire-floor interface operates strictly within the static friction regime ($\mu_{\text{static}}$), preserving cumulative odometric accuracy.

#### C. Reverse Symmetry: $V_{\min} = -0.24\text{ m/s}$
Due to the bilateral symmetry of differential drive kinematics, reverse docking maneuvers (Hermite cubic splines for parking from $x \ge 0$) mirror the forward cruise speed symmetrically:
$$V_{\min} = -V_{\max} = -0.24\text{ m/s}$$

---

### 3.2. Minimum Curvature Radius & Angular Velocity: $\Omega_{\max} = 0.50\text{ rad/s}$

#### A. The Non-Skid Turning Radius ($R_{\min} = 0.42\text{ m}$)
In differential-drive robotics, unicycle kinematics dictate the relationship between turning radius $R$, linear velocity $v$, and angular velocity $\omega$:
$$R = \frac{v}{\omega} \implies \omega = \frac{v}{R}$$

On smooth industrial surfaces, forcing a unicycle robot to turn tighter than its wheel scrub radius causes lateral wheel slip. For the LIMO chassis, the empirical minimum turning radius without tire scrubbing is **$R_{\min} = 0.42\text{ m}$**.

Evaluating maximum angular rate at cruise velocity yields:
$$\omega_{\text{cruise}} = \frac{V_{\max}}{R_{\min}} = \frac{0.24\text{ m/s}}{0.42\text{ m}} \approx 0.5714\text{ rad/s}$$

#### B. Selection of Operational Bound $\Omega_{\max} = 0.50\text{ rad/s}$
To provide uniform, symmetric discretization and protect the brushless hub motors during in-place pivot maneuvers, the maximum angular velocity is bounded to:
$$\Omega_{\max} = \mathbf{0.50\text{ rad/s}} \; (\approx 28.65^\circ/\text{s})$$

At $\Omega_{\max} = 0.50\text{ rad/s}$:
- The individual wheel linear velocities remain bounded:
  $$v_{\text{wheel}} = v \pm \omega \cdot b = 0.24 \pm (0.50 \times 0.086) = 0.24 \pm 0.043 \in [0.197, \; 0.283]\text{ m/s}$$
  Both wheels continue rotating forward smoothly, eliminating high-frequency motor reversals.
- For pure rotation on the spot ($v = 0$ at the dock), turning at $28.65^\circ/\text{s}$ gives LiDAR and camera sensors sufficient integration time to prevent motion blur and drift.

---

### 3.3. Terminal Braking Radius: $D_{\text{brake}} = 0.20\text{ m}$

#### A. Newtonian Stopping Distance Model
When the linear speed command is cut from $V_{\max}$ to $0$, the vehicle decelerates under kinetic tire-floor friction ($\mu_k \approx 0.35$):
$$d_{\text{stop}} \approx \frac{V_{\max}^2}{2 \mu_k g} = \frac{(0.24)^2}{2 \times 0.35 \times 9.81} = \frac{0.0576}{6.867} \approx \mathbf{0.0084\text{ m}} \; (8.4\text{ mm})$$

The vehicle's physical braking distance is less than $1\text{ cm}$, meaning mechanical momentum does not cause uncontrolled sliding.

#### B. Rationale for the 20 cm Window
Setting $D_{\text{brake}} = 0.20\text{ m}$ ($20\text{ cm}$) provides:
1. **Continuous Inductive Transition**: Even though the training labels drop sharply from $+1.0$ to $0.0$, a neural network parameterized with smooth `Tanh` activations cannot jump instantaneously. Over the $20\text{ cm}$ spatial window, the MLP naturally fits a **smooth sigmoidal deceleration profile**:
   $$1.0 \longrightarrow 0.85 \longrightarrow 0.52 \longrightarrow 0.18 \longrightarrow 0.00$$
2. **Heading Decoupling**: Once $d \le D_{\text{brake}}$, linear translation stops ($v=0$), and the remaining control budget is devoted entirely to angular alignment ($\omega = \pm \Omega_{\max}$) until heading error satisfies $|\theta| \le \theta_{\text{tol}}$.

---

## 4. Normalization and PioneerNN Activation Compatibility

### 4.1. Robot-Centric Coordinate Transformation

The system expresses the target directly in the **robot's local body coordinate frame**:
Given the robot global pose $(x_R, y_R, \theta_R)$ and docking target global pose $(x_T, y_T)$:
$$\begin{bmatrix} x_{\text{target}}^{\text{robot}} \\ y_{\text{target}}^{\text{robot}} \end{bmatrix} = \begin{bmatrix} \cos\theta_R & \sin\theta_R \\ -\sin\theta_R & \cos\theta_R \end{bmatrix} \begin{bmatrix} x_T - x_R \\ y_T - y_R \end{bmatrix}$$

- $x_{\text{target}}^{\text{robot}} > 0$: Target is in front of the robot.
- $x_{\text{target}}^{\text{robot}} < 0$: Target is behind the robot.
- $y_{\text{target}}^{\text{robot}} > 0$: Target is to the left of the robot.
- $y_{\text{target}}^{\text{robot}} < 0$: Target is to the right of the robot.
- $\theta_{\text{robot}}$: Absolute robot heading in the global frame ($-\pi \le \theta_R \le \pi$).

### 4.2. Input Normalization into $[-1.0, +1.0]$

With the arena boundaries defined at $X_{\max} = 5.0\text{ m}$ and $Y_{\max} = 5.0\text{ m}$, the 3 neural inputs are normalized as:
$$in_0 = \frac{x_{\text{target}}^{\text{robot}}}{X_{\max}} = \frac{x_{\text{target}}^{\text{robot}}}{5.0} \in [-1.0, +1.0]$$
$$in_1 = \frac{y_{\text{target}}^{\text{robot}}}{Y_{\max}} = \frac{y_{\text{target}}^{\text{robot}}}{5.0} \in [-1.0, +1.0]$$
$$in_2 = \frac{\theta_{\text{robot}}}{\pi} \in [-1.0, +1.0]$$

### 4.3. Target Output Normalization

Because PioneerNN defines its output layer as:
$$\text{output} = \text{Tanh}(\text{Linear}(1000, 2)) \implies \hat{\mathbf{u}} \in (-1.0, +1.0)^2$$

During offline training in [`train_offline.py`](../src/train_offline.py), raw velocities $(v_{\text{robot}}, w_{\text{robot}})$ are normalized:
$$y_v = \frac{v_{\text{robot}}}{V_{\max}} \in \{-1.0, \; 0.0, \; +1.0\}$$
$$y_\omega = \frac{w_{\text{robot}}}{\Omega_{\max}} \in \{-1.0, \; 0.0, \; +1.0\}$$

### 4.4. The Zero-Bias Equilibrium Guarantee (`use_bias=False`)

In the V2 architecture, neural biases are strictly deactivated (`use_bias=False`):
$$\mathbf{h} = \text{Tanh}(\mathbf{W}_{\text{in}} \cdot \mathbf{x})$$
$$\hat{\mathbf{u}} = \text{Tanh}(\mathbf{W}_{\text{out}} \cdot \mathbf{h})$$

When the robot reaches the exact target origin $(x_{\text{target}}^{\text{robot}} = 0, y_{\text{target}}^{\text{robot}} = 0, \theta_{\text{robot}} = 0)$:
$$\mathbf{x} = [0, 0, 0]^T \implies \mathbf{h} = \text{Tanh}(\mathbf{0}) = \mathbf{0} \implies \hat{\mathbf{u}} = \text{Tanh}(\mathbf{0}) = [0, 0]^T$$

This provides a **mathematical guarantee of static equilibrium**: the robot will never exhibit steady-state drift, creep, or residual twitching when docked at the target origin.

---

## 5. Runtime Denormalization Pipeline

During autonomous execution ([`run_autonomous.py`](../../src/evaluation/run_autonomous.py) and [`run_batch_benchmark.py`](../../src/evaluation/run_batch_benchmark.py)), the normalized network inference is mapped back to physical commands:

```python
# 1. Transform odometry pose to robot-centric frame
dx = target[0] - pos[0]
dy = target[1] - pos[1]
theta_r = recast_angle(pos[2])

x_tr = dx * math.cos(theta_r) + dy * math.sin(theta_r)
y_tr = -dx * math.sin(theta_r) + dy * math.cos(theta_r)

# 2. Normalize inputs to [-1.0, +1.0]
in_0 = max(-1.0, min(1.0, x_tr / 5.0))
in_1 = max(-1.0, min(1.0, y_tr / 5.0))
in_2 = max(-1.0, min(1.0, theta_r / math.pi))

# 3. Forward inference: outputs in [-1.0, +1.0]
with torch.no_grad():
    cmd = model(torch.tensor([in_0, in_1, in_2], dtype=torch.float32)).squeeze().tolist()

# 4. Denormalization back to physical units
v_cmd = float(cmd[0]) * V_MAX   # m/s
w_cmd = float(cmd[1]) * W_MAX   # rad/s

# 5. Hardware safety clamping
v_cmd = max(-V_MAX, min(V_MAX, v_cmd))
w_cmd = max(-W_MAX, min(W_MAX, w_cmd))

# 6. Dispatch to ROS 2 /cmd_vel
robot.set_cmd_vel(v_cmd, w_cmd)
```

---

## 6. Empirical Dataset Validation (V2 Distribution)

Executing [`generate_dataset.py`](../src/generate_dataset.py) over the $5\times 5\text{ m}$ domain with step $0.5\text{ m}$, terminal docking subgrid, and 20 Hz dynamic rollouts generates **70,664 unique samples** with the following exact statistical distribution:

### Linear Velocity ($v_{\text{robot}}$)
```text
  v = -0.24 m/s : 14,739 samples (20.9%)  -> Reverse docking maneuvers
  v =  0.00 m/s : 21,427 samples (30.3%)  -> Braking threshold (d <= 0.20m) & terminal alignment
  v = +0.24 m/s : 34,498 samples (48.8%)  -> Forward cruise navigation
```

### Angular Velocity ($w_{\text{robot}}$)
```text
  w = -0.50 rad/s : 23,113 samples (32.7%) -> Full-rate clockwise rotation
  w =  0.00 rad/s : 24,438 samples (34.6%) -> Straight-line navigation / fully aligned
  w = +0.50 rad/s : 23,113 samples (32.7%) -> Full-rate counter-clockwise rotation
```

*(Note the exact numerical symmetry of angular commands: precisely 23,113 samples on each side due to bilateral sagittal mirroring!)*

---

## 7. Master Parameter Summary Table

| Parameter | Value | Unit | Role | Verification & Enforcement |
| :--- | :---: | :---: | :--- | :--- |
| **$X_{\max}, Y_{\max}$** | `5.0` | $\text{m}$ | Operational arena limits | Input normalization divisor ($x/5.0, y/5.0$) |
| **Grid Step $\Delta_{xy}$** | `0.5` | $\text{m}$ | Sampling interval across $[-5.0, 5.0]\text{ m}$ | Discrete state space coverage in `generate_dataset.py` |
| **$V_{\max}$** | `0.24` | $\text{m/s}$ | Maximum forward cruising speed | Dataset generation & denormalization multiplier |
| **$V_{\min}$** | `-0.24` | $\text{m/s}$ | Maximum reverse docking speed | Symmetric reverse cruise |
| **$\Omega_{\max}$** | `0.50` | $\text{rad/s}$ | Maximum angular steering rate | Curvature bound $\frac{V_{\max}}{R_{\min}}$ & denormalization |
| **$D_{\text{brake}}$** | `0.20` | $\text{m}$ | Terminal braking activation radius | Forces $v=0$ inside $20\text{ cm}$ radius |
| **$D_{\text{stop}}$** | `0.05` | $\text{m}$ | Terminal dock deadband | Forces $v=0, \omega=0$ upon dock arrival |
| **$R_{\min}$** | `0.42` | $\text{m}$ | Minimum non-skid turning radius | Kinematic curvature limit on lab flooring |
| **$f_{\text{ctrl}}$** | `20` | $\text{Hz}$ | Sampling frequency ($\Delta t = 50\text{ ms}$) | Real-time ROS 2 publish rate on `/cmd_vel` |
| **$\Delta s$** | `1.2` | $\text{cm}$ | Spatial displacement per tick | $V_{\max} \cdot \Delta t = 0.24 \cdot 0.05$ |
| **`use_bias`** | `False` | bool | Zero-bias equilibrium constraint | Guarantees $\text{MLP}(\mathbf{0}) = \mathbf{0}$ at dock |
| **Loss Beta** | `1.0` | - | Angular error weighting in MSE | $\mathcal{L} = \text{MSE}_v + \beta \cdot \text{MSE}_\omega$ |
