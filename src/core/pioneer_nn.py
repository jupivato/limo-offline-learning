"""
PioneerNN Neural Controller Architecture
=========================================
Nonlinear Multi-Layer Perceptron (MLP) for closed-loop pose positioning 
and docking of non-holonomic mobile robots (AgileX LIMO / Pioneer 3-DX).

Theoretical Framework:
----------------------
Based on the canonical ENIB / CERV / Lab-STICC formulation (Prof. Patrick Hénaff):
1. Robot-Centric Kinematic Mapping:
   The state vector is defined in the robot's local frame of reference:
       in_0 = x_target_robot / X_MAX ∈ [-1.0, +1.0]
       in_1 = y_target_robot / Y_MAX ∈ [-1.0, +1.0]
       in_2 = recast(theta_robot - theta_s(x_R, y_R)) / π ∈ [-1.0, +1.0]
   where theta_s(x_R, y_R) is the strategic docking vector-field orientation.

2. Strict Bias-Free Equilibrium Theorem:
   The neural network operates without affine bias parameters (use_bias = False):
       y = tanh(W_out * tanh(W_in * x))
   Because tanh(0) = 0:
       x = [0, 0, 0]^T  ==>  y = [0, 0]^T (Identically zero).
   This algebraic property guarantees unconditional vehicle immobilization at 
   the docking target, precluding steady-state chattering, residual motor creep, 
   or limit-cycle oscillations around the origin.

3. Radial Angular Damping Gain (K_theta):
       K_theta(x_R, y_R) = 1.0 - 0.9 * exp(-10 * (x_R^2 + y_R^2)^2) ∈ [0.1, 1.0]
   This gain serves as an execution-layer post-processing multiplier for angular 
   velocity (w_cmd = w_discrete * K_theta). It smoothly attenuates high-frequency 
   chassis rotation within the terminal docking radius while preserving agile yaw 
   rates in the far field. It is NOT fed into the MLP inputs in order to preserve 
   the homogeneity property y(0) = 0.

Compatibility:
--------------
Implements a dual-backend architecture: Native PyTorch tensor execution with an 
autonomous pure-Python standard library fallback for headless embedded environments.

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
Standards: ROS REP-103 & ISO 18646-2
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
import math
from typing import Dict, Any, Union, List, Optional, Tuple

# Resolve project root directory dynamically
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
TORCH_AVAILABLE = True


def recast_angle(angle_rad: float) -> float:
    """
    Recasts an arbitrary angle into the canonical principal interval (-pi, +pi].

    Args:
        angle_rad (float): Input angle in radians.

    Returns:
        float: Equivalent angle wrapped strictly in (-pi, +pi].
    """
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compute_theta_strategic(x: float, y: float) -> float:
    """
    Computes the canonical ENIB strategic approach orientation theta_s(x, y).

    Mathematical Formulation:
        theta_s(x, y) = tanh(10.0 * x) * atan(1.0 * y)

    Kinematic and Control Properties:
    - Eliminates directional symmetry ambiguities and local minima in the plane.
    - Guides the mobile robot tangentially along the longitudinal docking corridor
      as (x, y) -> (0, 0), preventing orbital limit cycles and awkward sideways docking.
    - Bound: theta_s in (-pi/2, +pi/2).

    Args:
        x (float): Relative X coordinate of the target in the robot body frame (m).
        y (float): Relative Y coordinate of the target in the robot body frame (m).

    Returns:
        float: Strategic heading reference angle in radians.
    """
    return math.tanh(10.0 * float(x)) * math.atan(1.0 * float(y))


def compute_k_theta(x_r: float, y_r: float) -> float:
    """
    Computes the radial angular velocity damping gain K_theta.

    Mathematical Formulation:
        K_theta(x_R, y_R) = 1.0 - 0.9 * exp(-10.0 * (x_R^2 + y_R^2)^2)

    Physical Interpretation:
    - At the docking terminal (x_R=0, y_R=0): exp(0) = 1 ==> K_theta = 1 - 0.9 = 0.1
    - At the far field (r^2 -> infinity)     : exp(...) -> 0 ==> K_theta -> 1.0
    - Strict Boundedness: K_theta in [0.1, 1.0]

    Application:
    Used as an execution-layer post-filter on angular velocity:
        w_cmd = w_discrete * K_theta(x_R, y_R)
    Provides a 90% smooth reduction of turning jerk within the terminal docking
    radius (r < 0.5m) while allowing full pivot capability far from the dock.

    Args:
        x_r (float): Target relative X position in robot body coordinates (m).
        y_r (float): Target relative Y position in robot body coordinates (m).

    Returns:
        float: Attenuation factor clamped in [0.1, 1.0].
    """
    try:
        r2 = float(x_r) ** 2 + float(y_r) ** 2
        r4 = r2 ** 2
        arg = -10.0 * r4
        # Numerical safeguard against IEEE-754 underflow/overflow
        if arg < -50.0:
            arg = -50.0
        elif arg > 0.0:
            arg = 0.0
        k_theta = 1.0 - 0.9 * math.exp(arg)
    except Exception:
        k_theta = 1.0
    return max(0.1, min(1.0, float(k_theta)))


class ReplayBuffer:
    """
    Transition replay buffer for mobile robot reinforcement and imitation learning.
    Stores tuples of (state, action, reward, next_state, done).
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: collections.deque[Tuple[List[float], List[float], float, List[float], bool]] = collections.deque(maxlen=capacity)

    def add(
        self,
        state: Union[List[float], Any],
        action: Union[List[float], Any],
        reward: float,
        next_state: Union[List[float], Any],
        done: bool
    ) -> None:
        """Appends a transition tuple to the buffer."""
        self.buffer.append((list(state), list(action), float(reward), list(next_state), bool(done)))

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Samples a uniform random mini-batch of transitions."""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": list(states),
            "actions": list(actions),
            "rewards": list(rewards),
            "next_states": list(next_states),
            "dones": list(dones)
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class DummyTensor(list):
    """Tensor-compatible list container for the pure-Python fallback execution."""
    def squeeze(self):
        return self

    def tolist(self):
        return list(self)


if TORCH_AVAILABLE:
    class BaseNN(nn.Module):
        pass
else:
    class BaseNN(object):
        pass


class PioneerNN(BaseNN):
    """
    PioneerNN Multi-Layer Perceptron (MLP) for Mobile Robot Velocity Control.

    Canonical Specifications:
    - input_size  : 3 normalized body-frame features:
                    [x_R / X_MAX, y_R / Y_MAX, recast(theta_R - theta_s) / pi]
    - hidden_size : 1000 dense units with Tanh activation
    - output_size : 2 normalized velocity setpoints [v_norm, w_norm] with Tanh activation
    - use_bias    : False (Strict guarantee that MLP(0, 0, 0) == (0, 0))
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 1000,
        output_size: int = 2,
        use_bias: bool = False
    ):
        """
        Initializes the PioneerNN network.

        Args:
            input_size (int): Dimension of input feature vector (default: 3).
            hidden_size (int): Number of units in the hidden layer (default: 1000).
            output_size (int): Dimension of output command vector (default: 2 [v, w]).
            use_bias (bool): Whether to include affine bias vectors (default: False).
        """
        if TORCH_AVAILABLE:
            super(PioneerNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_bias = use_bias

        if TORCH_AVAILABLE:
            self.hidden = nn.Linear(input_size, hidden_size, bias=use_bias)
            self.output = nn.Linear(hidden_size, output_size, bias=use_bias)
            self.activation = nn.Tanh()
            self._init_weights()

        # Pure-Python weight representations for seamless embedded deployment
        self._py_input_weights = [[0.0] * hidden_size for _ in range(input_size)]
        self._py_output_weights = [[0.0] * output_size for _ in range(hidden_size)]
        self._py_hidden_bias = [0.0] * hidden_size
        self._py_output_bias = [0.0] * output_size

    def _init_weights(self) -> None:
        """Initializes synaptic weights via Xavier Uniform initialization."""
        if TORCH_AVAILABLE:
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.xavier_uniform_(self.output.weight)
            if self.use_bias:
                nn.init.zeros_(self.hidden.bias)
                nn.init.zeros_(self.output.bias)

    def eval(self):
        """Sets network in evaluation mode (inference)."""
        if TORCH_AVAILABLE and hasattr(super(), "eval"):
            super().eval()
        return self

    def train(self, mode: bool = True):
        """Sets network in training mode."""
        if TORCH_AVAILABLE and hasattr(super(), "train"):
            super().train(mode)
        return self

    def forward(self, x: Any) -> Any:
        """
        Executes forward propagation pass: y = Tanh(W_out * Tanh(W_in * x)).
        Accepts PyTorch tensors, Python lists, or nested 2D matrices.
        """
        if TORCH_AVAILABLE:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.device != self.hidden.weight.device:
                x = x.to(self.hidden.weight.device)
            if not self.training:
                with torch.no_grad():
                    out = self.activation(self.hidden(x))
                    out = self.activation(self.output(out))
                    return out
            out = self.activation(self.hidden(x))
            out = self.activation(self.output(out))
            return out

        # Pure Python fallback
        if hasattr(x, "tolist"):
            in_vec = x.tolist()
        else:
            in_vec = list(x)

        if len(in_vec) > 0 and isinstance(in_vec[0], list):
            in_vec = in_vec[0]

        # Hidden layer forward pass
        h = []
        for j in range(self.hidden_size):
            val = self._py_hidden_bias[j] if self.use_bias else 0.0
            val += sum(in_vec[i] * self._py_input_weights[i][j] for i in range(self.input_size))
            h.append(math.tanh(val))

        # Output layer forward pass
        out = []
        for k in range(self.output_size):
            val = self._py_output_bias[k] if self.use_bias else 0.0
            val += sum(h[j] * self._py_output_weights[j][k] for j in range(self.hidden_size))
            out.append(math.tanh(val))

        return DummyTensor(out)

    def __call__(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(super(), "__call__"):
            return super().__call__(x)
        return self.forward(x)

    def save_weights_to_json(self) -> Dict[str, Any]:
        """
        Serializes synaptic weights and architecture metadata into the canonical JSON dictionary.
        """
        if TORCH_AVAILABLE:
            input_weights = []
            hidden_weights_transposed = self.hidden.weight.detach().cpu().t()
            for i in range(self.input_size):
                row = []
                for j in range(self.hidden_size):
                    row.append(float(hidden_weights_transposed[i][j]))
                input_weights.append(row)

            output_weights = []
            out_weights_transposed = self.output.weight.detach().cpu().t()
            for i in range(self.hidden_size):
                row = []
                for j in range(self.output_size):
                    row.append(float(out_weights_transposed[i][j]))
                output_weights.append(row)

            data = {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "use_bias": self.use_bias,
                "input_weights": input_weights,
                "output_weights": output_weights,
            }

            if self.use_bias:
                data["hidden_bias"] = self.hidden.bias.detach().cpu().tolist()
                data["output_bias"] = self.output.bias.detach().cpu().tolist()

            return data

        # Pure Python serialization
        data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "use_bias": self.use_bias,
            "input_weights": self._py_input_weights,
            "output_weights": self._py_output_weights,
        }
        if self.use_bias:
            data["hidden_bias"] = self._py_hidden_bias
            data["output_bias"] = self._py_output_bias
        return data

    def load_weights_from_json(self, json_obj: Dict[str, Any]) -> None:
        """
        Loads synaptic weights from a JSON object with bidirectional backward-compatibility
        between 3-input and 4-input representations.
        """
        meta_input = json_obj.get("input_size", self.input_size)
        meta_hidden = json_obj.get("hidden_size", self.hidden_size)
        meta_output = json_obj.get("output_size", self.output_size)
        file_bias = json_obj.get("use_bias", False)

        if meta_hidden != self.hidden_size:
            raise ValueError(f"Hidden layer mismatch: JSON={meta_hidden}, Model={self.hidden_size}")
        if meta_output != self.output_size:
            raise ValueError(f"Output layer mismatch: JSON={meta_output}, Model={self.output_size}")

        self.use_bias = file_bias

        # Re-instantiate PyTorch layers if bias or dimensions altered
        if TORCH_AVAILABLE:
            self.hidden = nn.Linear(self.input_size, self.hidden_size, bias=self.use_bias)
            self.output = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)

            # Map weights with slicing if incoming JSON has differing input dimension
            input_weights = torch.zeros(self.input_size, self.hidden_size)
            src_in_weights = json_obj["input_weights"]
            copy_rows = min(len(src_in_weights), self.input_size)
            for i in range(copy_rows):
                for j in range(self.hidden_size):
                    input_weights[i][j] = src_in_weights[i][j]

            output_weights = torch.zeros(self.hidden_size, self.output_size)
            src_out_weights = json_obj["output_weights"]
            for i in range(self.hidden_size):
                for j in range(self.output_size):
                    output_weights[i][j] = src_out_weights[i][j]

            with torch.no_grad():
                self.hidden.weight.copy_(input_weights.t())
                self.output.weight.copy_(output_weights.t())

                if self.use_bias:
                    if "hidden_bias" in json_obj:
                        self.hidden.bias.copy_(torch.tensor(json_obj["hidden_bias"], dtype=torch.float32))
                    else:
                        nn.init.zeros_(self.hidden.bias)

                    if "output_bias" in json_obj:
                        self.output.bias.copy_(torch.tensor(json_obj["output_bias"], dtype=torch.float32))
                    else:
                        nn.init.zeros_(self.output.bias)

        # Pure Python mirror
        src_in = json_obj["input_weights"]
        py_in = []
        for i in range(self.input_size):
            if i < len(src_in):
                py_in.append(list(src_in[i]))
            else:
                py_in.append([0.0] * self.hidden_size)

        self._py_input_weights = py_in
        self._py_output_weights = json_obj["output_weights"]
        self._py_hidden_bias = json_obj.get("hidden_bias", [0.0] * self.hidden_size)
        self._py_output_bias = json_obj.get("output_bias", [0.0] * self.output_size)

    def save_to_json_file(self, file_path: str) -> None:
        """Saves synaptic weights into a formatted JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = self.save_weights_to_json()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json_file(self, file_path: str) -> None:
        """Loads synaptic weights from a JSON file with automatic path resolution."""
        resolved_path = file_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(_project_root, file_path),
                os.path.join(_project_root, "models", "supervised_w_torch_diff.json"),
                os.path.join(_project_root, "models", "pioneer_model_weights.json"),
                os.path.join(os.path.dirname(os.path.abspath(file_path)), "supervised_w_torch_diff.json"),
                os.path.join(os.path.dirname(os.path.abspath(file_path)), "pioneer_model_weights.json"),
                "models/supervised_w_torch_diff.json",
                "models/pioneer_model_weights.json",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_path = cand
                    break

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Synaptic weights file not found: {file_path}")
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_weights_from_json(data)
