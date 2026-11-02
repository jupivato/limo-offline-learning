"""
Modèle Réseau de Neurones : PioneerNN
=====================================
Implémentation du perceptron multicouche (MLP) pour la commande
en boucle fermée du robot mobile AgileX LIMO.

Architecture :
- Entrées (4)  : Erreurs cartésiennes normalisées dans le repère local du robot,
                 orientation absolue du robot et gain dynamique de rotation :
                 [in_0, in_1, in_2, K_theta]
                 avec K_theta = 1 - 0.9 * exp(-10 * (x_R^2 + y_R^2)^2).
- Cachée (1000): Couche dense avec fonction d'activation Tangente Hyperbolique (Tanh).
- Sorties (2)  : Vitesses de consigne normalisées [v_lin, v_ang] avec activation Tanh.

Compatible PyTorch natif avec repli (fallback) autonome en pur Python standard
pour les environnements où PyTorch n'est pas installé.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
import math
from typing import Dict, Any, Union, List, Optional, Tuple

# Project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Auto-detect packages matching the current Python version
_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
for _p in [
    f"/home/juliapivato/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages",
    os.path.expanduser(f"~/snap/antigravity-cli/common/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/local/lib/python{_py_ver}/dist-packages"),
    os.path.expanduser(f"~/.local/lib/python{_py_ver}/site-packages"),
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    class _DummyNN:
        Module = object
    nn = _DummyNN()


def compute_k_theta(x_r: float, y_r: float) -> float:
    """
    Calcule dynamiquement le gain K_theta à partir des coordonnées de la cible (x_R, y_R)
    exprimées dans le repère local du robot :

        K_theta = 1 - 0.9 * exp(-10 * (x_R^2 + y_R^2)^2)

    Propriétés mathématiques et stabilité :
    - À l'accostage (x_R=0, y_R=0) : exp(0) = 1 => K_theta = 1 - 0.9 = 0.1
    - À l'infini ((x_R^2+y_R^2) -> inf) : exp(...) -> 0 => K_theta -> 1.0
    - Bornage garanti : K_theta in [0.1, 1.0]
    - Aucune erreur numérique : l'exposant est borné dans [-50.0, 0.0].

    Args:
        x_r (float): Coordonnée X de la cible dans le repère du robot (m).
        y_r (float): Coordonnée Y de la cible dans le repère du robot (m).

    Returns:
        float: Valeur de K_theta strictement comprise entre 0.1 et 1.0.
    """
    try:
        r2 = float(x_r) ** 2 + float(y_r) ** 2
        r4 = r2 ** 2
        arg = -10.0 * r4
        # Protection stricte contre l'underflow / overflow numérique
        if arg < -50.0:
            arg = -50.0
        elif arg > 0.0:
            arg = 0.0
        k_theta = 1.0 - 0.9 * math.exp(arg)
    except Exception:
        k_theta = 1.0
    return max(0.1, min(1.0, float(k_theta)))


def compute_theta_strategic(x: float, y: float) -> float:
    """
    Calcule l'orientation stratégique ENIB évitant les minima locaux et les orbites en boucle :
        θ_s(x, y) = tanh(10 * x) * atan(1 * y)
    
    Args:
        x (float): Coordonnée X de la pose globale ou erreur X vers la cible (m).
        y (float): Coordonnée Y de la pose globale ou erreur Y vers la cible (m).

    Returns:
        float: Angle stratégique d'accostage en radians (-pi/2, +pi/2).
    """
    return math.tanh(10.0 * float(x)) * math.atan(1.0 * float(y))


class ReplayBuffer:
    """
    Mémoire tampon d'expériences (Replay Buffer) pour l'apprentissage du robot mobile.
    Prend en compte le vecteur d'état étendu à 4 dimensions [in_0, in_1, in_2, K_theta].
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
        """Ajoute une transition dans le buffer."""
        self.buffer.append((list(state), list(action), float(reward), list(next_state), bool(done)))

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Échantillonne un mini-lot de transitions de façon uniforme."""
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
    """Conteneur compatible tenseur pour le repli en pur Python."""
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
    Perceptron multicouche pour le contrôle de vitesse d'un robot différentiel.
    Compatible avec le format de sérialisation JSON utilisé dans le projet LIMO.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 1000,
        output_size: int = 2,
        use_bias: bool = True
    ):
        """
        Initialise le réseau PioneerNN.

        Args:
            input_size (int): Dimension du vecteur d'entrée (défaut: 4 [in_0, in_1, in_2, K_theta]).
            hidden_size (int): Nombre de neurones dans la couche cachée (défaut: 1000).
            output_size (int): Dimension du vecteur de sortie (défaut: 2 [v_lin, v_ang]).
            use_bias (bool): Active les termes de biais (défaut: True).
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

        # Initialisation systématique des attributs Python pour compatibilité universelle
        self._py_input_weights = [[0.0] * hidden_size for _ in range(input_size)]
        self._py_output_weights = [[0.0] * output_size for _ in range(hidden_size)]
        self._py_hidden_bias = [0.0] * hidden_size
        self._py_output_bias = [0.0] * output_size

    def _init_weights(self) -> None:
        """Initialise les poids selon la méthode de Xavier Uniforme si PyTorch est actif."""
        if TORCH_AVAILABLE:
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.xavier_uniform_(self.output.weight)
            if self.use_bias:
                nn.init.zeros_(self.hidden.bias)
                nn.init.zeros_(self.output.bias)

    def eval(self):
        """Mode évaluation."""
        if TORCH_AVAILABLE and hasattr(super(), "eval"):
            super().eval()
        return self

    def train(self, mode: bool = True):
        """Mode entraînement."""
        if TORCH_AVAILABLE and hasattr(super(), "train"):
            super().train(mode)
        return self

    def forward(self, x: Any) -> Any:
        """
        Passe avant (forward pass) du réseau.
        Supporte aussi bien les tenseurs PyTorch que les listes Python.
        """
        if TORCH_AVAILABLE:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.device != self.hidden.weight.device:
                x = x.to(self.hidden.weight.device)
            out = self.activation(self.hidden(x))
            out = self.activation(self.output(out))
            return out

        # Repli pur Python
        if hasattr(x, "tolist"):
            in_vec = x.tolist()
        else:
            in_vec = list(x)

        if len(in_vec) > 0 and isinstance(in_vec[0], list):
            in_vec = in_vec[0]

        h = []
        for j in range(self.hidden_size):
            val = self._py_hidden_bias[j] if self.use_bias else 0.0
            val += sum(in_vec[i] * self._py_input_weights[i][j] for i in range(self.input_size))
            h.append(math.tanh(val))

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
        """Convertit les poids du réseau dans le format dictionnaire JSON canonique."""
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

        # En pur Python
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
        """Charge les poids synaptiques depuis un dictionnaire JSON."""
        meta_input = json_obj.get("input_size")
        meta_hidden = json_obj.get("hidden_size")
        meta_output = json_obj.get("output_size")

        is_legacy_3in = (meta_input == 3 and self.input_size == 4)
        if meta_input is not None and meta_input != self.input_size and not is_legacy_3in:
            raise ValueError(f"Dimension d'entrée incompatible : fichier={meta_input}, réseau={self.input_size}")
        if meta_hidden is not None and meta_hidden != self.hidden_size:
            raise ValueError(f"Dimension cachée incompatible : fichier={meta_hidden}, réseau={self.hidden_size}")
        if meta_output is not None and meta_output != self.output_size:
            raise ValueError(f"Dimension de sortie incompatible : fichier={meta_output}, réseau={self.output_size}")

        if is_legacy_3in:
            print(f"[PioneerNN] Adaptation automatique : modèle à {meta_input} entrées étendu à {self.input_size} entrées (poids de la nouvelle entrée K_θ initialisés à 0).")

        file_bias = json_obj.get("use_bias", False)
        self.use_bias = file_bias

        if TORCH_AVAILABLE:
            self.hidden = nn.Linear(self.input_size, self.hidden_size, bias=self.use_bias)
            self.output = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)

            input_weights = torch.zeros(self.input_size, self.hidden_size)
            num_rows = min(len(json_obj["input_weights"]), self.input_size)
            for i in range(num_rows):
                for j in range(self.hidden_size):
                    input_weights[i][j] = json_obj["input_weights"][i][j]

            output_weights = torch.zeros(self.hidden_size, self.output_size)
            for i in range(self.hidden_size):
                for j in range(self.output_size):
                    output_weights[i][j] = json_obj["output_weights"][i][j]

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

        # Toujours garder une copie Python pour la compatibilité
        if is_legacy_3in:
            py_in = [list(row) for row in json_obj["input_weights"]]
            py_in.append([0.0] * self.hidden_size)
            self._py_input_weights = py_in
        else:
            self._py_input_weights = json_obj["input_weights"]
        self._py_output_weights = json_obj["output_weights"]
        self._py_hidden_bias = json_obj.get("hidden_bias", [0.0] * self.hidden_size)
        self._py_output_bias = json_obj.get("output_bias", [0.0] * self.output_size)

    def save_to_json_file(self, file_path: str) -> None:
        """Sauvegarde les poids dans un fichier JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = self.save_weights_to_json()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json_file(self, file_path: str) -> None:
        """Charge les poids depuis un fichier JSON avec résolution automatique des chemins."""
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
            raise FileNotFoundError(f"Fichier de poids introuvable : {file_path}")
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_weights_from_json(data)
