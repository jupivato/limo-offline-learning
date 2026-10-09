"""
Modèle Réseau de Neurones : PioneerNN
=====================================
Implémentation du perceptron multicouche (MLP) pour la commande
en boucle fermée du robot mobile AgileX LIMO.

Architecture :
- Entrées (3)  : Erreurs cartésiennes normalisées et erreur angulaire corrigée
                 de l'orientation stratégique [in_0, in_1, in_2].
- Cachée (1000): Couche dense avec fonction d'activation Tangente Hyperbolique (Tanh).
- Sorties (2)  : Vitesses de consigne normalisées [v_lin, v_ang] avec activation Tanh.

Compatible PyTorch natif avec repli (fallback) autonome en pur Python standard
pour les environnements où PyTorch n'est pas installé.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import json
import os
import math
from typing import Dict, Any, Union, List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


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
        input_size: int = 3,
        hidden_size: int = 1000,
        output_size: int = 2,
        use_bias: bool = True
    ):
        """
        Initialise le réseau PioneerNN.

        Args:
            input_size (int): Dimension du vecteur d'entrée (défaut: 3).
            hidden_size (int): Nombre de neurones dans la couche cachée (défaut: 1000).
            output_size (int): Dimension du vecteur de sortie (défaut: 2).
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
        else:
            # Poids stockés sous forme de listes Python en mode pur Python
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
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            x = self.activation(self.hidden(x))
            x = self.activation(self.output(x))
            return x

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

        if meta_input is not None and meta_input != self.input_size:
            raise ValueError(f"Dimension d'entrée incompatible : fichier={meta_input}, réseau={self.input_size}")
        if meta_hidden is not None and meta_hidden != self.hidden_size:
            raise ValueError(f"Dimension cachée incompatible : fichier={meta_hidden}, réseau={self.hidden_size}")
        if meta_output is not None and meta_output != self.output_size:
            raise ValueError(f"Dimension de sortie incompatible : fichier={meta_output}, réseau={self.output_size}")

        file_bias = json_obj.get("use_bias", False)
        self.use_bias = file_bias

        if TORCH_AVAILABLE:
            self.hidden = nn.Linear(self.input_size, self.hidden_size, bias=self.use_bias)
            self.output = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)

            input_weights = torch.zeros(self.input_size, self.hidden_size)
            for i in range(self.input_size):
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
        """Charge les poids depuis un fichier JSON."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier de poids introuvable : {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_weights_from_json(data)
