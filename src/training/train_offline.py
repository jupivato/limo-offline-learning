"""
Entraînement Supervisé Hors Ligne (Offline Training Pipeline)
============================================================
Entraîne le réseau PioneerNN sur le jeu de données de démonstrations humaines
(handcrafted dataset) selon le critère d'erreur quadratique moyenne (MSE).

Fonctionnalités :
- Classe PyTorch Dataset pour l'ingestion des triplets [in_0, in_1, in_2] -> [v_lin, v_ang].
- Découpage aléatoire en sous-ensembles d'Entraînement (80%) et de Validation (20%).
- Optimiseur AdamW avec décroissance de poids (weight decay) et planificateur de taux.
- Arrêt précoce (Early Stopping) pour garantir l'absence de surapprentissage (overfitting).
- Exportation automatique des poids optimisés au format standard JSON (PioneerNN).
- Sauvegarde de l'historique d'entraînement (pertes train/val) pour analyse graphique.

Projet : LIMO Offline Learning (CERV / ENIB)
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Tuple, Dict, Any, List

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

from src.core.pioneer_nn import PioneerNN


class DemonstrationDataset(Dataset):
    """
    Dataset PyTorch pour charger les échantillons de trajectoires.
    Entrées : in_0 (erreur x), in_1 (erreur y), in_2 (erreur theta corrigée de theta_s).
    Sorties : v_lin (vitesse d'avance), v_ang (vitesse de rotation).
    """

    def __init__(self, csv_file_path: str):
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"Jeu de données introuvable : {csv_file_path}")

        inputs: List[List[float]] = []
        targets: List[List[float]] = []

        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    in_0 = float(row["in_0"])
                    in_1 = float(row["in_1"])
                    in_2 = float(row["in_2"])
                    v_lin = float(row["v_lin"])
                    v_ang = float(row["v_ang"])

                    inputs.append([in_0, in_1, in_2])
                    targets.append([v_lin, v_ang])
                except (KeyError, ValueError):
                    continue

        if not inputs:
            raise ValueError(f"Aucune donnée valide trouvée dans {csv_file_path}")

        self.X = torch.tensor(inputs, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class WeightedMSELoss(nn.Module):
    """
    Fonction de perte MSE pondérée pour équilibrer la vitesse linéaire et angulaire.
    Loss = (v_lin - v_lin*)^2 + beta * (v_ang - v_ang*)^2
    """

    def __init__(self, beta: float = 1.0):
        super(WeightedMSELoss, self).__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_lin = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
        loss_ang = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
        return loss_lin + self.beta * loss_ang


def train_offline(
    dataset_path: str,
    weights_output_path: str = "models/supervised_w_torch_diff.json",
    history_output_path: str = "models/training_history.json",
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    beta: float = 1.0,
    patience: int = 15
) -> Dict[str, Any]:
    """
    Exécute le pipeline d'entraînement supervisé hors ligne.

    Args:
        dataset_path (str): Chemin vers le fichier CSV du dataset.
        weights_output_path (str): Chemin pour sauvegarder les poids JSON optimisés.
        history_output_path (str): Chemin pour sauvegarder la courbe de perte.
        epochs (int): Nombre maximal d'époques d'entraînement.
        batch_size (int): Taille de lot pour DataLoader.
        learning_rate (float): Taux d'apprentissage initial pour AdamW.
        weight_decay (float): Régularisation L2.
        beta (float): Pondération de l'erreur sur v_ang.
        patience (int): Seuil d'époques sans amélioration pour l'arrêt précoce.

    Returns:
        Dict[str, Any]: Métriques finales et historique d'entraînement.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch n'est pas installé dans cet environnement.")

    print("\n" + "=" * 65)
    print("--- DÉMARRAGE DE L'ENTRAÎNEMENT HORS LIGNE (BEHAVIORAL CLONING) ---")
    print(f"Jeu de données       : {dataset_path}")
    print(f"Poids de sortie      : {weights_output_path}")
    print(f"Époques maximales    : {epochs}")
    print(f"Taille de batch      : {batch_size}")
    print(f"Taux d'apprentissage : {learning_rate} (AdamW)")
    print(f"Coefficient bêta     : {beta}")
    print("=" * 65 + "\n")

    # 1. Chargement et partitionnement des données
    full_dataset = DemonstrationDataset(dataset_path)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    train_subset, val_subset = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f"Échantillons : {total_len} au total (Train: {train_len} | Val: {val_len})")

    # 2. Instanciation du modèle, critère et optimiseur
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique de calcul : {device}")
    model = PioneerNN(input_size=3, hidden_size=1000, output_size=2, use_bias=True).to(device)
    criterion = WeightedMSELoss(beta=beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 3. Boucle d'entraînement
    best_val_loss = float("inf")
    best_weights_json = None
    patience_counter = 0

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "lr": []
    }

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        # Phase Entraînement
        model.train()
        running_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(x_batch)

        epoch_train_loss = running_train_loss / train_len

        # Phase Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                running_val_loss += loss.item() * len(x_batch)

        epoch_val_loss = running_val_loss / val_len
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(epoch_val_loss)

        # Enregistrement historique
        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 6))
        history["val_loss"].append(round(epoch_val_loss, 6))
        history["lr"].append(current_lr)

        # Affichage périodique
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Époque {epoch:3d}/{epochs} | "
                f"Loss Train: {epoch_train_loss:.6f} | "
                f"Loss Val: {epoch_val_loss:.6f} | "
                f"LR: {current_lr:.6f}"
            )

        # Sauvegarde du meilleur modèle (Checkpointer)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights_json = model.save_weights_to_json()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Arrêt anticipé à l'époque {epoch} (pas de gain depuis {patience} époques).")
                break

    duration = time.time() - start_training_time
    print(f"\nEntraînement achevé en {duration:.2f} s. Meilleure perte de validation : {best_val_loss:.6f}")

    # 4. Sauvegarde des poids finaux au format JSON canonique
    os.makedirs(os.path.dirname(os.path.abspath(weights_output_path)), exist_ok=True)
    with open(weights_output_path, "w", encoding="utf-8") as f:
        json.dump(best_weights_json, f, indent=2)
    print(f"[Succès] Poids optimisés sauvegardés dans : {weights_output_path}")

    # 5. Sauvegarde de l'historique
    os.makedirs(os.path.dirname(os.path.abspath(history_output_path)), exist_ok=True)
    with open(history_output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Succès] Historique sauvegardé dans : {history_output_path}")

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["epochs"]),
        "duration_seconds": duration,
        "weights_file": weights_output_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement hors ligne pour LIMO.")
    parser.add_argument("--dataset", type=str, default="data/curated/handcrafted_dataset.csv", help="Chemin du dataset")
    parser.add_argument("--output", type=str, default="models/supervised_w_torch_diff.json", help="Chemin du modèle JSON")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=128, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--beta", type=float, default=1.0, help="Pondération de l'erreur angulaire")
    args = parser.parse_args()

    train_offline(
        dataset_path=args.dataset,
        weights_output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta
    )
