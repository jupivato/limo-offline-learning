"""
Module training : pipeline d'entraînement supervisé hors ligne et gestion des pertes.
"""

from src.training.train_offline import DemonstrationDataset, WeightedMSELoss, train_offline

__all__ = ["DemonstrationDataset", "WeightedMSELoss", "train_offline"]
