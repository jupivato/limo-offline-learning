"""
Visualization and reporting suite: convergence curves, trajectory maps, and diagnostic dashboards.
"""

from src.visualization.plot_curves import plot_training_history
from src.visualization.benchmark_visualizer import generate_all_evaluation_reports

__all__ = ["plot_training_history", "generate_all_evaluation_reports"]
