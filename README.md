# LIMO Offline Learning — Branche `dataset-v1` : Séries Temporelles 19 Colonnes

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble%20%7C%20Foxy-blue.svg)](https://docs.ros.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Norme ISO](https://img.shields.io/badge/Norme-ISO%2018646--2-yellow.svg)](https://www.iso.org/standard/66847.html)
[![Affiliation](https://img.shields.io/badge/Recherche-CERV%20%2F%20ENIB-purple.svg)](https://www.enib.fr/)
[![Branche](https://img.shields.io/badge/Branche-dataset--v1-blue.svg)](#)

Cette branche héberge la **Version 1** du pipeline d'apprentissage par clonage comportemental (*Behavioral Cloning*) pour le robot mobile **AgileX LIMO**. Elle contient l'environnement basé sur un **jeu de données continu à 19 colonnes** échantillonné à 20 Hz, couvrant les 9 manœuvres canoniques de l'ENIB.

> [!NOTE]
> Cette branche correspond à la première approche méthodologique du projet. Pour utiliser la version stabilisée avec schéma direct à 5 colonnes et réseau sans biais, veuillez basculer sur la branche [`dataset-v2`](https://github.com/jupivato/limo-offline-learning/tree/dataset-v2).

---

## 📌 Spécifications du Pipeline V1

### 1. Structure du Jeu de Données (19 Colonnes)
Le jeu de données est constitué de séries temporelles simulées à 20 Hz comportant 19 attributs :
```text
timestamp, x, y, theta, tx, ty, ttheta, e_x, e_y, e_theta, in_0, in_1, in_2, v_lin, v_ang, v_wheel_left, v_wheel_right, maneuver_id, quadrant
```
- **Cinématique globale** : Poses instantanées $(x, y, \theta)$, cibles $(t_x, t_y, t_\theta)$ et erreurs relatives.
- **Entrées normalisées** : `in_0`, `in_1`, `in_2` combinées avec l'orientation stratégique $\theta_s(x,y) = \tanh(10x)\arctan(y)$.
- **Consignes de commande** : Vitesses de consigne $(v_{\text{lin}}, v_{\text{ang}})$ et vitesses de roues.
- **Classification** : Identifiant de la manœuvre canonique (1 à 9) et quadrant spatial d'origine.

### 2. Les 9 Manœuvres Canoniques de l'ENIB
Le contrôleur expert intègre les 9 comportements fondamentaux de cinématique différentielle (cours Prof. Patrick Hénaff) :
1. **Avance** ($v > 0, \omega = 0$)
2. **Avance Gauche** ($v > 0, \omega > 0$)
3. **Recule Droite** ($v < 0, \omega < 0$)
4. **Recule** ($v < 0, \omega = 0$)
5. **Recule Gauche** ($v < 0, \omega > 0$)
6. **Avance Droite** ($v > 0, \omega < 0$)
7. **Pivote Gauche** ($v = 0, \omega > 0$)
8. **Pivote Droite** ($v = 0, \omega < 0$)
9. **Stop** ($v = 0, \omega = 0$)

### 3. Architecture Réseau PioneerNN
- Modèle : Perceptron multicouche 3 entrées $\to$ 1 000 neurones cachés $\to$ 2 sorties avec activation `Tanh`.
- Biais neuronaux activés (`use_bias=True`).
- Optimiseur AdamW avec fonction de perte MSE pondérée.

---

## 📂 Organisation des Fichiers

```text
limo-offline-learning/ (branche: dataset-v1)
├── calculus/
│   └── calculate_iso_metrics.py         # Calcul métrologique ISO 18646-2
├── data/
│   ├── curated/                         # Emplacement du dataset 19 colonnes généré
│   └── raw/                             # Démonstrations manuelles optionnelles
├── docs/
│   └── PLANIFICATION_APPRENTISSAGE_HORS_LIGNE.md  # Planification expérimentale
├── models/
│   └── supervised_w_torch_diff.json     # Poids PioneerNN au format standard JSON
├── src/
│   ├── core/
│   │   └── pioneer_nn.py                # Définition du réseau PioneerNN
│   ├── data/
│   │   ├── generate_rule_based_dataset.py # Générateur 19 colonnes (9 manœuvres ENIB)
│   │   ├── curate_dataset.py            # Filtrage et symétrie bilatérale
│   │   ├── export_interactive_visualizer.py # Visualiseur web des trajectoires V1
│   │   └── plot_dataset_trajectories.py # Graphiques de distribution des manœuvres
│   ├── evaluation/
│   │   ├── data_logger.py               # Enregistreur temps réel (20 Hz)
│   │   ├── run_autonomous.py            # Contrôleur autonome en boucle fermée
│   │   ├── run_batch_benchmark.py       # Exécution de séries d'évaluation
│   │   └── benchmark_visualizer.py      # Tracé des résultats
│   ├── ros2/
│   │   └── limo_interface.py           # Interface ROS 2 (/odom et /cmd_vel)
│   ├── teleop/
│   │   └── collect_demonstrations.py    # Collecte joystick / clavier
│   └── training/
│       └── train_offline.py             # Pipeline d'entraînement supervisé V1
├── requirements.txt
└── README.md                            # Documentation de la branche
```

---

## 🚀 Guide d'Utilisation : Exécuter le Pipeline V1

### 1. Installation des Dépendances
```bash
pip install -r requirements.txt
```

### 2. Générer le Jeu de Données V1 (19 Colonnes)
```bash
python3 -m src.data.generate_rule_based_dataset \
    --output data/curated/handcrafted_dataset.csv
```

### 3. Lancer l'Entraînement Supervisé
```bash
python3 -m src.training.train_offline \
    --dataset data/curated/handcrafted_dataset.csv \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001
```

### 4. Tester la Navigation en Boucle Fermée
```bash
python3 -m src.evaluation.run_autonomous \
    --weights models/supervised_w_torch_diff.json \
    --target 0.0 0.0 0.0
```

---

## 🧭 Navigation entre les Branches

- **`main`** : Présentation globale du projet et comparaison des types de datasets.
  ```bash
  git checkout main
  ```
- **`dataset-v1`** : Branche courante (séries temporelles 19 colonnes à 20 Hz).
  ```bash
  git checkout dataset-v1
  ```
- **`dataset-v2`** : Version avec schéma relatif 5 colonnes et modèle sans biais.
  ```bash
  git checkout dataset-v2
  ```
