# LIMO Offline Learning : Apprentissage Hors Ligne et Clonage Comportemental pour Robot Mobile AgileX LIMO

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble%20%7C%20Foxy-blue.svg)](https://docs.ros.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Norme ISO](https://img.shields.io/badge/Norme-ISO%2018646--2-green.svg)](https://www.iso.org/standard/66847.html)
[![Affiliation](https://img.shields.io/badge/Recherche-CERV%20%2F%20ENIB-purple.svg)](https://www.enib.fr/)

Ce projet implémente une loi de commande neuronale en boucle fermée pour la stabilisation et le positionnement du robot mobile **AgileX LIMO** sous **ROS 2**. L'approche repose sur l'**apprentissage supervisé par démonstration (*Learning from Demonstration - LfD*) / clonage comportemental (*Behavioral Cloning*)** à partir d'un jeu de données préparé manuellement (*handcrafted dataset*), avec une qualification métrologique conforme à la norme internationale **ISO 18646-2**.

---

## 📌 Présentation Générale

Dans les travaux précédents, l'entraînement du réseau de neurones était effectué en ligne par rétropropagation d'un gradient de coût calculé en temps réel (*online gradient descent*). Bien que fonctionnelle, cette approche présentait des risques mécaniques et d'instabilité en boucle ouverte dès que le taux d'apprentissage variait.

Ce projet propose une alternative robuste et sécurisée :
1. **Découplage Temporel et Sécurité** : L'apprentissage s'effectue entièrement **hors ligne** (*offline*). Aucun gradient n'est rétropropagé pendant le déplacement physique du robot, éliminant tout risque d'oscillations violentes ou de saturation moteur.
2. **Jeu de Données Artisanal Équilibré** : La base d'apprentissage est construite à partir de démonstrations manuelles ou canoniques couvrant les **9 manœuvres élémentaires de l'ENIB**, sur l'ensemble des 4 quadrants et pour des orientations variées.
3. **Compatibilité 1-pour-1 avec la Base Existante** : Conservation stricte de l'architecture du réseau `PioneerNN` (perceptron multicouche 3-1000-2 avec activation `Tanh`), des facteurs de normalisation $\boldsymbol{\alpha}$ et du format de sérialisation des poids en fichier JSON (`input_weights`, `output_weights`).
4. **Qualification Métrologique ISO 18646-2** : Évaluation rigoureuse de la précision de pose ($A_p, A_o$) et de la répétabilité de pose ($R_p, R_o$) en boucle fermée avec des poids gelés.

---

## 📐 Fondements Théoriques et Modélisation

### Modèle Cinématique et Variables d'Entrée
Le châssis différentiel du LIMO est décrit par sa pose cartésienne $\mathbf{x} = [x, y, \theta]^T$ et commandé par le vecteur de vitesses $\mathbf{u} = [v_{\text{lin}}, v_{\text{ang}}]^T$.
Les entrées normalisées du réseau de neurones sont définies par :
$$in_0 = (x - x_d) \cdot \alpha_x$$
$$in_1 = (y - y_d) \cdot \alpha_y$$
$$in_2 = (\text{recast}(\theta - \theta_d) - \theta_s(x, y)) \cdot \alpha_\theta$$

avec les facteurs d'échelle $\boldsymbol{\alpha} = [1/3, 1/3, 1/\pi]$ et la loi d'orientation stratégique ENIB évitant les minima locaux :
$$\theta_s(x, y) = \tanh(10x) \cdot \arctan(1y)$$

### Fonction de Perte Hors Ligne
L'optimisation des poids synaptiques $\mathbf{w}$ minimise l'erreur quadratique moyenne (MSE) entre les commandes prédites et les commandes de démonstration :
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N \left[ \left(\hat{v}_{\text{lin}}^{(i)} - v_{\text{lin}}^{(i)*}\right)^2 + \beta \left(\hat{v}_{\text{ang}}^{(i)} - v_{\text{ang}}^{(i)*}\right)^2 \right]$$

---

## 📂 Architecture du Dépôt

```text
limo-offline-learning/
├── .agents/                    # Compétences et runbooks Antigravity
│   └── skills/
│       ├── limo-ros2-env/
│       ├── limo-dataset-curator/
│       ├── pioneer-offline-trainer/
│       └── iso-18646-evaluator/
├── config/                     # Paramètres d'apprentissage et configurations cibles
├── data/
│   ├── raw/                    # Données brutes de téléopération (.csv)
│   ├── curated/                # Jeux de données nettoyés et équilibrés
│   └── augmented/              # Données augmentées par symétrie bilatérale
├── docs/                       # Planification détaillée et documentation technique
├── models/                     # Poids du réseau exportés au format JSON
├── src/
│   ├── core/                   # PioneerNN (PyTorch) et sérialiseur JSON
│   ├── ros2/                   # Interface ROS 2 (/odom et /cmd_vel à 20 Hz)
│   ├── teleop/                 # Nœud de téléopération et enregistrement (20 Hz)
│   ├── training/               # Entraînement hors ligne (MSE Loss + AdamW)
│   └── evaluation/             # Exécution autonome en boucle fermée
├── calculus/                   # Calcul des métriques ISO 18646-2
├── requirements.txt            # Dépendances Python
└── README.md
```

---

## 🚀 Guide d'Utilisation

### 1. Installation des Dépendances
```bash
pip install -r requirements.txt
```

### 2. Collecte des Démonstrations Manuelles
Téléopérer le robot vers une cible avec une fréquence rigoureuse de 20 Hz :
```bash
python3 -m src.teleop.collect_demonstrations --target 0.0 0.0 0.0
```

### 3. Traitement et Augmentation des Données
Nettoyer les trajectoires et appliquer la symétrie sagittale ($y \to -y$, $v_{\text{ang}} \to -v_{\text{ang}}$) :
```bash
python3 -m src.data.curate_dataset --input-dir data/raw/ --output data/curated/handcrafted_dataset.csv
```

### 4. Entraînement Hors Ligne
Entraîner le perceptron `PioneerNN` et exporter les poids au format JSON standard :
```bash
python3 -m src.training.train_offline --dataset data/curated/handcrafted_dataset.csv --epochs 100
```

### 5. Évaluation Autonome en Boucle Fermée
Exécuter la politique apprise avec poids gelés sous Gazebo ou sur robot réel :
```bash
python3 -m src.evaluation.run_autonomous --weights models/supervised_w_torch_diff.json --target 0.0 0.0 0.0
```

### 6. Benchmark Métrologique ISO 18646-2
Calculer la précision et la répétabilité de pose à partir des fichiers de télémétrie :
```bash
python3 calculus/calculate_iso_metrics.py
```
