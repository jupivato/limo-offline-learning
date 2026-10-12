# LIMO Offline Learning — Branche `dataset-v2` : Mapping Relatif 5 Colonnes

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble%20%7C%20Foxy-blue.svg)](https://docs.ros.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Norme ISO](https://img.shields.io/badge/Norme-ISO%2018646--2-green.svg)](https://www.iso.org/standard/66847.html)
[![Affiliation](https://img.shields.io/badge/Recherche-CERV%20%2F%20ENIB-purple.svg)](https://www.enib.fr/)
[![Branche](https://img.shields.io/badge/Branche-dataset--v2-orange.svg)](#)

Cette branche implémente la **Version 2** du pipeline d'apprentissage par clonage comportemental (*Behavioral Cloning*) pour le robot mobile **AgileX LIMO**. Elle fournit un environnement complet et reproductible permettant à tout utilisateur de générer son jeu de données, d'entraîner le modèle neuromorphique et de l'évaluer en boucle fermée sous ROS 2 / Gazebo.

---

## 📌 Spécifications Techniques du Pipeline V2

### 1. Schéma de Données Compact (5 Colonnes)
Contrairement aux séries temporelles globales, cette branche utilise un format direct centré sur la cible relative :
- **`x_r`** : Coordonnée X relative par rapport à la cible (m).
- **`y_r`** : Coordonnée Y relative par rapport à la cible (m).
- **`theta_r`** : Écart de cap angulaire recentré dans $[-\pi, \pi]$ (rad).
- **`v_r`** : Consigne experte de vitesse linéaire (m/s).
- **`w_r`** : Consigne experte de vitesse angulaire (rad/s).

### 2. Modèle Cinématique et Génération Analytique
- **Marche avant ($x \le 0$)** : Trajectoires lisses basées sur les **courbes de Dubins** avec rayon de courbure minimal ($R_{\min} = 0.42\text{ m}$).
- **Marche arrière et accostage ($x \ge 0$)** : Manœuvres fluides générées par **splines cubiques d'Hermite**.
- **Stabilisation terminale** : Funil d'alignement angulaire haute densité au quai forçant l'orientation finale vers $\theta = 0$.

### 3. Architecture PioneerNN sans Biais (`use_bias=False`)
Le perceptron multicouche (3 entrées $\to$ 1 000 neurones cachés $\to$ 2 sorties, activation `Tanh`) est configuré sans biais synaptiques :
$$\mathbf{y}(\mathbf{0}) = \tanh(W_2 \cdot \tanh(W_1 \cdot \mathbf{0})) = [0.0\text{ m/s}, 0.0\text{ rad/s}]^T$$
Cette propriété garantit l'arrêt strict à l'équilibre et empêche toute dérive résiduelle au point d'accostage.

### 4. Recalage Odométrique dans ROS 2 ($SO(2)$)
L'interface matérielle/simulation projette les coordonnées odometriques selon la matrice de rotation dans le plan, rendant le guidage invariant quel que soit le cap initial $\theta_0$ de démarrage.

---

## 📂 Organisation des Fichiers

```text
limo-offline-learning/ (branche: dataset-v2)
├── calculus/
│   └── calculate_iso_metrics.py         # Module de calcul des métriques normalisées ISO 18646-2
├── config/                              # Fichiers de configuration
├── data/
│   ├── curated/                         # Dossier cible pour le dataset généré (handcrafted_dataset.csv)
│   ├── evaluation/                      # Dossier cible pour les logs de télémétrie et visualisations
│   └── raw/                             # Démonstrations manuelles optionnelles
├── docs/
│   └── PLANIFICATION_APPRENTISSAGE_HORS_LIGNE.md  # Planification expérimentale
├── models/
│   └── supervised_w_torch_diff.json     # Fichier de destination des poids entraînés (format JSON standard)
├── src/
│   ├── core/
│   │   └── pioneer_nn.py                # Définition du réseau PioneerNN (3->1000->2, Tanh)
│   ├── data/
│   │   ├── generate_rule_based_dataset.py # Générateur analytique Dubins + Hermite (5 colonnes)
│   │   └── curate_dataset.py            # Filtrage et augmentation symétrique
│   ├── evaluation/
│   │   ├── benchmark_visualizer.py      # Générateur de tracés de trajectoires et dashboard web
│   │   ├── data_logger.py               # Enregistreur temps réel de télémétrie (20 Hz)
│   │   ├── run_autonomous.py            # Contrôleur autonome en boucle fermée
│   │   └── run_batch_benchmark.py       # Banc de test multi-scénarios ISO 18646-2
│   ├── ros2/
│   │   └── limo_interface.py           # Interface ROS 2 (/odom et /cmd_vel) avec recalage SO(2)
│   ├── teleop/
│   │   └── collect_demonstrations.py    # Nœud d'acquisition joystick / clavier à 20 Hz
│   └── training/
│       ├── plot_training_curves.py      # Outil de tracé des courbes de perte MSE
│       └── train_offline.py             # Pipeline d'entraînement PyTorch (AdamW + Scheduler)
├── requirements.txt                     # Dépendances Python
└── README.md                            # Documentation de la branche
```

---

## 🚀 Guide d'Utilisation : Exécuter le Pipeline

### 1. Installation des Dépendances
Dans votre environnement virtuel Python :
```bash
pip install -r requirements.txt
```

### 2. Générer le Jeu de Données V2 (5 Colonnes)
Générez l'ensemble des trajectoires analytiques pour l'ensemble des quadrants :
```bash
python3 -m src.data.generate_rule_based_dataset \
    --output data/curated/handcrafted_dataset.csv \
    --stride 2
```

### 3. Lancer l'Entraînement Supervisé Hors Ligne
Entraînez le réseau de neurones et exportez automatiquement le modèle JSON dans `models/` :
```bash
python3 -m src.training.train_offline \
    --dataset data/curated/handcrafted_dataset.csv \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --no-bias \
    --patience 15
```

### 4. Visualiser la Convergence de l'Entraînement
Générez les courbes d'évolution de la fonction de perte (PNG haute résolution et SVG vectoriel) :
```bash
python3 -m src.training.plot_training_curves \
    --history models/training_history.json \
    --outdir data/evaluation
```

### 5. Tester la Navigation en Boucle Fermée (Unitaire)
Lancez la commande neuronale en simulation Gazebo ou sur le robot réel :
```bash
python3 -m src.evaluation.run_autonomous \
    --weights models/supervised_w_torch_diff.json \
    --target 0.0 0.0 0.0
```

### 6. Lancer le Banc de Test Multi-Configurations & Calcul ISO 18646-2
Exécutez la batterie complète des 20 configurations de référence :
```bash
python3 -m src.evaluation.run_batch_benchmark \
    --weights models/supervised_w_torch_diff.json \
    --timeout 60.0 \
    --outdir data/evaluation/batch_run
```

Pour générer la vue synthétique de toutes vos trajectoires enregistrées :
```bash
python3 -m src.evaluation.benchmark_visualizer --batch-dir data/evaluation/batch_run
```

---

## 🧭 Navigation entre les Branches

- **`main`** : Présentation globale du projet et comparaison des types de datasets.
  ```bash
  git checkout main
  ```
- **`dataset-v1`** : Première itération (jeu de données continu 19 colonnes à 20 Hz).
  ```bash
  git checkout dataset-v1
  ```
- **`dataset-v2`** : Branche courante (jeu de données relatif 5 colonnes, sans biais).
  ```bash
  git checkout dataset-v2
  ```
