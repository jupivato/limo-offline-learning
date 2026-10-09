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
2. **Génération par Règles Expertes (9 Manœuvres ENIB)** : La base d'apprentissage est construite de façon autonome par un **contrôleur expert à règles conditionnelles (`if / elif / else`)** implémentant formellement les **9 manœuvres canoniques de l'ENIB**, complétée ou substituable par des démonstrations en téléopération manuelle à 20 Hz.
3. **Architecture PioneerNN Épurée (`bias=False`)** : Respect strict du perceptron multicouche 3-1000-2 avec activation `Tanh`. L'exclusion systématique des biais (`bias=False`) garantit la condition physique de repos à l'équilibre $f(0, 0, 0) = [0.0, 0.0]$, la préservation de la symétrie sagittale et l'exportation fidèle de ses **5 000 poids synaptiques** au format standard JSON (`input_weights`, `output_weights`).
4. **Qualification Métrologique ISO 18646-2** : Évaluation rigoureuse de la précision de pose ($A_p, A_o$) et de la répétabilité de pose ($R_p, R_o$) en boucle fermée avec des poids gelés.

---

## 📐 Fondements Théoriques et Modélisation

### Modèle Cinématique et Variables d'Entrée

Le châssis différentiel du LIMO est repéré par sa pose `(x, y, θ)` et commandé par le vecteur de vitesses `(v_lin, v_ang)`.  
Les 3 entrées normalisées transmises au réseau de neurones sont :

- **`in_0`** = `(x - x_cible) × α_x`
- **`in_1`** = `(y - y_cible) × α_y`
- **`in_2`** = `(recast(θ - θ_cible) - θ_s(x, y)) × α_θ`

Avec :
- Facteurs d'échelle : **`α = [1/3, 1/3, 1/π]`**
- Orientation stratégique (évite les minima locaux) : **`θ_s(x, y) = tanh(10·x) · arctan(1·y)`**

### Fonction de Perte Hors Ligne (MSE)

L'entraînement optimise les poids synaptiques en minimisant l'erreur quadratique moyenne entre les vitesses prédites et les vitesses de démonstration :

```text
Loss = (1 / N) · Σ [ (v_lin_pred - v_lin_demo)² + β · (v_ang_pred - v_ang_demo)² ]
```

Où :
- **`v_lin_pred`**, **`v_ang_pred`** : vitesses linéaire et angulaire prédites par le réseau.
- **`v_lin_demo`**, **`v_ang_demo`** : vitesses de consigne issues de la démonstration experte.
- **`β`** : coefficient de pondération angulaire (par défaut `β = 1.0`).
- **`N`** : nombre d'échantillons du lot (*batch*).

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
│   ├── core/                   # PioneerNN (bias=False, 5000 poids) et sérialiseur JSON
│   ├── data/                   # Contrôleur expert à règles et curateur de données
│   │   ├── generate_rule_based_dataset.py  # Générateur if/else (9 manœuvres ENIB)
│   │   └── curate_dataset.py               # Curateur, filtrage statique et miroir
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

### 2. Génération du Jeu de Données par Règles Expertes (Recommandé)
Générer automatiquement des dizaines de milliers d'échantillons couvrant les **9 manœuvres canoniques de l'ENIB** dans les 4 quadrants via l'expert conditionnel `if / elif / else` :
```bash
python3 -m src.data.generate_rule_based_dataset --output data/curated/handcrafted_dataset.csv
```

> **Alternative (Téléopération Manuelle)** :
> Si vous souhaitez capturer des trajectoires au joystick/clavier :
> ```bash
> # A. Enregistrement en direct à 20 Hz vers la cible :
> python3 -m src.teleop.collect_demonstrations --target 0.0 0.0 0.0
> # B. Curage et symétrie sagittale bilatérale :
> python3 -m src.data.curate_dataset --input-dir data/raw/ --output data/curated/handcrafted_dataset.csv
> ```

### 3. Entraînement Hors Ligne (Behavioral Cloning)
Entraîner le perceptron `PioneerNN` (5 000 poids synaptiques, sans biais) et exporter le modèle au format JSON standard :
```bash
python3 -m src.training.train_offline --dataset data/curated/handcrafted_dataset.csv --epochs 100 --batch-size 64
```

### 4. Évaluation Autonome en Boucle Fermée
Exécuter la politique apprise avec poids gelés sous Gazebo ou sur robot réel :
```bash
python3 -m src.evaluation.run_autonomous --weights models/supervised_w_torch_diff.json --target 0.0 0.0 0.0
```

### 5. Benchmark Métrologique ISO 18646-2
Calculer la précision ($A_p, A_o$) et la répétabilité ($R_p, R_o$) de pose à partir de la télémétrie :
```bash
python3 calculus/calculate_iso_metrics.py
```
