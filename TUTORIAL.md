# Guide Opérationnel : LIMO Offline Learning

Guide pas-à-pas complet et standardisé pour exécuter l'ensemble du projet **AgileX LIMO** : de la génération du jeu de données et de l'entraînement hors-ligne de la PioneerNN jusqu'aux évaluations sous simulation Gazebo 3D et sur le robot physique réel.

> [IMPORTANT]
> **Cible d'Amarrage Unique et Absolue :**
> Dans l'ensemble du projet, la cible terminale d'amarrage est **strictement fixée à $(x=0.0\text{ m}, y=0.0\text{ m}, \theta=0.0\text{ rad})$ avec un cap de $0^\circ$**. Quelle que soit la pose de départ du véhicule dans l'arène, le contrôleur neuronal autonome doit ramener le robot exactement sur ce quai d'amarrage et s'y immobiliser.

---

## Structure Essentielle du Répertoire

Tous les scripts superflus ont été retirés. Seuls les composants indispensables à l'exécution, à l'analyse des résultats finaux et à la génération des graphiques scientifiques sont conservés :

```text
limo-offline-learning/
├── calculus/
│   └── calculate_iso_metrics.py         # Moteur métrologique certifié ISO 18646-2
├── config/
│   └── config.json                      # Paramètres cinématiques et seuils de capture
├── data/
│   ├── dataset.csv                      # Dataset équilibré (45 000 échantillons, 9 classes)
│   └── evaluation/                      # Télémétries, graphiques et benchmarks enregistrés
├── docs/
│   └── images/                          # Graphiques diagnostiques haute résolution (300 DPI)
├── models/
│   ├── pioneer_model_weights.json       # Poids entraînés du réseau PioneerNN (use_bias=True)
│   ├── supervised_w_torch_diff.json     # Poids sérialisés synchronisés
│   └── training_history.json            # Historique de convergence de l'entraînement
├── src/
│   ├── generate_dataset.py              # Générateur du dataset d'apprentissage centré robot
│   ├── train_offline.py                 # Entraînement supervisé hors-ligne de la PioneerNN
│   ├── core/
│   │   └── pioneer_nn.py                # Architecture neuronale MLP (3 -> 1000 -> 2, Tanh, K_theta)
│   ├── ros2/
│   │   ├── limo_interface.py            # Interface matérielle ROS 2 (/odom et /cmd_vel)
│   │   └── show_pose.py                 # Moniteur de pose et de cap en direct dans le terminal
│   ├── evaluation/
│   │   ├── data_logger.py               # Enregistreur télémétrique 20 Hz et planches 3x3
│   │   ├── run_autonomous.py            # Navigation autonome unitaire en boucle fermée
│   │   └── run_batch_benchmark.py       # Banc d'évaluation multi-scénarios ISO 18646-2
│   └── visualization/
│       ├── plot_curves.py               # Tracé des courbes de perte MSE et taux d'apprentissage
│       ├── benchmark_visualizer.py      # Planches de trajectoires et dashboard interactif HTML
│       └── plot_confusion_and_dataset_analysis.py # Matrices de confusion et analyse du dataset
├── TUTORIAL.md                          # Le présent guide pas-à-pas
└── README.md                            # Documentation générale de référence
```

---

## Étape 0 : Préparation de l'Environnement ROS 2

Chaque fois que vous ouvrez un nouveau terminal pour lancer une commande du projet, initialisez les variables d'environnement :

```bash
# 1. Charger l'environnement ROS 2 Humble (ou foxy sur le LIMO COBOT)
source /opt/ros/humble/setup.bash

# 2. Charger les packages du robot AgileX LIMO
source ~/ros2_ws/install/setup.bash

# 3. Se positionner dans le répertoire du projet
cd ~/Documentos/PPE/limo-offline-learning
```

> [!NOTE]
> **Normes des Unités Physiques (Standard ROS REP-103) :**
> - **Vitesse linéaire ($v$)** : en $\text{m/s}$ (vitesse maximale fixée à $0.50\text{ m/s}$).
> - **Vitesse angulaire ($\omega$)** : en $\text{rad/s}$ (vitesse maximale fixée à $1.00\text{ rad/s} \approx 57.3^\circ/\text{s}$).
> - **Orientation ($\theta$)** : en radians dans l'intervalle $[-\pi, +\pi]$.

---

## Étape 1 : Dataset & Entraînement de la PioneerNN

*(Remarque : Le dataset canonique de 45 000 échantillons et les poids optimisés sont déjà inclus et prêts à l'emploi. N'exécutez ces étapes que si vous souhaitez régénérer les données ou réentraîner le modèle).*

### 1.1 Générer le Dataset Canonique (45 000 échantillons équilibrés)
Génère le jeu de données d'apprentissage sur $[-5, +5]\text{ m}$, avec la sous-grille fine d'amarrage, l'entonnoir de décélération progressive ($D_{\text{brake}} = 0.40\text{ m}$), la symétrie sagittale et le raccordement périodique à $\pm\pi$ :
```bash
python3 src/generate_dataset.py
```
* **Fichier produit** : `data/dataset.csv` (exactement 5 000 échantillons par classe pour les 9 manœuvres canoniques de l'ENIB).

### 1.2 Entraîner le Contrôleur Neuronal PioneerNN
Entraîne l'architecture MLP PioneerNN (`3 -> 1000 -> 2`, activation $\tanh$, `use_bias=True`) avec l'optimiseur AdamW et la perte MSE pondérée :
```bash
python3 src/train_offline.py
```
* **Poids sauvegardés** : `models/pioneer_model_weights.json` et `models/supervised_w_torch_diff.json`.

---

## Étape 2 : Simulation sous Gazebo 3D ou en Mode Cinématique

### Option A : Évaluation Cinématique Rapide (Sans interface graphique)
Pour vérifier immédiatement la conformité métrologique ISO 18646-2 sans lancer Gazebo :
```bash
python3 src/evaluation/run_batch_benchmark.py --suite canonical
```
* Exécute les 8 scénarios canoniques (2 par quadrant).
* Calcule $A_p$ (précision de position), $A_o$ (précision d'orientation), $R_p$, $R_o$ et vérifie les 30 secondes d'immobilité au quai $(0, 0, 0)$.

---

### Option B : Simulation Complète sous Gazebo 3D

#### Terminal 1 — Lancer le Simulateur Gazebo
```bash
# Nettoyage des éventuels processus résiduels
killall -9 gzserver gzclient 2>/dev/null

# Lancement du monde Gazebo avec le modèle LIMO différentiel
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch limo_description gazebo_models_diff.launch.py
```
> Attendez l'ouverture de la fenêtre 3D de Gazebo avec le sol gris et le robot au centre `(0, 0, 0)`. Vérifiez que le bouton en bas à gauche est en mode **Play**.

#### Terminal 2 — Lancer le Benchmark ISO 18646-2 sous Gazebo
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/Documentos/PPE/limo-offline-learning

python3 src/evaluation/run_batch_benchmark.py --suite extended --gazebo
```
* Le robot se téléporte successivement sur les 20 positions de départ et navigue de façon 100% autonome vers le quai $(0, 0, 0)$.
* Toutes les trajectoires sont enregistrées dans un sous-dossier horodaté de `data/evaluation/`.

---

## Étape 3 : Déploiement sur le Robot Physique Réel (LIMO)

Sur le robot réel, le contrôleur lit l'odométrie accumulée des roues de façon autonome : **aucune coordonnée manuelle n'est à saisir**.

### 3.1 Terminal 1 sur le Robot — Initialiser le Driver des Moteurs
Placez le robot éteint ou immobile **exactement sur le quai d'amarrage visé (la cible $(0, 0, 0)$ avec une orientation alignée vers l'avant à $0^\circ$)**, puis démarrez le driver :

```bash
# Pour LIMO COBOT (Ubuntu 20.04 / ROS 2 Foxy) :
source /opt/ros/foxy/setup.bash
source ~/limo_ros2_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py

# Ou pour LIMO Standard (Ubuntu 22.04 / ROS 2 Humble) :
source /opt/ros/humble/setup.bash
source ~/agilex_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py
```
*(Laissez ce terminal ouvert en permanence. L'odométrie est initialisée à zéro sur cette position).*

### 3.2 Terminal 2 sur le Robot — Déplacer le Robot au Clavier (Téléopération)
Ouvrez un second terminal pour éloigner manuellement le robot de la cible jusqu'à la position de départ choisie :
```bash
# Pour LIMO COBOT (Ubuntu 20.04 / ROS 2 Foxy) :
source /opt/ros/foxy/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Ou pour LIMO Standard (Ubuntu 22.04 / ROS 2 Humble) :
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**Touches de téléopération :**
- `i` : Avancer
- `,` : Reculer
- `j` : Tourner sur place à gauche
- `l` : Tourner sur place à droite
- `k` : Arrêt d'urgence immédiat

Conduisez le véhicule à l'endroit désiré (ex : à $1.5\text{ m}$ ou $2.0\text{ m}$ de distance), appuyez sur **`k`** pour immobiliser le robot, puis quittez avec **`Ctrl + C`**.

> [!TIP]
> Vous pouvez lancer `python3 src/ros2/show_pose.py` dans un terminal pour visualiser en continu les coordonnées $(X, Y)$ et le cap angulaire du robot.

### 3.3 Terminal 3 sur le Robot — Lancer le Contrôleur Autonome PioneerNN
```bash
# Pour LIMO COBOT (Ubuntu 20.04 / ROS 2 Foxy) :
source /opt/ros/foxy/setup.bash
cd ~/limo-offline-learning

python3 src/evaluation/run_autonomous.py --real --speed-scale 0.6

# Ou pour LIMO Standard (Ubuntu 22.04 / ROS 2 Humble) :
source /opt/ros/humble/setup.bash
cd ~/limo-offline-learning

python3 src/evaluation/run_autonomous.py --real --speed-scale 0.6
```
*(Le facteur `--speed-scale 0.6` fixe la vitesse de croisière à $0.30\text{ m/s}$ pour assurer un guidage précis sans patinage des roues sur le sol de l'arène).*

1. Le script affiche la pose initiale détectée depuis `/odom` :
   ```text
   [Odometry] Current robot pose: x = 1.84 m, y = -0.92 m, theta = 34.2°
   [Ready] Press Enter to start autonomous docking...
   ```
2. Appuyez sur **`[Entrée]`** ! La PioneerNN prend les commandes sur `/cmd_vel`, effectue l'amarrage fluide vers $(0, 0, 0)$, s'arrête dans le rayon de $8\text{ cm}$ et valide les 30 secondes d'immobilité continue requises par la norme ISO 18646-2.

---

## Étape 4 : Analyse des Données Finales & Visualisation Scientifique

Tous les outils de tracé et de diagnostic sont regroupés dans `src/visualization/` et `calculus/` :

### 4.1 Courbes d'Entraînement Hors-Ligne (Perte MSE & Taux d'Apprentissage)
Visualise la décroissance de la fonction de perte pour l'entraînement et la validation :
```bash
python3 src/visualization/plot_curves.py
```
* **Fichiers générés** : `data/evaluation/training_curves.png` et `training_curves.svg`.

### 4.2 Tableau de Bord & Planches Comparatives de Trajectoires
Génère la suite graphique complète à partir des résultats d'un dossier de test :
```bash
python3 src/visualization/benchmark_visualizer.py --dir data/evaluation/test_2026_11_10_16_40_10
```
* **Fichiers générés** :
  - `benchmark_trajectory.png` : Planche de superposition de toutes les trajectoires.
  - `benchmark_all.png` : Dashboard 3x3 complet (trajectoires, fonction de coût, vitesses, gradients et erreurs cartésiennes/angulaires).
  - `benchmark_quadrants.png` : Décomposition des trajectoires selon les 4 quadrants de l'arène.
  - `benchmark_trajectory.svg` : Carte vectorielle haute précision.
  - `benchmark_dashboard.html` : Tableau de bord interactif autonome avec métriques ISO.

### 4.3 Matrices de Confusion & Analyse du Dataset
Évalue les performances de classification de la PioneerNN sur les 45 000 échantillons par rapport au superviseur expert :
```bash
python3 src/visualization/plot_confusion_and_dataset_analysis.py
```
* **Fichiers générés** :
  - `docs/images/matriz_confusao_velocidade_linear.png` (commandes de recul, arrêt, avance).
  - `docs/images/matriz_confusao_velocidade_angular.png` (commandes de rotation horaire, tout droit, anti-horaire).
  - `docs/images/distribuicao_espacial_dataset.png` (carte de densité 2D sur l'arène 5x5m).
  - `docs/images/histogramas_predicao_vs_real.png` (distributions comparatives des vitesses).

### 4.4 Calcul Métrologique Direct ISO 18646-2
Calcule directement les grandeurs métrologiques normalisées ($A_p$, $R_p$, $A_o$, $R_o$) depuis les traces CSV :
```bash
python3 calculus/calculate_iso_metrics.py --dir data/evaluation/test_2026_11_10_16_40_10
```

---

## Résumé des Commandes Utiles & Dépannage

| Action | Commande |
| :--- | :--- |
| **Arrêt d'urgence de tous les processus Gazebo** | `killall -9 gzserver gzclient 2>/dev/null` |
| **Arrêt d'urgence immédiat du robot physique** | `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" --once` |
| **Vérifier le flux de l'odométrie** | `ros2 topic hz /odom` (ou `/wheel/odom`) |
| **Afficher la pose et le cap en temps réel** | `python3 src/ros2/show_pose.py` |
| **Copier les données d'essai du robot vers le PC** | `scp -r agilex@<IP_ROBOT>:~/limo-offline-learning/data/evaluation/test_* ./data/evaluation/` |
