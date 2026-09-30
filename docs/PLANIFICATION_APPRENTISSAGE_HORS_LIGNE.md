# Planification Expérimentale : Apprentissage Hors Ligne pour Robot Mobile AgileX LIMO

## 1. Synthèse de l'Analyse et Cadrage Théorique

### 1.1. Bilan des Travaux Précédents (Rapport de Stage M2)
Le projet précédent a établi les bases de la commande neuronale en boucle fermée sur le robot **AgileX LIMO** sous **ROS 2** :
- **Architecture Logicielle** : Isolation stricte entre le noyau d'apprentissage (`torch_back.py`, `torch_online.py`, `monitoring.py`) et la couche d'interface matérielle (`ros2_limo_interface.py`).
- **Cinématique Différentielle** : Commande en vitesses cartésiennes $[v_{\text{lin}}, v_{\text{ang}}]^T$.
- **Critère de Coût et Orientation Stratégique** :
  $$J = \alpha_x^2 (x - x_d)^2 + \alpha_y^2 (y - y_d)^2 + \alpha_\theta^2 (\theta - \theta_d - \theta_s(x,y))^2$$
  avec $\boldsymbol{\alpha} = [1/6, 1/3, 1/\pi]$ et $\theta_s(x,y) = \tanh(20x)\arctan(2y)$.
- **Limites de l'Apprentissage en Ligne** : Forte sensibilité au taux d'apprentissage ($\eta \ge 0.50$ conduit à des instabilités violentes avec risques matériels sur les réducteurs du robot), dérive odométrique cumulative (*wheel slip*).

### 1.2. La Norme Métrologique ISO 18646-2
La qualification des performances s'appuie sur quatre caractéristiques de pose normalisées :
- **Précision de Position ($A_p$)** : Déviation euclidienne entre la consigne $(x_c, y_c)$ et le barycentre des poses atteintes $(\bar{x}, \bar{y})$.
- **Précision d'Orientation ($A_o$)** : Valeur absolue de l'erreur angulaire moyenne après recentrage dans $(-\pi, \pi]$.
- **Répétabilité de Position ($R_p$)** : Rayon de dispersion $\bar{l} + 3S_l$.
- **Répétabilité d'Orientation ($R_o$)** : Étalement angulaire $3S_o$.

### 1.3. Les Fondements de l'Apprentissage Hors Ligne (Cours ENIB & Littérature)
- Exploitation des **9 manœuvres canoniques de l'ENIB** pour couvrir exhaustivement l'espace d'état différentiel.
- Remplacement du signal de récompense clairsemé du RL (Farias et al., 2020) par une supervision dense issue de trajectoires expertes, réduisant l'effort d'échantillonnage de plusieurs millions d'itérations à quelques milliers d'échantillons.

---

## 2. Architecture de l'Apprentissage Hors Ligne

### 2.1. Principes Directeurs
L'approche retenue est le **clonage comportemental (*Behavioral Cloning*)** :
1. **Génération d'un Jeu de Données Artisanal** : Collecte de trajectoires manuelles lissées (joystick/téléopération) ou synthétiques canoniques, échantillonnées à 20 Hz.
2. **Augmentation des Données** : Exploitation de la symétrie sagittale du châssis différentiel :
   $$(x, y, \theta, v_{\text{lin}}, v_{\text{ang}}) \implies (x, -y, -\theta, v_{\text{lin}}, -v_{\text{ang}})$$
3. **Entraînement Supervisé Hors Ligne** : Optimisation par descente de gradient (AdamW) sur la perte quadratique moyenne (MSE) découplée de la machine physique.
4. **Validation en Boucle Fermée avec Poids Gelés** : Déploiement du modèle sérialisé au format standard JSON sur le simulateur Gazebo et sur le robot réel.

---

## 3. Matrice Expérimentale et Protocole d'Évaluation

- **Cible Fixe** : $(0.0\text{ m}, 0.0\text{ m}, 0.0\text{ rad})$.
- **Positions Initiales Multi-Quadrants ($n \ge 10$ essais)** :
  - Quadrant 1 : $(+2.5, +2.5)$
  - Quadrant 2 : $(-2.5, +2.5)$
  - Quadrant 3 : $(-2.5, -2.5)$
  - Quadrant 4 : $(+2.5, -2.5)$
- **Critères Métrologiques Cibles** :
  - $A_p \le 0.10\text{ m}$
  - $A_o \le 0.15\text{ rad}$
  - $R_p \le 0.25\text{ m}$
  - $R_o \le 0.30\text{ rad}$
