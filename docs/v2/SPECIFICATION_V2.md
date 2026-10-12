# Spécification Technique — Approche V2 : Mapping Relatif 5 Colonnes

**Projet** : LIMO Offline Learning (CERV / ENIB)  
**Approche** : V2 — Mapping Relatif 5 Colonnes & Trajectoires Analytiques Lisses  
**Dossier du code** : `src/approaches/v2_relative/`  
**Dossier des données** : `data/curated/v2/`  
**Dossier des modèles** : `models/v2/`  
**Dossier des évaluations** : `data/evaluation/v2/`  

---

## 1. Principes et Hypothèses de l'Approche V2

L'approche V2 élimine les instabilités de la V1 grâce à une refonte géométrique et mathématique :
1. **Repère relatif centré sur la cible $(0, 0, 0)$** : Simplifie l'espace d'état en un mapping direct `x_r, y_r, theta_r -> v_r, w_r`.
2. **Génération analytique hybride** :
   - Avance ($x \le 0$) par **courbes de Dubins** ($R_{\min} = 0.42\text{ m}$).
   - Recul et évitement ($x \ge 0$) par **splines cubiques d'Hermite**.
3. **Funil terminal d'accostage haute densité ($R \le 0.15\text{ m}$)** : Force la convergence du cap angulaire vers $\theta = 0$ par commande de rotation pure ($v = 0, \omega = -1.2\theta$).
4. **Architecture sans biais (`use_bias=False`)** : Garantie mathématique absolue d'équilibre au repos $\text{MLP}(\vec{0}) = \vec{0}$, éliminant tout recul ou dérive résiduelle.
5. **Recalage odométrique $SO(2)$** : Projection des coordonnées par matrice de rotation 2D dans l'interface ROS 2.

---

## 2. Schéma de Données (5 Colonnes)

| N° | Colonne | Description |
| :---: | :--- | :--- |
| 1 | `x_r` | Coordonnée cartésienne X relative à la cible (m) |
| 2 | `y_r` | Coordonnée cartésienne Y relative à la cible (m) |
| 3 | `theta_r` | Erreur de cap relative recentrée dans $[-\pi, \pi]$ (rad) |
| 4 | `v_r` | Vitesse linéaire de commande experte (m/s) |
| 5 | `w_r` | Vitesse angulaire de commande experte (rad/s) |

---

## 3. Guide d'Exécution de la V2

```bash
# 1. Génération du dataset V2
python3 -m src.approaches.v2_relative.generate_dataset --stride 2

# 2. Entraînement hors ligne V2 (sans biais)
python3 -m src.approaches.v2_relative.train_offline --epochs 100 --batch-size 256 --no-bias

# 3. Tracé des courbes d'entraînement
python3 -m src.approaches.v2_relative.plot_curves

# 4. Évaluation autonome en boucle fermée
python3 -m src.evaluation.run_batch_benchmark \
    --weights models/v2/supervised_w_torch_diff.json \
    --outdir data/evaluation/v2/batch_run
```

---

## 4. Hyperparamètres et Réglages Expérimentaux

Le pipeline V2 supporte des variations d'architecture et de cinématique :
- **Architecture PioneerNN** :
  - Neurones cachés : `--hidden-size 1000` (par défaut, testable à 256, 512, 2000).
  - Biais : `--no-bias` (recommandé pour $\text{MLP}(\vec{0}) = \vec{0}$) ou `--use-bias`.
- **Optimisation** :
  - Taux d'apprentissage : `--lr 0.001` (AdamW).
  - Poids angulaire : `--beta 1.0` (augmenter à 2.0 pour prioriser le cap terminal).
  - Épocas et batch : `--epochs 100 --batch-size 128`.
- **Normalisation et Cinématique** :
  - Facteurs d'échelle : $\boldsymbol{\alpha} = [1/3, 1/3, 1/\pi]$ dans `train_offline.py`.
  - Résolution spatiale : `--stride 2` ou `--stride 5` dans `generate_dataset.py`.

