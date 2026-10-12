# Spécification Technique — Approche V1 : Séries Temporelles 19 Colonnes

**Projet** : LIMO Offline Learning (CERV / ENIB)  
**Approche** : V1 — Séries Temporelles Continues & 9 Manœuvres Canoniques ENIB  
**Dossier du code** : `src/approaches/v1_timeseries/`  
**Dossier des données** : `data/curated/v1/`  
**Dossier des modèles** : `models/v1/`  
**Dossier des évaluations** : `data/evaluation/v1/`  

---

## 1. Principes et Hypothèses de l'Approche V1

L'approche V1 modélise le problème d'apprentissage par clonage comportemental à partir de séries temporelles continues à 20 Hz ($T_s = 50\text{ ms}$) avec des cibles globales :
1. **Intégration cinématique continue** : Simulation des trajectoires pas à pas selon le modèle cinématique différentiel d'Euler.
2. **Couverture des 9 manœuvres canoniques ENIB** : Formalisation explicite des 9 classes de comportements définies par le Prof. Patrick Hénaff.
3. **Multi-cibles cartésiennes** : Échantillonnage de trajectoires visant des points cibles variés dans le repère global.
4. **Perceptron avec termes de biais** : Optimisation d'un modèle `PioneerNN` (3-1000-2) avec `use_bias=True`.

---

## 2. Schéma de Données (19 Colonnes)

| N° | Colonne | Description |
| :---: | :--- | :--- |
| 1 | `timestamp` | Temps continu depuis le départ de la manœuvre (s) |
| 2-4 | `x`, `y`, `theta` | Pose cartésienne instantanée du robot dans le repère mondial |
| 5-7 | `tx`, `ty`, `ttheta` | Consigne cible $(x_{\text{cible}}, y_{\text{cible}}, \theta_{\text{cible}})$ |
| 8-10 | `e_x`, `e_y`, `e_theta` | Écarts cartésiens : $x - tx$, $y - ty$, $\text{recast}(\theta - ttheta)$ |
| 11-13 | `in_0`, `in_1`, `in_2` | Entrées normalisées transmises au réseau PioneerNN |
| 14-15 | `v_lin`, `v_ang` | Vitesses linéaires et angulaires démontrées |
| 16-17 | `v_wheel_left`, `v_wheel_right` | Vitesses différentielles des roues gauche et droite |
| 18 | `maneuver_id` | Identifiant de la manœuvre ENIB (1 à 9) |
| 19 | `quadrant` | Quadrant de départ (1 à 4) |

---

## 3. Guide d'Exécution de la V1

```bash
# 1. Génération du dataset V1
python3 -m src.approaches.v1_timeseries.generate_dataset

# 2. Entraînement hors ligne V1
python3 -m src.approaches.v1_timeseries.train_offline

# 3. Évaluation autonome en boucle fermée
python3 -m src.evaluation.run_batch_benchmark \
    --weights models/v1/supervised_w_torch_diff.json \
    --outdir data/evaluation/v1/batch_run
```
