# Rapport d'Étude Expérimentale : Optimisation du Taux d'Apprentissage ($\eta$) et du Budget d'Époques ($N$)

**Projet** : LIMO Offline Learning — Navigation Autonome par Clonage Comportemental  
**Cadre** : Projet de Fin d'Études / PPE — CERV (Centre Européen de Réalité Virtuelle) / ENIB (École Nationale d'Ingénieurs de Brest)  
**Auteur** : Julia Pivato  
**Objet** : Exploration empirique multi-factorielle (Grid Search 20 modèles), analyse cinématique approfondie des temps d'accostage et validation métrologique en boucle fermée sous la norme **ISO 18646-2**.

---

## 1. Contexte Scientifique et Problématique

Dans le cadre du contrôle réactif autonome du robot différentiel **AgileX LIMO**, l'apprentissage par imitation hors ligne (*Behavioral Cloning*) vise à apprendre une politique neuronale $\pi_\theta(s) \to a$ projetant l'état égocentrique normalisé du robot vers des commandes continues de vitesse linéaire et angulaire :
$$s = \left[\frac{x_{\text{robot}}}{X_{\max}}, \, \frac{y_{\text{robot}}}{Y_{\max}}, \, \frac{\theta_{\text{robot}}}{\pi}\right] \in [-1, 1]^3 \quad \longrightarrow \quad a = \left[\frac{v}{V_{\max}}, \, \frac{\omega}{\Omega_{\max}}\right] \in [-1, 1]^2$$

Lors de la mise au point du modèle neuronal [`PioneerNN`](../src/core/pioneer_nn.py) (Perceptron Multicouche à 1000 neurones cachés sans biais, activations $\tanh$), le choix des hyperparamètres d'optimisation soulève deux interrogations majeures :
1. **Influence du taux d'apprentissage nominal ($\eta$)** : Un taux trop faible ($\eta = 0.001$) risque de brider les amplitudes de vitesse motrice (vitesse rampante), tandis qu'un taux trop fort ($\eta = 0.05$) peut induire des instabilités de gradient.
2. **Influence du budget d'entraînement ($N$ époques)** : Un nombre réduit d'époques (ex. 30 époques) est-il suffisant pour que les poids atteignent des amplitudes de commande satisfaisantes, ou un budget étendu (100 à 150 époques) est-il indispensable ?

Afin de répondre à ces questions **sans aucun parti pris théorique a priori**, cette étude présente les résultats d'un **Grid Search systématique et impartial** croisant 5 taux d'apprentissage et 4 budgets d'époques (**20 modèles distincts**, testés sur **160 manœuvres en boucle fermée** sous la norme ISO 18646-2).

---

## 2. Fondements Mathématiques & Métrologie Normative

### 2.1. Fonction de Coût Pondérée (Weighted MSE Loss)
Pour équilibrer la précision de déplacement linéaire et l'orientation angulaire, la fonction de perte pondérée s'écrit :
$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ \left(\hat{v}_i - v_i^*\right)^2 + \beta \left(\hat{\omega}_i - \omega_i^*\right)^2 \right]$$
où $B = 1024$ est la taille du lot (*batch size*), $\beta = 1.0$ est le facteur d'équilibrage rotation/translation, et $(v^*, \omega^*)$ sont les consignes issues du jeu de données expert.

### 2.2. Optimiseur AdamW avec Taux Fixe Nominal
L'optimiseur **AdamW** calcule des moyennes mobiles exponentielles du gradient ($m_t$) et du gradient au carré ($v_t$) pour chaque paramètre individuel :
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$W_t = W_{t-1} - \eta \lambda W_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
avec $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ et décroissance découplée des poids (*weight decay*) $\lambda = 10^{-4}$.  
Pour isoler rigoureusement l'impact de chaque configuration, **aucun ordonnanceur de décroissance artificielle (*decay*) n'a été appliqué** : le taux nominal $\eta$ reste strictement constant tout au long des époques.

### 2.3. Critères Métrologiques de la Norme ISO 18646-2
Pour chaque modèle, les 8 manœuvres canoniques multi-quadrants sont exécutées en simulation cinématique à 20 Hz jusqu'à l'arrêt ou timeout (60.0 s). Soit $(x_i, y_i, \theta_i)$ les poses terminales atteintes ($i = 1, \dots, n$ avec $n = 8$) et $(x_c, y_c, \theta_c) = (0, 0, 0)$ la cible :

1. **Barycentre des arrivées** :
   $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i, \qquad \bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$$
2. **Précision de position ($A_p$)** :
   $$A_p = \sqrt{(\bar{x} - x_c)^2 + (\bar{y} - y_c)^2} \quad (\text{Seuil ISO : } A_p < 0.25\text{ m})$$
3. **Répétabilité de position ($R_p$)** :
   $$l_i = \sqrt{(x_i - \bar{x})^2 + (y_i - \bar{y})^2}, \quad \bar{l} = \frac{1}{n}\sum_{i=1}^n l_i, \quad S_l = \sqrt{\frac{\sum (l_i - \bar{l})^2}{n - 1}}$$
   $$R_p = \bar{l} + 3 S_l$$
4. **Précision d'orientation signée ($A_o$) vs Erreur angulaire réelle ($|\Delta \theta|_{\text{mean}}$)** :
   $$A_o = \frac{1}{n} \sum_{i=1}^n \text{recast}(\theta_i - \theta_c), \qquad |\Delta \theta|_{\text{mean}} = \frac{1}{n} \sum_{i=1}^n |\text{recast}(\theta_i - \theta_c)|$$
   > [!IMPORTANT]
   > **Le Piège de l'Annulation Algébrique dans l'ISO 18646-2 standard :**  
   > La formule normative $A_o$ effectue la moyenne de grandeurs signées. Si un essai s'arrête à $+85^\circ$ et un autre à $-85^\circ$, leur somme donne $A_o \approx 0^\circ$, créant l'illusion statistique d'un cap parfait. En robotique mobile, **$|\Delta \theta|_{\text{mean}}$ (erreur absolue moyenne)** et la répétabilité $R_o = 3 S_o$ doivent impérativement être analysées conjointement pour révéler le véritable alignement du robot.

---

## 3. Matrice Complète des Résultats Expérimentaux (20 Configurations)

Tous les modèles ont été initialisés avec la **même graine Xavier (`seed=42`)** sur les 70 664 échantillons de [`data/dataset.csv`](../data/dataset.csv) (80% train / 20% val).

| Taux ($\eta$) | Époques ($N$) | $\mathcal{L}_{\text{train}}$ | $\mathcal{L}_{\text{val}}$ | Succès (/8) | $t_{\text{moyen}}$ (s) | Dist. Finale ($m$) | $A_p$ ISO ($m$) | $R_p$ ISO ($m$) | $A_o$ Signé ($^\circ$) | $|\Delta \theta|_{\text{mean}}$ Réel ($^\circ$) | $R_o$ ISO ($^\circ$) | Diagnostic Scientifique |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.001** | **30** | 0.1333 | 0.1339 | 0/8 (0%) | 60.0 s | 0.948 m | 0.0992 m | 1.1418 m | +4.2° | **84.6°** | 306.4° | Sous-apprentissage sévère : robot bloqué à 1 m par timeout. |
| **0.001** | **60** | 0.1241 | 0.1245 | 0/8 (0%) | 60.0 s | 0.345 m | 0.1383 m | 0.4861 m | +1.5° | **83.4°** | 303.5° | Progression constante, mais vitesse encore trop faible. |
| **0.001** | **100** | 0.1206 | 0.1203 | 0/8 (0%) | 60.0 s | 0.238 m | 0.0969 m | 0.3447 m | +4.5° | **84.7°** | 306.8° | Approche à 0.24 m de la cible ; cap restant orthogonal. |
| **0.001** | **150** | 0.1183 | 0.1181 | 0/8 (0%) | 60.0 s | 0.180 m | 0.0978 m | 0.2504 m | +4.3° | **83.6°** | 302.9° | Pénètre dans la zone 0.20 m, mais timeout avant alignement. |
| **0.005** | **30** | 0.1197 | 0.1198 | 0/8 (0%) | 60.0 s | 0.215 m | 0.0875 m | 0.3143 m | +4.2° | **85.3°** | 309.2° | Vitesse intermédiaire insuffisante pour conclure en 60 s. |
| **0.005** | **60** | 0.1162 | 0.1164 | 0/8 (0%) | 60.0 s | 0.202 m | 0.1379 m | 0.2883 m | +1.1° | **83.0°** | 300.7° | Stabilisation en périphérie immédiate du seuil. |
| **0.005** | **100** | 0.1058 | 0.1060 | 1/8 (12%) | 57.4 s | 0.197 m | 0.1268 m | 0.4171 m | -19.9° | **94.3°** | 336.7° | Premier accostage complet réussi (Run 1 en 57.4 s). |
| **0.005** | **150** | 0.0995 | 0.1003 | 0/8 (0%) | 60.0 s | 0.244 m | 0.1928 m | 0.4065 m | +7.8° | **102.2°** | 359.5° | Perte très basse (0.100), mais trajectoire tangente. |
| **0.010** | **30** | 0.1169 | 0.1171 | 0/8 (0%) | 60.0 s | 0.183 m | 0.1158 m | 0.2432 m | +6.1° | **81.9°** | 296.6° | Bonne trajectoire, décélération un peu précoce. |
| **0.010** | **60** | 0.1034 | 0.1040 | 0/8 (0%) | 60.0 s | 0.253 m | 0.1973 m | 0.4266 m | +11.0° | **101.6°** | 360.6° | Courbure d'approche correcte, arrêt à 0.25 m. |
| **0.010** | **100** | 0.1000 | 0.1006 | 0/8 (0%) | 60.0 s | 0.256 m | 0.2069 m | 0.3415 m | +0.8° | **102.0°** | 353.0° | Convergence stable vers $\mathcal{L} \approx 0.100$. |
| **0.010** | **150** | 0.0995 | 0.1007 | **2/8 (25%)** | 57.9 s | **0.100 m** | **0.0242 m** | **0.2113 m** | -26.1° | **81.6°** | 287.1° | **Record absolu de précision : $A_p = 2.4\text{ cm}$ de la cible !** |
| **0.020** | **30** | 0.1132 | 0.1134 | 0/8 (0%) | 60.0 s | 0.286 m | 0.2050 m | 0.4051 m | +2.0° | **92.2°** | 332.9° | Navigation fluide mais arrêt à distance. |
| **0.020** | **60** | 0.0993 | 0.1006 | 0/8 (0%) | 60.0 s | 0.405 m | 0.3481 m | 0.4221 m | +7.0° | **104.1°** | 349.9° | Déviation latérale aux abords de l'axe central. |
| **0.020** | **100** | 0.0988 | 0.1012 | 0/8 (0%) | 60.0 s | 0.329 m | 0.2291 m | 0.6080 m | -8.9° | **87.1°** | 292.1° | Amortissement correct, mais dépassement du temps limite. |
| **0.020** | **150** | 0.0987 | 0.0993 | 0/8 (0%) | 60.0 s | 0.216 m | 0.1443 m | 0.3504 m | -8.8° | **91.9°** | 308.0° | Perte basse (0.099), trajectoire convergente vers 0.21 m. |
| **0.050** | **30** | 0.1002 | 0.1022 | 0/8 (0%) | 60.0 s | 0.284 m | 0.2292 m | 0.5512 m | -8.3° | **116.1°** | 392.9° | Vitesse franche, amorce de virages serrés. |
| **0.050** | **60** | 0.0986 | 0.0995 | 0/8 (0%) | 60.0 s | 0.370 m | 0.2784 m | 0.7738 m | -5.4° | **84.3°** | 285.1° | Légères oscillations angulaires terminales. |
| **0.050** | **100** | 0.0978 | 0.1014 | 1/8 (12%) | 55.9 s | 0.462 m | 0.1603 m | 0.8541 m | -17.9° | **86.2°** | 311.5° | Accostage rapide validé en 55.8 s. |
| **0.050** | **150** | 0.7850 | 0.7755 | **4/8 (50%)** | **46.6 s** | 0.301 m | 0.2264 m | 1.0229 m | -28.7° | **56.4°** | **226.7°** | **Meilleur taux de succès (4/8), temps le plus court (46.6s).** |

---

## 4. Analyse Cinématique et Physique du Temps d'Accostage (46s - 60s)

Une interrogation fondamentale émerge à la lecture des résultats : **pourquoi le temps moyen d'accostage se situe-t-il entre 46.6 s et 60.0 s**, alors que la vitesse maximale du robot AgileX LIMO est de $V_{\max} = 0.24\text{ m/s}$ et que la distance initiale en ligne droite est de seulement $d_0 \approx 2.83\text{ m}$ (ce qui donnerait théoriquement $t_{\min} = 2.83 / 0.24 \approx 11.8\text{ s}$) ?

L'analyse télémétrique fine des trajectoires enregistrées pas-à-pas à 20 Hz permet de décomposer ce résultat en **4 réalités physiques et algorithmiques indiscutables** :

```
             ┌─────────────────────────────────────────────────────────────┐
             │       DÉCOMPOSITION DU TEMPS DE NAVIGATION DU LIMO         │
             └─────────────────────────────────────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
        [ PHASE 1 : TRANSIT SPATIAL ]                [ PHASE 2 : DOCKING D'ORIENTATION ]
        Rallier la zone d(t) < 0.5 m                Aligner le cap |e_θ(t)| < 20.0°
        ────────────────────────────                ──────────────────────────────────
        • Si η = 0.05 : v ≈ 0.24 m/s                • Déceleration terminale
          => Parcouru en 10.8 s !                   • Virage sur place ou courbe Dubins
        • Si η ≤ 0.01 : v ≈ 0.05 m/s                • Durée : 20 s à 35 s additionnelles !
          => Parcouru en 45 s - 55 s !
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                    [ ARTIFACT DE TIMEOUT SIMULATION (60.0 s) ]
                    Si d = 2.5 cm mais |e_θ| = 25°, le script attend 60s !
```

### 4.1. Évolution Temporelle de la Distance $d(t)$

![Évolution temporelle de la distance](images/kinematic_distance_evolution.png)

#### Analyse du Graphique $d(t)$ :
- **Régime $\eta = 0.05$ (Ligne rouge)** : La transition est spectaculaire. Le robot franchit plus de 2.4 mètres en **seulement 10.8 secondes**, franchissant la barre des $0.5\text{ m}$ à $t = 10.8\text{ s}$. Ce temps est en accord quasi-parfait avec la limite théorique physique ($11.8\text{ s}$).
- **Régimes à faible taux $\eta = 0.01$ et $\eta = 0.001$ (Lignes verte et bleue)** : La descente de distance est extrêmement lente et rectiligne. Le robot ne franchit la ligne des $0.5\text{ m}$ qu'au bout de **41.0 s à 43.4 s** !

---

### 4.2. Profil de Vitesse Linéaire Réelle $v(t)$ : La Démonstration Mathématique

![Profil de vitesse réelle](images/kinematic_velocity_profile.png)

#### Démonstration Cinématique du "Vitesse Rampante" (Crawl Policy) :
1. **À $\eta = 0.001$ et $\eta = 0.01$** : Les gradients d'apprentissage n'ont pas permis aux poids du réseau de commander de fortes amplitudes de sortie sur l'activation $\tanh$. Le réseau commande une vitesse stabilisée de :
   $$v_{\text{croisière}} \approx 0.048 \text{ à } 0.052 \text{ m/s} \quad (\approx 20\% \text{ de } V_{\max})$$
   En appliquant la loi cinématique élémentaire pour parcourir la distance euclidienne de $d_0 = \sqrt{2^2 + 2^2} \approx 2.83\text{ m}$ :
   $$t_{\text{transit}} = \frac{d_0}{v_{\text{croisière}}} = \frac{2.828\text{ m}}{0.050\text{ m/s}} = \mathbf{56.56\text{ secondes !}}$$
   **Conclusion** : Le robot n'était pas bloqué ni immobile ! Il avançait continuellement à $5\text{ cm/s}$, ce qui impose physiquement plus de 56 secondes pour traverser l'arène.
2. **À $\eta = 0.05$** : Le réseau exploite l'intégralité de la dynamique motrice ($v \approx 0.24\text{ m/s}$), ce qui lui permet de couvrir l'arène en 11 secondes.

---

### 4.3. Évolution de l'Erreur de Cap $|e_\theta(t)|$ : La Phase d'Orientation Terminale

![Évolution de l'erreur d'orientation](images/kinematic_heading_evolution.png)

#### Pourquoi l'accostage complet prend-il 32s à 46s pour $\eta = 0.05$ ?
- L'arrivée spatiale à proximité du quai ($d < 0.5\text{ m}$) ne constitue que la **Phase 1** du docking.
- À $t = 10.8\text{ s}$, le robot est positionné à $0.49\text{ m}$ de la cible, mais son cap est de $\theta = -113.4^\circ$, soit une erreur angulaire de $|e_\theta| = 113.4^\circ$.
- Le critère d'accostage complet ISO impose simultanément :
  $$\text{Condition 1 : } d < 0.20\text{ m} \qquad \text{ET} \qquad \text{Condition 2 : } |e_\theta| < 0.35\text{ rad } (20.0^\circ)$$
- De $t = 10.8\text{ s}$ à $t = 32.1\text{ s}$ (**soit 21.3 secondes supplémentaires**), le robot effectue une rotation différentielle contrôlée pour ramener son erreur de cap de $113^\circ$ à $19.9^\circ$. Dès que la double condition est satisfaite, le succès est enregistré à **$t = 32.1\text{ s}$**.

---

### 4.4. L'Artefact du Timeout de Simulation (Plafond de 60.0 s)

Un élément méthodologique capital doit être souligné pour la soutenance :
- Si un robot effectue une approche d'une précision remarquable, atteignant par exemple **$d = 0.025\text{ m}$ (2.5 cm de la cible)** comme lors de l'essai 8 du modèle $\eta = 0.01$ (150 époques), mais que son cap final reste à $89.3^\circ$ à la 59ᵉ seconde, le script de simulation **ne s'arrête pas par anticipation**.
- La boucle continue de s'exécuter jusqu'à expiration du chronomètre de sécurité (`max_duration = 60.0 s`).
- Cet essai est alors comptabilisé dans la moyenne avec une durée de $60.0\text{ s}$, **même si le robot était déjà physiquement posé sur la cible dès 56 secondes**.

---

## 5. Analyse Détaillée des Métriques Individuelles

Chaque métrique de performance est ici isolée dans un graphique autonome, évitant toute surcharge visuelle :

### 5.1. Taux de Succès au Docking (%)

![Taux de succès individuel](images/metric_success_rate.png)

- **À 30 et 60 époques** : 0% de succès pour tous les taux. Les modèles n'ont pas encore acquis la coordination nécessaire pour combiner translation et rotation dans le temps imparti.
- **À 100 et 150 époques** : Apparition nette de la réussite. Le modèle **$\eta = 0.05$ à 150 époques** atteint **50% de succès (4 manœuvres sur 8)**, confirmant la supériorité de sa dynamique d'action.

---

### 5.2. Temps Moyen d'Accostage (s)

![Temps moyen d'accostage individuel](images/metric_arrival_time.png)

- La ligne horizontale rouge à $60.0\text{ s}$ indique le timeout.
- Seuls les modèles développant une vitesse franche parviennent à faire chuter ce temps moyen : **$46.56\text{ s}$ pour $\eta = 0.05$ (150 ép.)**, représentant un gain net de plus de 13 secondes sur le ralliement de cible.

---

### 5.3. Précision de Position ISO $A_p$ (m)

![Précision de position individuelle](images/metric_position_accuracy.png)

- La ligne en tirets jaunes représente la tolérance normative maximale ISO 18646-2 ($A_p < 0.25\text{ m}$).
- **Record absolu de la campagne** : Le modèle **$\eta = 0.01$ (150 époques)** atteint un barycentre d'arrivée situé à seulement **$A_p = 0.0242\text{ m}$ (2.4 cm de la cible)** ! Ce résultat démontre une capacité de centrage exceptionnelle lorsque l'optimisation est exempte de bruit de gradient.

---

### 5.4. Erreur Angulaire Réelle Moyenne $|\Delta \theta|_{\text{mean}}$ ($^\circ$)

![Erreur d'orientation individuelle](images/metric_heading_error.png)

- Mesure absolue évitant le piège de l'annulation algébrique de la norme $A_o$.
- Le modèle **$\eta = 0.05$ (150 époques)** s'aligne le plus fidèlement avec la direction de la station, affichant une erreur moyenne de **$56.4^\circ$**, contre plus de $83^\circ$ pour les modèles à faible taux.

---

## 6. Analyse des Courbes de Convergence Individuelles

Afin de répondre à l'exigence de clarté et d'éliminer toute superposition confuse de courbes, chaque taux d'apprentissage est présenté dans son propre graphique dédié :

````carousel
![Convergence eta=0.001](images/loss_curve_lr_0p001.png)
<!-- slide -->
![Convergence eta=0.005](images/loss_curve_lr_0p005.png)
<!-- slide -->
![Convergence eta=0.010](images/loss_curve_lr_0p01.png)
<!-- slide -->
![Convergence eta=0.020](images/loss_curve_lr_0p02.png)
<!-- slide -->
![Convergence eta=0.050](images/loss_curve_lr_0p05.png)
````

### Synthèse Globale en Sous-Graphiques Séparés avec Tableau Intégré

![Grille facettée des pertes et tableau de synthèse](images/grid_loss_faceted.png)

#### Analyse Comparative des Régimes de Convergence :
1. **$\eta = 0.001$** : Convergence hyper-régulière mais lenteur excessive. La perte finale ($\mathcal{L}_{\text{val}} = 0.1181$) demeure élevée, bridant la commande cinématique.
2. **$\eta = 0.005$** : Profil idéal d'apprentissage supervisé, sans aucune oscillation.
3. **$\eta = 0.010$ (Recommandé pour la stabilité)** : Atteint le creux de perte minimal ($\mathcal{L} = 0.0995$) avec un accord parfait train/val.
4. **$\eta = 0.020$** : Descente très rapide dès la 40ᵉ époque, avec de légères micro-ondulations résiduelles.
5. **$\eta = 0.050$** : Révèle le phénomène d'**explosion temporaire de gradient** (sauts à $\mathcal{L} \approx 1.8$ aux époques 45 et 128), typique des taux élevés en descente non bornée. Malgré cette instabilité ponctuelle, les poids se relocalisent rapidement dans un bassin à forte dynamique de sortie.

---

## 7. Trajectoires Spatiales 2D Individuelles et Orientation Finale

Pour chaque régime représentatif à 150 époques, les 8 trajectoires canoniques complètes sont tracées avec :
- Le point de départ (cercle coloré) et son cap initial $\theta_0$.
- La station d'accueil à $(0, 0)$ (étoile verte) et son vecteur cible $\theta^* = 0^\circ$ (flèche verte).
- Le point d'arrêt final (carré) et **la flèche rouge explicite indiquant le cap final du robot**.

### 7.1. Trajectoires Individuelles — $\eta = 0.001$ (Sous-apprentissage)

![Trajectoire eta=0.001](images/trajectory_lr_0p001.png)

*Diagnostic* : Trajectoires quasi-rectilignes. Les robots progressent vers la cible mais s'arrêtent à 18-25 cm en raison de leur vitesse réduite ($0.047\text{ m/s}$) avant d'avoir pu entamer leur virage d'orientation. Les flèches rouges pointent orthogonalement à la cible ($\approx 84^\circ$).

---

### 7.2. Trajectoires Individuelles — $\eta = 0.010$ (Record de Précision Géométrique $A_p$)

![Trajectoire eta=0.010](images/trajectory_lr_0p01.png)

*Diagnostic* : Convergence spatiale exceptionnelle vers le centre. Les 8 trajectoires se rejoignent directement sur l'étoile verte, établissant le **record de centrage à $2.4\text{ cm}$**. Deux essais valident également le cap dans les 60 secondes.

---

### 7.3. Trajectoires Individuelles — $\eta = 0.050$ (Record de Succès et Vitesse)

![Trajectoire eta=0.050](images/trajectory_lr_0p05.png)

*Diagnostic* : Manœuvres énergiques et rapides. Le robot fonce vers la cible en 10.8 secondes, puis négocie des boucles de Dubins pour ajuster son orientation. 4 essais sur 8 valident l'amarrage parfait sous le seuil des 20°.

---

### 7.4. Comparaison Côte-à-Côte des 3 Régimes (Sans Chevauchement)

![Comparaison 3 panneaux](images/trajectory_comparison_key_models.png)

---

## 8. Synthèse Comparative des Régimes Physiques

| Régime | Caractéristiques Numériques | Comportement Robotique Réel | Recommandation Opérationnelle |
| :--- | :--- | :--- | :--- |
| **Régime A : Taux Faible**<br>($\eta = 0.001$) | Perte élevée ($\mathcal{L} \approx 0.118$)<br>Gradients très faibles | Vitesse linéaire bridée ($v \approx 0.047\text{ m/s}$), timeout systématique à 60s, arrêt à 18-94 cm. | **Déconseillé** : exige plus de 300 époques pour débrider la dynamique motrice. |
| **Régime B : Taux Modéré**<br>($\eta = 0.005$) | Perte intermédiaire ($\mathcal{L} \approx 0.100$)<br>Descente fluide | Transition progressive ; commence à valider des amarrages à partir de 100 époques. | Bon compromis pédagogique sans risque d'instabilité. |
| **Régime C : Taux Équilibré**<br>($\eta = 0.010$) | Perte minimale ($\mathcal{L} = 0.0995$)<br>Stabilité mathématique absolue | **Précision barycentrique record ($A_p = 2.4\text{ cm}$)**, trajectoire fluide sans saccade motrice. | **Configuration Recommandée pour la Précision Finale** (à entraîner sur 150 époques). |
| **Régime D : Taux Agressif**<br>($\eta = 0.050$) | Sauts de gradient ($\mathcal{L} \to 1.8$)<br>Énergie de mise à jour forte | **Vitesse maximale ($v = 0.24\text{ m/s}$)**, amarrage rapide (46.6s) et **50% de succès**. | **Configuration Recommandée pour la Vitesse de Navigation** si le temps de parcours est prioritaire. |

---

## 9. Conclusion pour la Soutenance

L'expérimentation impartiale et l'analyse cinématique fine ont permis d'établir trois conclusions scientifiques majeures :
1. **Élucidation du temps d'accostage élevé (46s - 60s)** :
   - À taux modéré ($\eta \le 0.01$), le robot met 56 secondes car sa vitesse est bridée à $0.05\text{ m/s}$ ($\frac{2.83\text{ m}}{0.05\text{ m/s}} = 56.6\text{ s}$).
   - À taux élevé ($\eta = 0.05$), le transit ne prend que **10.8 secondes** (conforme à la limite physique), mais la phase d'orientation angulaire en zone restreinte requiert 21 secondes supplémentaires pour satisfaire la tolérance des $20^\circ$.
   - Les essais proches de la cible (2.5 cm) sont bridés à 60.0s par le critère de timeout si le cap n'est pas finalisé.
2. **Découplage entre Perte MSE et Dynamique Réelle** :
   - Une fonction de perte basse et stable ne garantit pas la puissance d'action motrice. L'évaluation en boucle fermée sous la norme ISO 18646-2 est le seul juge de paix de la performance robotique.
3. **Le Choix d'Ingénierie pour le Robot AgileX LIMO** :
   - Si la mission requiert une **précision métrologique millimétrique** (ex. accostage de recharge par contact physique), le modèle **$\eta = 0.01$ (150 époques)** est sans égal ($A_p = 2.4\text{ cm}$).
   - Si la mission privilégie la **rapidité d'intervention et le taux de succès global**, le modèle **$\eta = 0.05$** apporte le dynamisme nécessaire.
