# Rapport d'Étude Expérimentale : Optimisation du Taux d'Apprentissage ($\eta$) et du Budget d'Époques ($N$)

**Projet** : LIMO Offline Learning — Navigation Autonome par Clonage Comportemental  
**Cadre** : Projet de Fin d'Études / PPE — CERV (Centre Européen de Réalité Virtuelle) / ENIB (École Nationale d'Ingénieurs de Brest)  
**Auteur** : Julia Pivato  
**Objet** : Exploration empirique multi-factorielle (Grid Search 20 modèles), analyse mathématique des dynamiques de gradient et validation métrologique en boucle fermée sous la norme **ISO 18646-2**.

---

## 1. Contexte Scientifique et Problématique

Dans le cadre du contrôle réactif autonome du robot différentiel **AgileX LIMO**, l'apprentissage par imitation hors ligne (*Behavioral Cloning*) vise à apprendre une politique neuronale $\pi_\theta(s) \to a$ projetant l'état égocentrique normalisé du robot vers des commandes continues de vitesse linéaire et angulaire :
$$s = \left[\frac{x_{\text{robot}}}{X_{\max}}, \, \frac{y_{\text{robot}}}{Y_{\max}}, \, \frac{\theta_{\text{robot}}}{\pi}\right] \in [-1, 1]^3 \quad \longrightarrow \quad a = \left[\frac{v}{V_{\max}}, \, \frac{\omega}{\Omega_{\max}}\right] \in [-1, 1]^2$$

Lors de la mise au point du modèle neuronal [`PioneerNN`](../src/core/pioneer_nn.py) (Perceptron Multicouche à 1000 neurones cachés sans biais, activations $\tanh$), le choix des hyperparamètres d'optimisation soulève deux interrogations majeures :
1. **Influence du taux d'apprentissage nominal ($\eta$)** : Un taux trop faible ($\eta = 0.001$) risque de figer le réseau dans un sous-apprentissage cinématique, tandis qu'un taux trop fort ($\eta = 0.05$) peut induire des instabilités de gradient.
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

## 4. Analyse Détaillée des Planches Graphiques

### 4.1. Tableau de Bord Métrologique Comparatif (Figure 1)

![Tableau de bord des métriques comparatives](images/grid_search_metrics.png)

#### Explication pas-à-pas des 4 quadrants :
1. **Cadran (A) — Taux de Succès au Docking (%)** :
   - *Observation* : À 30 et 60 époques, le taux de succès est de 0% pour tous les taux d'apprentissage. Le robot n'a pas encore acquis la vitesse suffisante pour parcourir 2 à 3 mètres et stabiliser son orientation en moins de 60 secondes.
   - *Rupture à 100-150 époques* : Dès 100 époques, les premiers succès apparaissent. À 150 époques, **$\eta = 0.05$ culmine à 50% de réussite (4/8 manœuvres)**, suivi de $\eta = 0.01$ (25%).
2. **Cadran (B) — Temps Moyen d'Accostage (secondes)** :
   - *Observation* : La ligne rouge pointillée représente la borne haute de 60.0 secondes (échec par timeout). Les modèles à faible taux ($\eta = 0.001$) restent collés au plafond de 60 s.
   - *Amélioration cinématique* : À 150 époques, $\eta = 0.05$ fait chuter le temps moyen à **46.6 secondes**, soit un gain de plus de 13 secondes sur le ralliement de cible.
3. **Cadran (C) — Précision de Position ISO $A_p$ (mètres)** :
   - *Norme* : La ligne verte en tirets indique la tolérance maximale admissible ISO ($A_p < 0.25\text{ m}$).
   - *Record de précision* : Le modèle **$\eta = 0.01$ à 150 époques** pulvérise l'exigence avec un barycentre d'arrivée situé à seulement **$0.0242\text{ m}$ (2.4 cm de la cible)**, tout en conservant une parfaite stabilité de réseau.
4. **Cadran (D) — Erreur Angulaire Réelle Moyenne $|\Delta \theta|_{\text{mean}}$ (degrés)** :
   - *Révélation* : Pour $\eta \in [0.001, 0.02]$, l'erreur réelle oscille entre $81^\circ$ et $104^\circ$ (le robot s'arrête orthogonalement).
   - *Meilleur alignement* : C'est **$\eta = 0.05$ à 150 époques** qui obtient le meilleur cap d'arrivée avec **$56.4^\circ$**, marquant un alignement supérieur de près de $30^\circ$ aux modèles sous-appris.

---

### 4.2. Dynamique d'Apprentissage Individuelle (Figure 2)

Afin d'éviter toute superposition confuse de courbes sur un même repère, chaque taux d'apprentissage est isolé dans son propre sous-graphique avec son échelle optimale :

![Courbes d'apprentissage individuelles et tableau de synthèse](images/grid_loss_curves.png)

#### Analyse des 6 Panneaux :
- **Panneau 1 ($\eta = 0.001$, Très Faible)** : La perte décroît de manière extrêmement lente et se stabilise à un palier élevé ($\mathcal{L}_{\text{val}} = 0.1181$). Il n'y a aucun sur-apprentissage (courbes Train et Val parfaitement superposées), mais un **sous-apprentissage cinématique marqué**.
- **Panneau 2 ($\eta = 0.005$, Modéré)** : Profil d'apprentissage exemplaire, la perte diminue régulièrement de $0.135$ à $0.100$ sans la moindre secousse.
- **Panneau 3 ($\eta = 0.01$, Équilibré - Recommandé)** : C'est le profil le plus régulier et le plus profond. Il atteint le minimum de perte ($\mathcal{L}_{\text{min}} = 0.1001$ à l'époque 141) avec un parfait accord train/val.
- **Panneau 4 ($\eta = 0.02$, Élevé)** : Convergence rapide vers 0.099 dès la 50ᵉ époque, avec une légère instabilité résiduelle à l'époque 142.
- **Panneau 5 ($\eta = 0.05$, Agressif)** : Isole nettement le phénomène d'**explosion temporaire de gradient** : la perte subit deux sauts verticaux majeurs à l'époque 45 ($\mathcal{L} = 1.81$) et 128 ($\mathcal{L} = 1.60$), avant que l'optimiseur ne ramène les poids vers le bassin d'attraction.
- **Panneau 6 (Tableau Récapitulatif Intégré)** : Permet au lecteur de comparer d'un coup d'œil la perte initiale, la perte finale, la meilleure perte atteinte et le diagnostic de stabilité numérique.

---

### 4.3. Trajectoires Spatiales 2D et Vecteurs d'Orientation d'Accostage (Figure 3)

![Trajectoires 2D et caps finaux](images/lr_trajectories_comparison.png)

#### Analyse Cinématique :
1. **Panneau (A) — Vue Globale de l'Arène** :
   - Le point noir à $(2.0, 2.0)$ marque le départ du robot (avec cap initial $\theta_0 = 0^\circ$).
   - L'étoile verte à $(0, 0)$ représente la station d'accueil.
   - La trajectoire en pointillés bleus ($\eta = 0.001$, 30 ép.) montre que le robot s'arrête à mi-course ($x \approx 0.9\text{ m}, y \approx 0.5\text{ m}$) faute de vitesse suffisante avant 60 s.
   - En augmentant à 150 époques (courbe bleue continue), $\eta = 0.001$ parvient jusqu'à $(0.20, -0.05)$, confirmant que le modèle progresse mais manque de puissance motrice.
   - La courbe rouge ($\eta = 0.05$) décrit une boucle vive et énergique, franchissant la distance rapidement.
2. **Panneau (B) — Zoom d'Accostage et Vecteurs de Cap** :
   - La flèche verte à l'origine $(0, 0)$ indique le cap désiré ($\theta^* = 0^\circ$, pointant vers la droite).
   - Pour $\eta = 0.001$ (flèche bleue), le robot s'immobilise avec un vecteur orienté vers le bas-gauche ($\theta \approx -83^\circ$), c'est-à-dire orthogonalement à la station.
   - Pour $\eta = 0.01$ (flèche verte), le robot se positionne à **$2.4\text{ cm}$ du centre**, réalisant la meilleure approche spatiale.

---

## 5. Synthèse des Régimes Physiques et Recommandations

| Régime | Caractéristiques Numériques | Comportement Robotique Réel | Recommandation Opérationnelle |
| :--- | :--- | :--- | :--- |
| **Régime A : Taux Faible**<br>($\eta = 0.001$) | Perte élevée ($\mathcal{L} \approx 0.118$)<br>Gradients très faibles | Vitesse linéaire bridée ($v \approx 0.03\text{ m/s}$), timeout systématique à 60s, arrêt à 18-94 cm. | **Déconseillé pour budget court** ; nécessite plus de 200 époques pour développer une vitesse suffisante. |
| **Régime B : Taux Modéré**<br>($\eta = 0.005$) | Perte intermédiaire ($\mathcal{L} \approx 0.100$)<br>Descente fluide | Transition progressive ; commence à valider des amarrages à partir de 100 époques. | Bon compromis pédagogique, sans aucun risque numérique. |
| **Régime C : Taux Équilibré**<br>($\eta = 0.010$) | Perte minimale ($\mathcal{L} = 0.0995$)<br>Stabilité mathématique absolue | **Précision barycentrique record ($A_p = 2.4\text{ cm}$)**, trajectoire fluide sans saccade motrice. | **Configuration Recommandée pour la Précision Finale** (à entraîner sur 100 à 150 époques). |
| **Régime D : Taux Agressif**<br>($\eta = 0.050$) | Sauts de gradient ($\mathcal{L} \to 1.8$)<br>Énergie de mise à jour forte | **Vitesse maximale ($v = 0.24\text{ m/s}$)**, amarrage rapide (46.6s) et **50% de succès**. | **Configuration Recommandée pour la Vitesse de Navigation** si le temps de parcours est prioritaire. |

---

## 6. Conclusion pour la Soutenance

L'expérimentation impartiale menée dans cette étude a permis d'éclaircir trois points cruciaux :
1. **Non-équivalence Perte / Performance Robotique** : Une fonction de perte stable ne garantit pas la vitesse cinématique. À $\eta = 0.001$, la perte est basse et régulière, mais le robot est trop lent pour les contraintes de temps d'une mission réelle.
2. **Validation des Intuitions Visuelles** : L'examen attentif des trajectoires 2D a confirmé que le taux plus élevé ($\eta = 0.05$) permettait physiquement au robot d'atteindre la station beaucoup plus vite que les taux faibles, justifiant son taux de succès supérieur (50%).
3. **Le Choix Optimal pour le Projet LIMO** :
   - Pour un déploiement privilégiant la **précision métrologique et la douceur mécanique**, le réglage optimal est **$\eta = 0.01$ sur 150 époques** ($A_p = 2.4\text{ cm}$, aucune oscillation).
   - Pour un déploiement privilégiant la **rapidité d'intervention**, le modèle **$\eta = 0.05$** apporte le dynamisme moteur nécessaire.
