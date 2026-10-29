# Références de performance et de qualité

## Données et conditions de mesure

Les données brutes et leurs sources ci-dessous sont enregistrées dans l'espace de travail local du mainteneur, non distribuées avec le dépôt ; les chiffres de cette page sont les résultats enregistrés de ces fichiers.

- CSV brut de prefill `backend-prefill-20261025-v3.csv` : mesures d'origine reprises, aucune donnée ne remplacée
- CSV brut du decode de 1k avec contexte 16k `backend-decode-1k-20261026.csv`
- Enregistrement de recette 1k, hachage de binaire et d'intégrité `backend-short-20261026.json`
- CSV de l'AR par blocs de 0 à 4k par longueur `block-ar-4k-20261026.csv`
- Contrôle officiel 4k, hachages modèle/programme/données `block-ar-4k-20261026.json`
- RMSE combiné et durée moyenne de 4k `block-ar-4k-summary-20261026.csv`
- Source du tracé et confirmation des échecs numériques `block-ar-4k-summary-20261026.json`

CPU AMD Ryzen 7 7735H, GPU AMD Radeon(TM) Graphics du périphérique énuméré 0. La collecte tolère une légère charge quotidienne, sans garantie d'être totalement au repos. La recette s'exécute sériellement ; pendant le chronométrage aucun autre benchmark n'est construit ni exécuté ; la validation Python officielle est exécutée après la fin du chronométrage.

La collecte 1k du backend comprend 16 recettes. BLAS AVX2 comprend 1, 2, 4 threads ; les autres recettes BLAS sans nombre de threads indiqué sont à 1 thread. Vulkan ordinaire et accéléré sont tous conservés ; les budgets de dispatch logique 512/4096 sont aussi dans les données brutes et ne correspondent pas au nombre de EU physiques. La figure principale montre 12 recettes représentatives parmi elles ; les sources du prefill et du decode sont enregistrées séparément, sans passer le nouveau decode pour de l'ancien 16k en changeant le titre.

## Entrées et notation

La comparaison de backends utilise la configuration de signal à huit canaux (entrée d'outil local, non versionnée) sur le premier canal : historique réel index 0 à 16383, futur index 16384 à 17407, monovarié. ar résout 1 patch à chaque fois, 32 tours d'autorégression ; par fait un aller-retours normal de 32 patches d'un coup pour obtenir 1024 points, sans appeler de RoPE Parallel inter-blocs. La comparaison des backends et des pas désactive tous les deux symmetricAveraging/clipNonnegative, en conservant le refine natif et le traitement de plage des quantiles.

La courbe de qualité utilise la configuration à six fonctions (entrée d'outil local, non versionnée), chaque fonction étant une requête monovariée indépendante. Cette tour utilise 16k d'historique et 4k de futur ; le JSON fournit les fonctions, le bruit et les paramètres d'échantillonnage, dont le champ horizon de l'ancien exemple ne détermine pas la longueur réelle de prédiction de cet outil ; la forme du tenseur de 4096 points, le nombre de lignes des données brutes et la longueur de la requête officielle sont tous vérifiés séparément. Les six groupes sont : mono-sinus période 128, double harmonique, sinus lent période 512, mélange périodes 128 et 176, approximation de carré à 5 harmoniques impairs, sinus modulé d'amplitude porteur 128/enveloppe 2048. Intervalle 0.01 s, écart type du bruit 0.03, sans gigue ni bruit de phase, graine 42+index de fonction×10007. Points de départ 16384 et 16448 ; à chaque fois on ne lit que les 16384 points d'observation réels antérieurs ; les vraies valeurs sans bruit du futur ne servent qu'à la notation et n'entrent pas dans le contexte, le masque ou la rétroaction.

## Flux de décodage officiel

Le patch_size de Timestral-5M est 32, le block_size natif est 32, un aller-retours de bloc complet fait 1024 points. Settings.patchBatch public contrôle combien de patches prédit par tour, de 1 à 32 autorisés, au-delà de la taille de bloc du modèle c'est borné, 0 est refusé ; par défaut 16. Pour la tâche 4k, le RMSE combiné est 0.283 à 16 patches et 0.768 à 32 patches ; les moyennes de décodage 4k complet sont respectivement 224 et 166 ms ; les données sont dans la section suivante et le CSV combiné. Entre les blocs on rétroalimente la médiane prédite ; à entrée et budget identiques, le plus petit bloc ne garantit ni plus de précision ni plus de rapidité.

Le par de cette page désigne spécifiquement un aller-retours normal de 32 patches de la tâche 1k, toujours par l'implémentation d'AR par blocs ordinaire ; une requête 4k avec patchBatch=32 prédit normalement quatre blocs. L'ancien RoPE Parallel inter-blocs de futur, préfixe figé et saut de rétroaction inter-blocs, est conservé comme expérience historique par un marquage C++ deprecated, et n'entre pas dans le predict public ni dans les nouvelles figures.

Le forecast officiel active par défaut la moyenne symétrique et la troncature non négative ; cette tour les désactive uniformément pour comparer les pas, ce qui n'est pas équivalent au post-traitement par défaut complet. continuousRope ne contrôle que la coordonnée temporelle ; ce n'est pas la même fonction que le RoPE Parallel inter-blocs abandonné.

## Précision des courtes prédictions et taille de bloc

La [notice principale](<../../README_fr.md#prévisions-courtes-et-taille-de-bloc-de-décodage>) ne porte que [une figure combinée](<../../assets/block-ar-4k-summary-20261026.png>) : à gauche le RMSE combiné multi-fonctions, à droite la durée moyenne de décodage des requêtes 4k complètes. Les pas restent 1, 2, 4, 8, 16, 32 patches, tous avec l'AR par blocs ordinaire.

Chaque trajectoire prédit réellement 4096 points, puis note en cumulé 32, 64, …, 4096 points ; 0 sur l'axe horizontal n'est que le point de départ, sans erreur fabriquée à 0 point. Le RMSE combiné à H points est sqrt(Σ[toutes fonctions, tous points de départ, H premiers points](médiane prédite − vérité)² / Σ nombre d'échantillons). Actuellement 2 points de départ par fonction, le dénominateur est donc 12H, et les fonctions sont équivalentes au niveau de l'erreur quadratique moyenne. **Pas de moyenne directe des RMSE, pas de suppression des fonctions à grande erreur, et pas de sélection d'échantillons par pas différents.** La ligne pointillée bleue fusionne de la même manière l'erreur de prédiction nulle. La note combinée ne résume que ce groupe de fonctions synthétiques de même unité ; ce n'est pas une garantie de précision universelle sur des jeux de données réels ; les résultats par fonction restent dans le CSV brut.

Chaque requête 4k complète est échauffée 1 fois, chronométrée 3 fois pour prendre la médiane, avec rotation de l'ordre des six pas. Le histogramme fait la moyenne arithmétique des 12 durées médianes de 6 fonctions × 2 points de départ, sans rééchantillonner en répétant la même latence par longueur cumulée. Le chronométrage comprend predictStreamed, la construction/copie de la piscine de pages de feedback privée, l'encodage du feedback de prédiction et le post-traitement ; il ne comprend pas le prefill des observations initiales, le chargement, la génération de l'entrée ni la notation. Facteur = durée combinée 1 patch / durée combinée du pas actuel ; 1 patch n'est pas re-marcqué 1.0 ; les millisecondes sont affichées à l'entier ; les valeurs exactes restent dans le CSV. decode_ms_4k désigne toujours la requête complète, pas une latence indépendante de chaque préfixe cumulé.

La collecte est 72 trajectoires, 9216 métriques cumulées, 294912 points de notation de médiane ; le CSV combiné est 6 pas × 128 longueurs = 768 lignes. On a recalculé indépendamment les vraies valeurs JSON, le SSE, les comptes et le RMSE, puis comparé une à une le forecast officiel de même pas sur la base amont figée a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff. Les deux côtés désactivent la moyenne symétrique et la troncature non négative. Le budget d'origine atol1e-4/rtol1e-3 reste inchangé. Après vérification stricte de l'exécution complète, 67/72 trajectoires n'excèdent pas ; 5 autres au total 54/294912 points excèdent, et le verdict reste non entièrement passé. La fonction/point de départ/pas et l'erreur correspondants sont intégralement conservés dans les métadonnées, sans remplacer la courbe mesurée C++ par la courbe officielle. Le tracé utilise une confirmation explicite des échecs, la figure conserve une brève alerte ; le JSON combiné de même nom enregistre cette confirmation et le hachage de l'enregistrement source, sans modifier l'enregistrement de vérification stricte ni son état d'échec. L'ensemble des têtes de quantiles ou des autres backends de ce groupe de requêtes 4k n'a pas été vérifié.

Les données 2k par fonction et les enregistrements numériques anciens (conservés localement) servent d'historique, sans changer le titre pour les passer pour 4k.

## Granularité du prefill

Longueurs 512, 1024, 2048, 4096, 8192, 16384, 32768 ; chaque longueur mesure séparément 1, 2, 4, 8 variables. Chaque groupe construit 3 fois une piscine de pages vide ; le chronométrage va de appendPages à la préparation de toutes les pages de préfixe, en prenant la médiane. Le chargement des poids, la construction du pipeline et l'allocation préalable de la piscine de pages ne sont pas comptés.

[Les résultats monovariés](<../../README_fr.md#durée-de-préremplissage>) sont dans la notice principale ; chaque cellule ci-dessous n'a qu'une valeur en millisecondes.

## Prefill multivarié

### 2 variables

| Longueur de séquence | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan ordinaire | Vulkan accéléré | Vulkan accéléré 512 | Vulkan accéléré 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 99.27 | 28.96 | 19.46 | 9.38 | 9.67 | 4.23 | 3.00 | 2.45 | 8.21 | 4.12 | 4.47 | 4.45 | 4.47 | 4.22 | 26.47 | 6.34 |
| 1k | 179.93 | 61.11 | 38.69 | 18.75 | 19.03 | 8.08 | 5.38 | 4.31 | 15.51 | 8.15 | 9.35 | 8.51 | 6.14 | 5.79 | 49.41 | 10.21 |
| 2k | 358.67 | 126.98 | 90.76 | 53.94 | 42.71 | 16.97 | 13.56 | 10.60 | 32.35 | 17.65 | 18.04 | 17.46 | 10.36 | 7.84 | 47.28 | 12.02 |
| 4k | 782.63 | 313.91 | 223.25 | 130.25 | 135.44 | 39.97 | 25.58 | 20.22 | 77.62 | 40.74 | 42.58 | 42.03 | 21.14 | 14.07 | 112.73 | 24.07 |
| 8k | 1655.37 | 759.91 | 586.45 | 388.80 | 385.30 | 106.83 | 65.76 | 56.61 | 181.20 | 103.62 | 107.55 | 108.59 | 46.00 | 30.00 | 295.15 | 57.96 |
| 16k | 3889.62 | 2136.96 | 1792.12 | 1660.37 | 1555.39 | 312.53 | 187.85 | 139.89 | 470.71 | 308.60 | 320.81 | 324.57 | 112.40 | 83.38 | 860.09 | 149.63 |
| 32k | 10041.90 | 6564.70 | 5971.97 | 5756.42 | 5217.45 | 980.05 | 796.81 | 407.36 | 1386.27 | 1003.04 | 1004.14 | 999.28 | 336.24 | 269.54 | 2912.99 | 455.25 |

### 4 variables

| Longueur de séquence | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan ordinaire | Vulkan accéléré | Vulkan accéléré 512 | Vulkan accéléré 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 187.17 | 58.52 | 37.18 | 16.20 | 16.52 | 7.63 | 4.80 | 3.88 | 15.92 | 7.68 | 8.10 | 8.03 | 6.13 | 6.00 | 48.01 | 10.32 |
| 1k | 357.64 | 119.75 | 82.08 | 37.18 | 36.73 | 15.45 | 12.90 | 8.57 | 30.52 | 15.45 | 17.92 | 16.22 | 9.68 | 6.42 | 43.53 | 10.90 |
| 2k | 737.83 | 277.12 | 202.85 | 109.67 | 90.79 | 33.90 | 23.36 | 16.92 | 65.66 | 35.73 | 35.82 | 35.75 | 19.25 | 11.92 | 94.90 | 22.46 |
| 4k | 1554.77 | 605.84 | 452.34 | 257.57 | 261.02 | 84.14 | 54.74 | 41.65 | 157.25 | 84.13 | 85.95 | 86.00 | 39.17 | 23.18 | 220.13 | 46.51 |
| 8k | 3509.92 | 1559.81 | 1180.40 | 826.31 | 769.28 | 223.00 | 139.43 | 104.73 | 365.61 | 214.35 | 223.38 | 224.15 | 85.57 | 56.13 | 573.43 | 109.56 |
| 16k | 7833.76 | 4219.80 | 3611.78 | 3189.16 | 2800.60 | 609.46 | 359.34 | 236.45 | 925.28 | 605.52 | 627.19 | 631.80 | 218.07 | 165.42 | 1720.01 | 291.73 |
| 32k | 19993.70 | 13212.52 | 12102.66 | 11393.67 | 10490.66 | 2002.03 | 1537.82 | 900.05 | 2792.45 | 1963.34 | 2076.01 | 2046.38 | 646.14 | 526.88 | 5741.83 | 915.63 |

### 8 variables

| Longueur de séquence | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan ordinaire | Vulkan accéléré | Vulkan accéléré 512 | Vulkan accéléré 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 356.82 | 114.24 | 75.75 | 32.30 | 33.17 | 14.56 | 12.21 | 8.81 | 29.68 | 14.83 | 15.73 | 15.54 | 9.87 | 7.80 | 41.80 | 10.89 |
| 1k | 716.13 | 244.31 | 177.06 | 74.91 | 72.89 | 30.96 | 20.41 | 15.39 | 62.06 | 32.18 | 35.31 | 32.95 | 18.69 | 11.61 | 85.41 | 21.02 |
| 2k | 1461.50 | 530.54 | 362.30 | 188.60 | 178.68 | 71.78 | 46.88 | 36.77 | 138.94 | 78.51 | 74.03 | 72.89 | 36.11 | 21.65 | 183.18 | 42.40 |
| 4k | 3048.56 | 1211.35 | 895.08 | 504.51 | 504.80 | 170.25 | 115.18 | 93.95 | 309.33 | 172.51 | 177.54 | 180.27 | 71.63 | 43.02 | 437.19 | 91.02 |
| 8k | 6855.54 | 3022.08 | 2310.21 | 1804.24 | 1800.13 | 441.29 | 260.55 | 181.99 | 720.87 | 428.94 | 448.72 | 443.31 | 167.39 | 111.03 | 1183.78 | 213.05 |
| 16k | 15648.41 | 8308.61 | 7132.50 | 6194.14 | 5539.69 | 1262.54 | 732.38 | 481.52 | 1898.24 | 1265.96 | 1302.86 | 1291.84 | 433.55 | 321.66 | 3475.49 | 587.07 |
| 32k | 40023.28 | 26440.70 | 27766.18 | 22761.36 | 20549.86 | 4029.73 | 3138.08 | 1952.89 | 5701.30 | 4015.34 | 4130.65 | 4114.87 | 1610.08 | 1312.02 | 11623.59 | 1845.87 |

## Granularité du decode

Les nouvelles figures du backend fixent 16k d'historique, 1k de futur, monovarié, ar=1 patch et par=32 patches. Chaque chemin est échauffé 1 fois, chronométré 3 fois pour la médiane, avec alternance de l'ordre, en excluant le prefill de l'historique initial. par n'appelle qu'un seul aller-retours de bloc normal ; ar fait 32 décodages et 31 ajouts de feedback ; l'AR par blocs normal utilise une trajectoire de prédiction privée, sans modifier le préfixe d'observation partagé.

L'API publique utilise aussi Settings.patchBatch, mais cette valeur est figée à la construction, ce n'est pas un item de couverture par Request. Un monovarié à 16k d'historique exige 512 pages d'observation, plus les pages miroir de la moyenne symétrique pour un total de 1024 pages, dans la limite par défaut cachePages=2048. Request.horizonSeconds est en secondes ; 2k à un intervalle de 0.01 s correspond à 20.48 s. Les temps de la figure ne doivent pas être pris directement comme latence bout en bout de toute requête publique.

## Reproduction

Le benchmark de matrices (outil local `example-tools/backendMatrix.cpp`) ajoute un mode short-decode, fixant par défaut 16k/1k, ar1/par32 ; l'ancien mode decode ne conserve que l'expérience historique RoPE abandonnée, sans plus alimenter les nouvelles figures. La collecte de qualité, la vérification officielle et le tracé sont des outils locaux, non versionnés.

~~~powershell
./build-blas/Release/backendMatrix.exe blas-avx2 model/timestral-5m .agents/data/example-tools/benchmark.json short-decode --threads 1
python .agents/data/example-tools/backendPlot.py .agents/data/benchmark/backend-prefill-20261025-v3.csv .agents/data/benchmark/backend-decode-1k-20261026.csv assets/backend-compare-1k-20261026.png
~~~

Le générateur de figures refuse les anciennes données 16k de decode, les pas erronés, les échantillons manquants ou en double. La vérification de qualité et le tracé refusent par défaut les données n'ayant pas passé le budget numérique ; seul un --allow-recorded-deviations explicite permet de conserver les enregistrements d'échec en exportant une figure d'alerte, sans changer la tolérance. Tous les contrastes de courtes prédictions de cette tour passent, sans besoin de cet interrupteur. Le balayage de fenêtre glissante suit les anciennes données, sans relance.

## Diagnostic historique

Ci-dessous ne restent que des entrées de liens anciens, sans valeur d'usage recommandé actuel. Les anciennes limites de multicanal, de 8 variables GPU en plus, de 2 variables ytensor en plus et de longue requête de périphérique 1 restent dans les notes d'acceptation locales ; le passage de cette tour de courtes prédictions ne les couvre pas.

### Limites de qualité de la longue prédiction

Voir [le diagnostic de longue prédiction historique](<forecastHistory.md#limites-de-qualité-de-la-prévision-longue>).

### Courbes de longue prédiction multi-fonctions

Voir [les courbes multi-fonctions historiques](<forecastHistory.md#courbes-de-prévision-longue-multi-fonctions>).

### Expérience complémentaire d'AR 1 patch

Voir [l'expérience historique 1 patch/32 patches](<forecastHistory.md#expérience-complémentaire-ar-mono-patch>).
