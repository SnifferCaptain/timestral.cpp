# Architecture du modèle Timestral et modules du dépôt

Ce document présente, dans l'ordre d'un cycle de prédiction, la structure réseau de Timestral, puis explique comment le dépôt C++ découpe ses modules. Les dimensions sont celles de Timestral-5M, tirées du config.json du modèle publié localement (les fichiers de modèle n'entrent pas sous contrôle de version) ; le nombre total de poids déduit de ces dimensions est 5208960, ce qui correspond au nombre de valeurs F32 des poids 5M dans les enregistrements de poids locaux. La sémantique du réseau a été portée de la base amont figée a534bdbd ; les correspondances figurent dans les en-têtes de fichier de chaque source.

## Le problème à résoudre

L'entrée de la prédiction de séries temporelles est l'historique de relevés d'une ou de plusieurs canaux ; la sortie est la valeur des quelques pas futurs. Le modèle de base doit en plus affronter trois difficultés : les unités de mesure de séries différentes peuvent différer de plusieurs ordres de grandeur, l'historique peut contenir des valeurs manquantes, et ce dont l'appelant a besoin est souvent non pas un seul point de prédiction, mais un intervalle de quantiles (sans garantie d'être calibré sur n'importe quelle donnée).

L'approche la plus naïve consiste à traiter chaque point d'échantillonnage comme un token et à prédire point par point en autorégression, à la manière d'un modèle de langage. Cette voie atteint rapidement son plafond sur les séries temporelles : le contexte de la configuration 5M est de 32768 points d'échantillonnage, un token par point ferait atteindre à la matrice d'attention une taille de 32768×32768 ; prédire 512 pas exigerait 512 allers-retours ; et la prédiction ponctuelle n'exprime aucune incertitude.

La réponse de Timestral se résume en quatre points, développés chacun plus loin :

- Découpage par patch : chaque groupe de 32 points d'échantillonnage, avec leurs drapeaux de valeurs manquantes, est compressé en un token ; un contexte de 32768 points devient 1024 tokens de patch.
- Normalisation causale : chaque patch est mis à l'échelle par la moyenne et l'écart-type cumulés jusqu'à sa fin, puis transformé par asinh, ce qui élimine les différences d'échelle sans regarder dans le futur.
- Requêtes de prédiction indépendantes : chaque patch à prédire est représenté par un vecteur de requête appris ; un seul aller-retour produit les 21 quantiles des 32 pas de ce patch.
- Deux axes d'attention : l'axe time effectue l'attention entre les patches du même canal, l'axe variate effectue l'attention entre les différents canaux au même instant ; les deux s'alternent selon la configuration.

## Configuration 5M

| Champ | Valeur | Sens |
| --- | --- | --- |
| `patch_size` | 32 | Nombre de points d'échantillonnage P par patch |
| `d_model` | 192 | Dimension cachée |
| `num_heads` | 3 | Nombre de têtes d'attention, dimension par tête 64 |
| `mlp_hidden_dim` | 512 | Dimension intermédiaire de SwiGLU |
| `block_schedule` | 8 entrées | time×3, time_variate, time×3, time_variate |
| `quantile_knots` | 21 points | 0.01, de 0.05 à 0.95 par pas de 0.05, 0.99 |
| `block_size` | 32 | Nombre maximal de patches décodés par aller-retour |
| `context_length` | 32768 | Nombre de points d'échantillonnage du contexte, soit 1024 patches |
| `rope_partial_factor` | 0.5 | 32 des 64 dimensions de la tête participent à la rotation |
| `time_attn_gate`, `variate_attn_gate` | `sdpa_sigmoid` | La sortie d'attention est multipliée par une porte sigmoid |
| `quantile_width_factor` | 3.0 | Facteur d'espacement des têtes de quantiles |
| `use_asinh` | true | asinh appliqué après normalisation |
| `attend_missing_patches` | true | Les patches manquants restent consultables après la première observation |
| `attn_bias`, `mlp_bias` | false | Aucune biais dans les projections d'attention et MLP |

La lecture et la validation des champs figurent dans [config.hpp](<../../timestral/utils/config.hpp>) ; tout champ inconnu et toute obligation manquante sont rejetés.

![Structure d'un aller-retour de Timestral-5M](<../../assets/arch-overview.svg>)

## Normalisation causale des patches

Chaque ligne (un canal) cumule le long de la dimension temporelle par l'algorithme de Welford, en ne cumulant que les points observés. L'état à la fin du patch p donne le loc et le scale communs à tous les points de ce patch :

$$
\mu_p=\frac{\text{sum}}{\text{count}},\qquad
\sigma_p=\max\left(\sqrt{\frac{M_2}{\max(\text{count}-1,1)}},\ \text{minimum\_scale}\right)
$$

$$
\tilde{x}_t=\operatorname{asinh}\left(\frac{x_t-\mu_p}{\max(\sigma_p,\varepsilon_{\text{FP32}})}\right)
$$

La valeur normalisée des points manquants est fixée à 0 ; l'information de manquant est portée séparément par des bits indicatifs. Les grandeurs cumulées sont conservées en double dans [patch.cpp](<../../timestral/utils/patch.cpp>) ; les commentaires du code précisent que l'ordre d'addition, asinh et minimum_scale sont alignés sur causal_patch_loc_scale amont.

L'essentiel ici est le caractère « causal » : la mise à l'échelle du patch p ne dépend que des observations jusqu'à lui. L'ajout de données en fin de série ne modifie pas les entrées des patches antérieurs ; c'est l'une des conditions de validité du cache de KV paginé décrit plus loin. Chaque page enregistre donc le count, le sum et le M2 à la fin du patch, et le décodage incrémental reprend la cumulation depuis là, sans rescaner tout le préfixe ; voir [Conception de l'inférence](<inference.md>).

Les lignes de covariables known-future font exception : elles sont normalisées par la variance globale de la fenêtre complète (contexte plus horizon) ; une ligne binaire ne prenant que 0 et 1 a loc=0 et scale=1 fixes, sans asinh ; l'implémentation est dans [transformer.cpp](<../../timestral/utils/transformer.cpp#L502-L544>).

## Insertion des patches et requêtes de prédiction

L'entrée de chaque patch est de 2P=64 dimensions : les 32 premières sont les valeurs normalisées, les 32 suivantes sont les indicateurs de manquant (1 si manquant). La couche d'insertion `representation.patch_proj` est un MLP à deux couches avec branche résiduelle ; les trois projections portent des biais :

$$
h=W_o\,\operatorname{SiLU}(W_h u+b_h)+b_o+W_r u+b_r,\qquad u\in\mathbb{R}^{64},\ W_h\in\mathbb{R}^{768\times 64},\ W_o\in\mathbb{R}^{192\times 768}
$$

Les patches à prédire ne passent pas par la couche d'insertion : on y place uniformément un vecteur appris de 192 dimensions, `representation.forecast_query`. Sans covariable, la séquence de tokens d'une ligne est composée de N patches de contexte suivis de H emplacements de requête ; les emplacements de requête se distinguent entre eux par l'encodage de position et l'attention. Cela concorde avec la description du dépôt officiel consignée dans les notes : la prédiction provient de tokens de prédiction indépendants qui lisent l'historique encodé en causal (cette description provient d'un extrait de recherche du dépôt officiel ; l'article en ligne n'a pas été récupéré intégralement).

En présence de covariables, la séquence devient à deux flux : le flux d'encodage contient tous les patches du contexte et de l'horizon (les valeurs futures known-future entrent aussi dans l'encodage), et le flux de décodage place une requête pour chaque emplacement d'encodage ; seules les H dernières sorties de décodage sont envoyées à la tête de quantiles, ce qui correspond à `encodedCount` et `decodedCount` dans le code.

## Planning des blocs et deux axes d'attention

Chaque entrée de `block_schedule` se déploie en un ou deux sous-couches d'attention, voir [patch.cpp](<../../timestral/utils/patch.cpp#L20-L34>) :

| Valeur | Ordre des sous-couches |
| --- | --- |
| `time` | time |
| `variate` | variate |
| `time_variate` | time, puis variate |
| `variate_time` | variate, puis time |

5M compte 8 blocs ; dépliés, ils donnent 8 sous-couches time et 2 sous-couches variate. Chaque sous-couche a sa propre `attn_norms`, son `wQKV`, son `wO` et son `_gate`, et chaque bloc n'a qu'un seul MLP.

L'attention sur l'axe time a lieu entre les emplacements d'une même ligne et porte les dépendances temporelles. L'attention sur l'axe variate a lieu entre les lignes d'un même emplacement, n'autorisant que les lignes d'un même `groupIds` à se voir mutuellement, et porte l'échange d'information entre canaux ; elle n'a besoin d'aucun K/V inter-patch. Le cache paginé ne conserve donc que les K/V des sous-couches time, soit 8 par page pour 5M.

## Masque causal et patches manquants

Le masque de l'axe time est donné par `timeAllowed`, voir [transformer.cpp](<../../timestral/utils/transformer.cpp#L706-L739>). D'abord la définition d'une clé valide : un patch est valide s'il a des observations, ou si l'attention aux manquants est autorisée et que la ligne a déjà produit des observations plus tôt ; dans ce cas il est lisible par les emplacements suivants ([buildKeyValid](<../../timestral/utils/patch.cpp#L167-L188>)). Sans covariable, les règles sont les suivantes :

- Une requête d'encodage ne lit que les clés d'encodage valides ne postériorant pas elle-même ; elle-même est toujours visible.
- Une requête de décodage lit toutes les clés d'encodage de contexte valides, ainsi que les emplacements de décodage ne postériorant pas elle-même.
- Une requête d'encodage ne lit pas le flux de décodage.

![Exemple de masque d'axe time](<../../assets/arch-mask.svg>)

Avec des covariables, il y a trois règles supplémentaires : les requêtes d'encodage des lignes known sont visibles dans les deux sens temporels ; les requêtes du flux de décodage situées dans la section de contexte ne lisent pas le flux d'encodage ; les requêtes de décodage de la section future ne lisent que les clés d'encodage de la section de contexte. L'axe variate a par ailleurs une « cloison » : les lignes de covariables ne peuvent pas lire les lignes cible, les lignes known ne peuvent pas lire les lignes non known ; l'information ne coule que des covariables vers la cible.

Le fait que les lignes known soient visibles dans les deux sens affecte directement le cache : les valeurs futures known peuvent modifier l'état caché des patches d'historique. Les sondes amont figées des notes montrent qu'ajouter 3 à une seule valeur future known modifie le hidden de l'historique de la cible dès la couche 4 ; voir la vérification known-future dans les notes d'acceptation locales. Le chemin conditionnel ne réutilise donc pas le KV d'historique partagé entre requêtes.

## RoPE partiel

Seules les sous-couches time appliquent l'encodage de position rotationnel sur Q et K ; V et les sous-couches variate ne sont pas rotés. Les dimensions participant à la rotation sont la dimension par tête multipliée par `rope_partial_factor`, soit 64×0.5=32 pour 5M. Le regroupement n'est pas par dimensions adjacentes, mais par paires entre la j-ième dimension et la (j+32)-ième, j allant de 0 à 15 :

$$
\theta_j=10000^{-2j/32},\qquad
\begin{bmatrix}x'_j\\ x'_{j+32}\end{bmatrix}=
\begin{bmatrix}\cos p\theta_j & -\sin p\theta_j\\ \sin p\theta_j & \cos p\theta_j\end{bmatrix}
\begin{bmatrix}x_j\\ x_{j+32}\end{bmatrix}
$$

![Disposition du RoPE partiel](<../../assets/arch-rope.svg>)

L'implémentation est dans [kernels.cpp](<../../timestral/reference/kernels.cpp#L128-L162>). Le QKV fusionné des poids publiés est disposé en (qkv, headDim, head), alors que l'attention a besoin de (head, headDim) ; la permutation se fait dans `prepareQkv`, voir [reference/model.cpp](<../../timestral/reference/model.cpp#L35-L60>).

La position p est l'indice d'emplacement moins l'indice du premier patch observé du même groupe. Sans covariable, les emplacements de requête continuent la numérotation après le contexte ; avec covariable, les positions du flux de décodage sont alignées sur celles du flux d'encodage, c'est-à-dire l'emplacement moins le nombre d'emplacements d'encodage. `Settings::continuousRope` peut remplacer l'emplacement entier par une coordonnée réelle convertie depuis l'heure réelle ; c'est une option expérimentale désactivée par défaut, et les exemples des notes n'y montrent pas d'amélioration de précision.

## Attention à porte

Chaque sous-couche d'attention suit une structure pre-norm :

$$
\hat{x}=\operatorname{RMSNorm}(x),\quad
o=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{64}}+M\right)V,\quad
x\leftarrow x+W_O\left(o\odot\sigma(W_g\hat{x})\right)
$$

La porte `_gate` lit le même input normalisé que le QKV et agit élément par élément sur la sortie d'attention concaténée de 192 dimensions. Si la configuration vaut `none`, il n'y a pas de porte, et le contrat des poids n'exige pas non plus `_gate`. L'epsilon de RMSNorm est 1e-6.

## MLP SwiGLU

Le MLP est lui aussi en pre-norm : `w12` projette 192 dimensions vers 1024, on multiplié élément par élément la première moitié par SiLU(seconde moitié) pour obtenir 512 dimensions, puis `w3` projette de retour vers 192 et ajoute au résidu. Cet ordre où « seule la seconde moitié passe par SiLU » est aligné sur l'amont ; inverser les deux donne un résultat entièrement différent, voir [transformer.cpp](<../../timestral/utils/transformer.cpp#L35-L51>).

## Tête de quantiles

Après la dernière couche vient `out_norm` ; l'état caché des emplacements de décodage passe par `head.projection` (un MLP résiduel isomorphe à la couche d'insertion, 192→768→672, sans biais), puis est réorganisé en valeurs brutes r de 32 pas × 21 quantiles. Pour garantir la monotonie des quantiles, le canal médiane r_m est émis tel quel ; les autres canaux accumulent des écarts en softplus :

$$
q_j=r_m-\sum_{i=j}^{m-1}w\,(\tau_{i+1}-\tau_i)\,\operatorname{softplus}(r_i),\quad j<m
$$

$$
q_j=r_m+\sum_{i=m+1}^{j}w\,(\tau_i-\tau_{i-1})\,\operatorname{softplus}(r_i),\quad j>m
$$

où w est le `quantile_width_factor` et τ les points de quantile. Le résultat est borné dans le domaine fini FP32. `predictBlock` effectue ensuite la dé-normalisation : d'abord sinh, puis multiplication par le scale et addition du loc du dernier point de contexte ; les infinis dus à un débordement de sinh sont remplacés par le plus grand ou le plus petit quantile fini à cette position, et par 0 si aucun n'est fini.

## Vérification du nombre de paramètres

Compter élément par élément selon la structure ci-dessus permet de vérifier que la compréhension de la structure ne manque rien :

| Composant | Poids | Nombre de paramètres |
| --- | --- | ---: |
| Insertion des patches | trois projections de patch_proj, avec biais | 210048 |
| Requête de prédiction | forecast_query | 192 |
| Sous-couches d'attention ×10 | attn_norm, wQKV, wO, _gate | 1845120 |
| MLP ×8 | mlp_norm, w12, w3 | 2360832 |
| Normalisation de sortie | out_norm | 192 |
| Tête de quantiles | trois projections de head.projection | 792576 |
| Total | | 5208960 |

Le [constructeur de Transformer](<../../timestral/utils/transformer.cpp#L146-L214>) valide selon la même liste le nom, la forme et la finitude de chaque poids ; il refuse le chargement s'il en manque un ou s'il y en a un de trop.

## Modules du dépôt C++

~~~text
timestral/
├── timestral.hpp/.cpp          # Entrée publique, transfert au service de planification interne
├── utils/                      # Infrastructures et sémantique de modèle commune, indépendantes du backend
│   ├── service.hpp/.cpp        # File de requêtes, trois stratégies de planification, arbre de préfixe partagé
│   ├── engine.hpp/.cpp         # Moteur au niveau tenseur : autorégression par blocs, refine, orchestration du post-traitement
│   ├── condition.hpp/.cpp      # Alignement sur grille des observations intercalées et complétion conditionnelle par segments
│   ├── forecast.hpp/.cpp       # Validation des requêtes et extraction des résultats
│   ├── time.hpp/.cpp           # Heure réelle vers coordonnées de patch réelles
│   ├── transformer.hpp/.cpp    # Orchestration de modèle commune, masques, exécution paginée
│   ├── patch.hpp/.cpp          # Normalisation causale, assemblage des patches, positions, clés de page
│   ├── cache.hpp/.cpp          # Piscine de pages, clés de page FNV, références et candidats LRU
│   ├── model.hpp               # Interface de backend timestral::Model
│   ├── backend.hpp             # Déclaration de loadBackendModel
│   ├── attention.hpp           # Vues d'attention et description des masques
│   └── Fichiers de base restants  # tensor, config, json, safetensors, postprocess, stats
├── reference/                  # Noyaux scalaires, backend par défaut pour les cibles non x86
├── eigen/                      # Backend optionnel Eigen
├── ytensor/                    # Backend optionnel ytensor
└── blas/                       # Backend optionnel BLAS maison
~~~

Les limites de responsabilité sont appliquées selon la norme de code interne du dépôt : le répertoire racine `timestral/` ne contient que les deux fichiers d'entrée ; `utils/` porte tout ce qui est indépendant du backend, y compris planification, cache, normalisation, post-traitement et statistiques ; les répertoires de backend ne contiennent que l'algèbre linéaire et les noyaux propres à ce backend, pas la gestion du cache ni le post-traitement. Il n'y a pas de couche d'opérateurs séparée : les « opérateurs » partagés entre backends sont en réalité de la sémantique de modèle, rattachés à `utils`.

![Découpage des modules et sens des dépendances](<../../assets/arch-modules.svg>)

L'en-tête public [timestral.hpp](<../../timestral/timestral.hpp>) n'expose que `Sample` et `Settings`, `Request`, `Result`, `Policy` imbriqués dans `Timestral` ; les types internes n'apparaissent qu'en déclaration anticipée de `Service`. Les tenseurs, la piscine de pages et la table de rôles restent en interne ; l'appelant ne manipule que des points d'échantillonnage.

## Le backend est choisi à la liaison

Le choix du backend ne passe pas par des macros. [backend.hpp](<../../timestral/utils/backend.hpp>) ne déclare qu'une fonction fabrique `loadBackendModel` ; chaque répertoire de backend fournit une définition dans son `backend.cpp` ; CMake n'en ajoute qu'une au but de bibliothèque au moment de la configuration :

~~~cmake
set(TIMESTRAL_SELECTED_BACKEND timestral/reference/backend.cpp)

if(TIMESTRAL_USE_EIGEN)
    set(TIMESTRAL_SELECTED_BACKEND timestral/eigen/backend.cpp)
endif()
~~~

Ce ci-dessus est un extrait de [CMakeLists.txt](<../../CMakeLists.txt#L19-L35>) ; les branches ytensor et BLAS fonctionnent de la même façon ; activer simultanément deux backends optionnels déclenche une sortie FATAL_ERROR au stade de la configuration. La conséquence est la suivante : sur tout le build, `loadBackendModel` n'a qu'une seule définition, aucun précompilation conditionnelle de backend n'est visible dans les en-têtes publics et le code de planification, et le build par défaut n'inclut pas les en-têtes Eigen ou ytensor. Le prix est qu'un seul backend par build ; pour comparer les backends il faut configurer des répertoires de build séparés.

## Flux de données d'une prédiction

1. À la construction de `Timestral`, `loadEngine` lit config.json et model.safetensors, appelle `loadBackendModel` lié pour obtenir un modèle en lecture seule, puis démarre le thread de travail du `Service`.
2. `submit` accomplit toute la validation dans le thread appelant : au moins 2 patches d'historique, horodatages strictement croissants, déduction de l'intervalle d'échantillonnage et du nombre de pas de prédiction, rognage de l'historique selon `maxContext`, et décision d'utiliser ou non la grille paginée.
3. Le thread de travail retire une tour de requêtes selon la stratégie de planification ; celles qui touchent une page de préfixe passent par le chemin paginé, les autres sont recalculées intégralement ou assemblées à froid.
4. [Engine](<../../timestral/utils/engine.cpp>) remplit à gauche le contexte jusqu'à un multiple entier de patch, puis boucle par blocs de `patchBatch` patches : un aller-retour produit les quantiles, un refine advancing est appliqué, et la médiane de la cible est renvoyée comme contexte du bloc suivant.
5. Après tous les blocs, on applique successivement la troncature de plage, la moyenne symétrique et la troncature non négative, puis on assemble un `Result`, avec le chronométrage et les statistiques de toucher de page.

Les détails de planification, de cache et de post-traitement des étapes 2 à 5 sont dans [Conception de l'inférence](<inference.md>).

## Arbitrages

L'orchestration du modèle n'est écrite qu'une fois. [Transformer](<../../timestral/utils/transformer.hpp>) détient toute la logique de l'insertion, des masques, de la pagination et de la tête de quantiles ; reference et BLAS ne couvrent que des crochets étroits — projections, RMSNorm, disposition QKV, attention —, et une modification de masque ou de pagination ne doit pas être synchronisée dans plusieurs backends. Eigen et ytensor sont des backends connectés plus tôt ; ils conservent chacun une orchestration complète, ne remplaçant que la multiplication matricielle par leur propre bibliothèque et appelant les noyaux reference pour les opérateurs élément par élément ; cette redondance est un coût historique connu, et une modification de sémantique commune doit être vérifiée en trois endroits à la fois.

Le champ `split_context_horizon_attn` est analysé et conservé, mais aucun branche de calcul qui le lit n'a été trouvé dans la source C++ actuelle ; la comparaison du mini-modèle « split désactivé » des notes passe, mais la sémantique exacte de cet interrupteur en amont n'a pas encore été vérifiée ligne par ligne ; il est ici marqué non vérifié.

L'expérience que l'on peut tirer de cette structure est la suivante : la capacité d'un modèle à l'inférence incrémentale dépend principalement de sa structure de dépendance, pas de sa mise en œuvre d'ingénierie. La normalisation causale, le masque time causal et les requêtes de prédiction indépendantes font que les K/V des patches d'historique ne dépendent que de l'historique lui-même, ce qui rend le cache paginé pertinent ; l'attention bidirectionnelle des lignes known et les statistiques sur toute la fenêtre cassent cette propriété, et le chemin correspondant ne peut que tout recalculer. Comprendre le masque décide, bien avant l'optimisation de n'importe quel GEMM, ce que le côté inférence peut faire.
