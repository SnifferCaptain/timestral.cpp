# Conception du côté inférence

[L'architecture du modèle](<architecture.md>) répond à « qu'est-ce qu'un aller-retour calcule » ; ce document répond à « comment une requête devient un ou plusieurs aller-retours, et quels calculs peuvent être économisés ». Le contenu suit les mécanismes qui existent réellement dans le code : planification, autorégression par blocs et post-traitement, cache de KV paginé, multivarié et complétion conditionnelle, ainsi que l'expérience à fenêtre glissante de préfixe gelé fournie uniquement dans les exemples. Les preuves numériques citent uniformément les notes d'acceptation et de progression locales (non versionnées) ; ce document ne contient aucune re-mesure.

## Du point d'échantillonnage au tenseur

L'interface publique ne manipule que des points d'échantillonnage. `Request::variables` est un `vector<vector<Sample>>` : l'extérieur est le canal, l'intérieur des `Sample{time, value}` ordonnés par temps croissant, NaN signifiant une valeur manquante ; `horizonSeconds` donne le nombre de secondes à prédire à partir de la dernière observation valide ; si `sampleInterval` vaut 0, il est déduit des horodatages. Tous les canaux partagent la même axe temporelle ; quelle portion est condition et quelle portion est à prédre dépend uniquement des positions d'observation propres à chaque canal, sans table de rôles. La déclaration complète est dans [timestral.hpp](<../../timestral/timestral.hpp>).

L'approche naïve consiste à réencoder l'historique complet à chaque requête, puis à prédre bloc par bloc. Elle est simple et sûrement correcte ; le problème, c'est que l'usage typique d'un flux de capteurs est « prédire à chaque petit morceau de nouvelles données », la majeure partie de l'historique est commune entre deux requêtes voisines, et le réencodage redondant pèse le principal coût. L'essentiel de la conception du côté inférence s'articule autour de deux choses : réutiliser les calculs d'historique en garantissant la cohérence avec le recalcul intégral, et fusionner plusieurs requêtes concurrentes en moins d'allers-retours.

## Soumission et planification

![D'une requête à la soumission jusqu'au résultat](<../../assets/infer-flow.svg>)

`submit` accomplit toute la validation et le calcul de grille dans le thread appelant ([Service::prepare](<../../timestral/utils/service.cpp#L218-L285>)) ; le thread de travail ne fait que du calcul numérique et de l'entretien du cache de pages. Ainsi une requête illégale lève une exception immédiatement à la soumission et n'occupe pas la file ; le thread de travail n'a pas à traiter d'erreurs de format. Le service n'a qu'un seul thread de travail ; les poids du modèle sont en lecture seule après construction ; plusieurs threads peuvent soumettre en concurrence.

La sémantique des trois stratégies est la suivante :

- Fifo : à chaque fois, on prend et exécute une requête en tête de file.
- Mailbox : à la soumission, s'il existe déjà dans la file une requête non encore démarrée du même `streamId`, le futur de l'ancienne requête se termine par `RequestSuperseded` et elle quitte la file ; seule la plus récente est conservée ; le passage à Mailbox déduplique aussi la file existante par origine. Les requêtes anonymes (`streamId` vide) reçoivent chacune une identité unique et ne sont pas dédupliquées entre elles. Adapté aux flux de capteurs « qui ne se soucient que des données les plus récentes ».
- Continuous : à partir de la tête de file, on prend en FCFS de manière continue les requêtes concaténables en lot, au maximum `maxBatch` requêtes, on s'arrête à la première requête non concaténable, sans sauter et sans préempter. Une bouée de requêtes est exécutée ensemble, libère les emplacements ensemble, sans état intermédiaire par token.

Les conditions pour que deux requêtes soient concaténables sont dans [Service::batchable](<../../timestral/utils/service.cpp#L368-L408>) : même nombre de canaux, même longueur d'historique, même nombre de pas, même intervalle, même usage de la grille paginée et mêmes deux interrupteurs de post-traitement ; les requêtes de complétion conditionnelle exigent en plus que le plan des observations soit point par point identique. `abort(streamId)` invalide toutes les requêtes non démarrées de cette origine et rend leurs références dans le cache de pages.

Après extraction d'une bouée, [executeRound](<../../timestral/utils/service.cpp#L410-L521>) les répartit en trois voies : une seule requête s'exécute individuellement ; plusieurs requêtes ayant touché le même tableau de pages forment un groupe de décodage partagé ; les autres requêtes non touchées sont concaténées à froid en un seul aller-retour `(B,V,T)`.

## Grille et contexte

Le chemin standard exige que tous les canaux aient des horodatages identiques et une valeur valide à la fin. À la soumission, on conserve les échantillons les plus récents selon `maxContext` (défaut 0, prend le `context_length` du modèle) ; un historique de moins de 2 patches est rejeté directement. Deux grilles existent ensuite :

- Le chemin intégral remplit à gauche l'historique conservé jusqu'à un multiple entier de patch ; les positions de remplissage sont marquées non observées.
- Le chemin paginé aligne à gauche l'historique sur les frontières de patch ; le préfixe de patches entiers entre dans les pages ; la queue incomplète d'un patch n'entre pas dans les pages et est encodée avec le horizon au moment du décodage. L'alignement à gauche garantit que lorsque la même origine appuie des données, les frontières des patches antérieurs ne changent pas, si bien que les clés de pages peuvent toucher.

L'usage de la grille paginée est décidé à la soumission : le backend prend en charge l'interface incrémentale ; en mode RoPE continu, l'historique doit être un multiple entier de patch ; après complétion, ne pas dépasser `maxContext` ; le nombre de patches entiers ne pas dépasser `cachePages`. Les 2048 pages par défaut couvrent 65536 points d'échantillonnage, ce qui dépasse le contexte de 32768 points de 5M ; par conséquent, avec les réglages par défaut, tous les historiques monocanaux passent par le chemin paginé ; la décision est dans [le code de décision](<../../timestral/utils/service.cpp#L361-L366>). Le pass miroir de la moyenne symétrique prend le préfixe négatif et construit une chaîne de pages séparée, partageant la piscine de pages avec les pages positives ; en cas de capacité insuffisante, seul le pass miroir est recalculé intégralement.

## Autorégression par blocs et refine

Le modèle décode au maximum `block_size` patches d'un coup (32 pour 5M, soit 1024 pas). `Settings::patchBatch` vaut 16 par défaut, borné à `block_size` à la construction. Si l'horizon est plus long, [rawForecast](<../../timestral/utils/engine.cpp#L180-L426>) boucle par blocs :

1. Un aller-retour sur un bloc avec le contexte actuel ; `decodePages` est appelé en pagination, sinon `predictBlock`.
2. Pour les blocs de plus de 1 patch, un refine advancing : le cadre de sortie de chaque patch prédit avance avec le contenu déjà généré ; le premier patch reprend le cadre du dernier patch de contexte ; les patches suivants replient la médiane sur l'état après fusion de préfixe Welford et dé-normalisent les autres quantiles. La fusion de préfixe est faite à la manière de Hillis-Steele, préservant l'ordre associatif des fusions deux à deux amont.
3. La médiane des lignes cibles est concaténée à la fin du contexte comme entrée du bloc suivant, marquée observée ; une médiane d'absolue valeur supérieure ou égale à 1e30 ou non finie est renvoyée comme 0. Les lignes de covariables known renvoient les valeurs futures réelles, les lignes de covariables past renvoient du vide.
4. Quand le contexte dépasse `maxContext`, la fenêtre glisse à gauche ; à ce moment la grille du tableau de pages devient invalide, on libère le tableau de pages et les blocs restants sont recalculés intégralement.

L'autorégression par blocs signifie que l'erreur s'accumule sur un horizon long. Le cas réel 5M de 1030 pas des notes présente une erreur absolue maximale de 0.310 par rapport à l'amont, contre un RMS de sortie de référence d'environ 854, bien plus grand que les cas courts ; on ne peut pas en conclure que la précision des horizons courts et longs est équivalente.

## Post-traitement des quantiles

Après la boucle de blocs, [Engine::computePrediction](<../../timestral/utils/engine.cpp#L497-L628>) fait successivement trois étapes, l'algorithme est dans [postprocess.cpp](<../../timestral/utils/postprocess.cpp>) :

- Troncature de plage : toutes les sorties de chaque trajectoire cible sont bornées dans l'intervalle « min et max des observations du contexte initial, étendus chacun de 1e4× le dernier scale vers l'extérieur ». Les seuils 1e4 et le seuil de rétroalimentation 1e30 sont des sémantiques fixes de post-traitement, non exposés comme options publiques.
- Moyenne symétrique : si `symmetricAveraging` est vrai, on négatifie l'entrée et prédit une fois de plus intégralement ; les quantiles du résultat miroir sont renversés, négatifs puis moyennés avec le résultat d'origine, soit `q[i] = 0.5 × (q[i] − m[Q−1−i])`. L'allers-retours miroir utilise une piscine de pages vide et ne passe pas par le cache ; l'activer fait qu'une partie des calculs d'une requête ne peut être économisée par le cache de préfixe.
- Troncature non négative : si `clipNonnegative` est vrai et que toutes les observations du contexte de ce canal sont non négatives, la sortie est tronquée au-dessus de 0.

Toute sortie NaN est signalée par exception ; aucune fausse prédiction n'est retournée. `Result` range 21 quantiles par (canal, pas, niveau) ; `quantile(step, level)` interpole linéairement entre points de quantile adjacents et prend les extrémités hors grille ; `lower` et `upper` donnent les deux bornes de l'intervalle central, par exemple level=0.9 prend les quantiles 0.05 et 0.95 ; `isObserved` distingue observations d'origine et valeurs générées.

## Cache de KV paginé

D'abord préciser ce qui est cacheable. Sur le chemin standard, toutes cibles et sans covariable, trois conditions sont simultanément vérifiées : la normalisation est causale, le masque d'encodage de l'axe time est causal, la requête de prédiction n'entre pas dans le flux d'encodage. Donc le K/V du patch p dans chaque sous-couche time ne dépend que des entrées des p premiers patches ; les données suivantes ne le modifient pas. L'attention variate a lieu seulement entre canaux au sein d'un même slot, sans K/V inter-patch ; mais elle fait qu'un changement d'un canal influence les K/V de haut niveau des autres canaux du même groupe ; une page doit donc conserver le groupe entier de canaux, et ne peut pas toucher indépendamment par canal unique.

![Cache de préfixe paginé par adresse de contenu](<../../assets/infer-pages.svg>)

La définition d'une page est dans [cache.hpp](<../../timestral/utils/cache.hpp>) : une page est le K/V d'un patch, d'un groupe de V lignes, dans toutes les sous-couches time, plus les bits de masque de chaque ligne, les positions RoPE et les cumulés de normalisation à la fin du patch. Géométriquement pour 5M, le K/V par ligne et par page est 8 sous-couches × 2 × 192 FP32, soit 12 Ko, hors métadonnées ; la limite par défaut de 2048 pages donne une piscine de pages monocanale d'au maximum environ 24 Mo ; les pages sont allouées à la demande et l'occupation réelle croît avec le nombre de pages utilisées. Ce sont des chiffres géométriques ; aucune mesure de mémoire de processus n'a été effectuée.

La piscine de pages est construite séparément selon le nombre réel V de canaux utilisés, avec une capacité maximale `cachePages` ; les tampons K/V des pages sont alloués à la première occupation et réutilisés en place après récupération. L'identité d'une page est déterminée par son contenu, par référence au prefix caching de vLLM :

$$
\text{key}_i=\text{FNV-1a}\left(\text{seed},\ \text{key}_{i-1},\ \text{bits d'observation},\ \text{motif de bits des valeurs},\ \text{bits de masque},\ \text{position},\ \text{empreinte de configuration}\right)
$$

Les valeurs sont hachées par motif de bits IEEE 754 ; la charge d'un NaN et le zéro signé influencent la clé de page. Une `inputFingerprint` ne couvrant que l'entrée de cette page est calculée avec une graine différente ; la recherche compare simultanément clé de page, clé de page parent et empreinte d'entrée, réduisant le risque qu'une collision de hachage prenne deux contenus différents pour la même page. L'empreinte de configuration ne couvre que les champs qui déterminent le K/V ; les paramètres de post-traitement des quantiles n'y participent pas ; voir [modelFingerprint](<../../timestral/utils/cache.cpp>).

La conséquence directe de l'adressage par contenu est que le cache n'est pas isolé par origine : deux origines dont le contenu de préfixe est identique touchent le même ensemble de pages physiques, et le compteur de références s'incrémente. Chaque `streamId` ne détient qu'une vue de numéros de pages. Quand la piscine est pleine, on n'évictue que la page la moins récemment utilisée de compteur de références nul ; s'il n'y a pas de page évictable, la requête retombe sur le recalcul intégral et signale 0 page de statistique, sans préempter les requêtes en cours d'exécution. La vue des requêtes anonymes est rendue immédiatement après exécution ; les clés de pages sont laissées au recyclage LRU. Tout changement de la statistique de normalisation, du masque, de la position ou d'une valeur d'entrée modifie la clé de page ; les pages anciennes ne touchent plus naturellement, sans notification d'inactivation explicite.

## Préfixe gelé et décodage

Le chemin paginé découpe une prédiction en deux tronçons, correspondant au prefill/decode des modèles de langage :

- prefill : [appendPages](<../../timestral/utils/transformer.cpp#L997-L1236>) encode dans la piscine de pages les patches entiers non encore paginés après le tableau de pages ; la séquence de non-touchs consécutifs de la queue est encodée en un seul prefill et déposée dans les pages ; le préfixe touché est réutilisé directement. Le backend BLAS a activé `optimizedPrefill` : un prefill pur s'arrête après écriture du K/V du dernier sous-couche time dans les pages, sans calculer les MLP suivants ni la tête de quantiles.
- décodage : `decodePages` reprend la normalisation de la queue non paginée depuis les cumulés de la dernière page du tableau de pages, lit le K/V des pages de préfixe comme clés de l'attention time, et ne calcule l'état caché que pour la queue et les emplacements de requête ; le cadre de sortie prend les cumulés de la fin du dernier patch.

Le critère de correctitude est la cohérence avec un recalcul intégral de même masque et de même sémantique de normalisation, à un seuil max(1e-6, 1e-5×RMS de sortie). Les notes de progression contiennent un jeu de mesures historiques (build par défaut, 5M réel, intervalle 0.01 s) : cas d'alignement et de manquant max_abs=0 ; ajout progressif de 1000 points max_abs=0, taux de toucher montant de 0.5 à environ 0.90 ; seconde requête de même préfixe 0 page de calcul, durée totale 25.52 ms contre 67.57 ms pour la première. Le streamTest.cpp ayant produit ces chiffres n'est plus dans le répertoire tests actuel ; les chiffres n'ont pas été re-mesurés sur le code actuel.

Plusieurs requêtes d'une même bouée avec un tableau de pages strictement identique passent par `decodePagesShared` : le groupe partage une seule table de pages de V lignes, chaque requête lit le K/V de préfixe du canal correspondant par `row % V`, la queue est encodée indépendamment par requête, et le chemin partagé est en lecture seule sans ajout de pages. La couche modèle vérifie que les observations et valeurs de préfixe des requêtes sont point par point identiques, sinon refuse.

## Multivarié et covariables

Les canaux d'une même requête forment un groupe de modélisation conjointe ; au sein du groupe l'attention variate échange de l'information ; les requêtes différentes sont isolées par `groupIds`, et la concaténation en lot ne mélange pas les canaux d'équipements différents dans un même groupe. `maxVariables` vaut 8 par défaut, c'est une valeur de protection du service ; les notes précisent que la configuration amont n'a pas cette limite dure. La sortie reste les quantiles marginaux de chaque canal à chaque instant, pas une distribution conjointe.

L'intérieur du moteur suit les rôles de ligne amont : Target=0, PastCov=1, KnownCov=2, un rôle par ligne et non par point. Les covariables past n'ont que de l'historique ; les covariables known fournissent les valeurs du contexte plus de l'horizon, sont normalisées par statistique de toute la fenêtre, et sont visibles dans les deux sens à l'encodage de l'axe time. L'entrée publique ne génère pas de lignes past ; les lignes past/known du moteur servent surtout à reproduire les contrastes de covariables amont.

Dans l'interface publique, lorsque les fins d'observation des canaux ne sont pas alignées, on effectue une complétion segmentée automatique selon la disponibilité des observations, implémentée dans [condition.cpp](<../../timestral/utils/condition.cpp>). Par exemple, trois canaux d'historique de longueur 3, 7, 4, prédiction jusqu'à 10 : d'abord A seul comme cible prédit la position 4 ; ensuite A et C comme cibles, B comme known, prédisent de 5 à 7 ; enfin les trois conjointement prédisent de 8 à 10. Chaque segment appelle la prédiction d'horizon court native ; le contexte du segment suivant est composé de lectures réelles et des médianes déjà générées ; les observations réelles ne sont pas écrasées par les sorties du modèle. Le découpage en segments modifie le remplissage à gauche des patches et le point de départ de la normalisation ; c'est une stratégie d'inférence explicite, non équivalente au support natif de rôles par point par le modèle.

Le chemin conditionnel ne réutilise pas le KV d'historique inter-requêtes. La raison a déjà été évoquée dans [l'architecture du modèle](<architecture.md>) : les lignes known sont visibles dans les deux sens et utilisent des statistiques sur toute la fenêtre, la cible lit known en inter-canal au même slot, et les K/V d'historique des couches suivantes dépendent des valeurs futures known. Le fait que tous les canaux aient des observations n'est qu'une condition nécessaire, insuffisante pour prouver la réutilisabilité des états cachés.

## Expérience : préfixe gelé et fenêtre glissante FIFO

![Préfixe gelé et fenêtre glissante FIFO](<../../assets/infer-swa.svg>)

[L'implémentation de l'exemple de streaming](<../../example/tui/streaming.cpp>) fournit un chemin approximatif activé uniquement au niveau de l'exemple, pour répondre à « après remplissage de la fenêtre, peut-on avancer indéfiniment avec un stockage KV borné ». Les patches entiers de l'historique commun sont écrits dans des pages persistantes par le chemin causal toutes cibles ; quand le tableau de pages atteint sa capacité, on libère la plus ancienne page avant d'ajouter les nouvelles ; `positionOffset` fait conserver aux pages retenues leur position absolue d'origine, sans renumérotation des KV anciens ; les cumulés de normalisation sont continus depuis le début du flux, sans retour en arrière lors de l'éviction de pages anciennes. À la prédiction, `predictFrozen` lit ces pages gelées et ne recalcule que les canaux auxiliaires et la queue de prédiction.

Ce n'est pas équivalent à un modèle « qui recalcule les échantillons récents de la fenêtre à chaque bouée » : les pages retenues portent l'information de contexte au moment de leur création et ne sont pas recalculées après glissement de la fenêtre. Ce n'est pas un encodeur bidirectionnel retapé en décodeur : le chemin known-future natif a un encodage bidirectionnel et des statistiques sur toute la fenêtre ; geler ses dépendances d'historique est une approximation délibérée. Le service public par défaut ne passe pas par ce chemin. L'interface, les clés de configuration et les champs de sortie par bouée sont dans [l'interface de streaming](<../../example/tui/streamingTui.cpp>) et [la notice de l'exemple](<../../example/README.md>).

L'outil de balayage (outil local `example-tools/swaBench.cpp`, non versionné) fait avancer l'historique commun jusqu'à 131072 points sur six fenêtres de 128, 512, 1k, 2k, 4k, 8k, recalculant à chaque contexte d'environ 1,2 fois l'espacement une prédiction de 512 pas du premier canal, et la comparant au recalcul intégral natif de la même fenêtre. Conditions de mesure : 5M réel, deux canaux de Fourier corrélés, 100 Hz, bruit d'amplitude 0.03, horizon=512 (16 patches), 1 patch de rétroaction par position, moyenne symétrique et troncature non négative désactivées. K est le KV d'historique commun gelé avec éviction FIFO de la page la plus ancienne ; N est le recalcul natif segmenté selon le plan des observations ; le RMSE ne compte que les positions de prédiction non observées, comparé au reference sans bruit.

| Fenêtre | K RMSE @128k | N RMSE @128k | Frame intégral natif ms | Frame intégral en cache ms | Facteur |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.484568 | 0.290793 | 7.88 | 6.09 | 1.3 |
| 256 | 0.314858 | 0.433044 | 7.56 | 6.24 | 1.2 |
| 512 | 0.321666 | 0.152261 | 10.35 | 6.44 | 1.6 |
| 1k | 0.194051 | 0.077387 | 14.39 | 6.94 | 2.1 |
| 2k | 0.112031 | 0.055461 | 25.11 | 7.70 | 3.3 |
| 4k | 0.083611 | 0.067024 | 49.64 | 9.45 | 5.3 |
| 8k | 0.069494 | 0.058639 | 123.95 | 13.15 | 9.4 |

![Balayage de fenêtre SWA](<../../assets/swa-rmse.png>)

Les colonnes RMSE et la courbe proviennent d'un balayage Eigen FP32 quad thread existant, signal bicanaux, 1 patch de rétroaction par position. Les colonnes de latence proviennent d'une autre re-mesure selon un régime de production normal, hors de ce balayage : monovarié, BLAS AVX2 mono thread, horizon=512 (16 patches, 32 points prédits par patch), une frame complète tous les 512 pas de prédiction. La frame intégrale native est `predictBlock`, sans cache, le fenêtre entier est réencodé à chaque frame ; la frame intégrale en cache est `decodePages`, le préfixe de la fenêtre est déjà résident dans le KV paginé, seul le décodage de cette frame est compté, l'`appendPages` du préfixe n'est pas dans le chronométrage. À la même granularité, un aller-retour complet unique et un prefill de même longueur sont du même ordre : frame intégral natif de fenêtre 8k (contexte 8192 plus 512 prédiction) 123.95 ms, contre 104.15 ms de prefill monovarié 8k dans le même état de machine, soit environ 19 % de plus. Les deux chemins sont numériquement cohérents à la même granularité de frame (RMSE 0.0696039 contre 0.0696051 en fenêtre 8k). Médiane de 3 mesures, 1 échauffement avant chaque chronométrage, hors chargement du modèle. Matériel AMD Ryzen 7 7735H, MSVC Release ; une autre machine ou un autre nombre de threads impose une re-mesure ; l'état d'alimentation influence les valeurs absolues (à la même granularité de prefill 8k bicanaux, benchmark.md a enregistré 106.83 ms lors d'une collecte, la re-mesure actuelle donne 233.4 ms) ; entre collectes, seul le facteur relatif et l'ordre de grandeur sont comparables. Les données brutes (table RMSE, table de latence de frame intégrale, contrôle non glissant) sont conservées dans l'espace de travail local.

Ligne de contrôle non glissante (fenêtre 8192 ne poussée que jusqu'à ce point, K=N) :

| context | 128 | 256 | 512 | 1k | 2k | 4k | 8k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.9459 | 0.2154 | 0.0913 | 0.0733 | 0.0538 | 0.0494 | 0.0358 |

Observations :

- La ligne grise de contrôle est le tronçon non glissant d'une fenêtre 8192 poussée seulement jusqu'à 8192 ; K et N sont cohérents dans la marge d'erreur d'arrondi flottant ; elle montre l'effet de capacité pur : environ 0.95 à 128 points, environ 0.21 à 256 points, environ 0.094 à 320 points, remontée à environ 0.11 aux alentours de 672 points, puis nouvelle descente à environ 0.036 à 8192 points ; ce n'est pas une courbe racine carrée monotone. La ligne de contrôle inclut explicitement les départs des fenêtres 512, 1024, 2048, 4096, 8192 ; chaque courbe colorée K se raccorde à cette ligne depuis son propre départ, n'entrant dans la zone d'éviction qu'après.
- En fenêtre 512, K est à RMSE environ 0.09 à context=512, monte à 0.14–0.25 dès le début de l'éviction, puis oscille entre 0.25 et 0.33 jusqu'à 128k ; le N de la même fenêtre oscille entre 0.04 et 0.21.
- Sur ce signal, l'erreur n'explose pas linéairement avec le contexte : K entre en plateforme après glissement, et à 128k reste du même ordre que la plateforme en contextes 2k à 8k. La plateforme ne doit pas se lire comme « l'erreur n'accumule pas » ; elle dépend de la stationnarité du signal et de l'information utile portée encore par les pages anciennes ; avec 5×sin(0.03t) de dérive basse fréquence ajoutée au signal de fenêtre 512, K passe de 0.046 à 0.367, ce qui suffit à infirmer toute garantie universelle.
- Plus la fenêtre est grande, plus K se rapproche de N, mais le décodage en cache est aussi plus lent : la frame intégrale en cache monte lentement de 6 ms en fenêtre 128 à 13 ms en fenêtre 8k, la frame intégrale native de 8 ms à 124 ms, environ 9,4 fois en fenêtre 8k ; 6.4 ms contre 10.4 ms en fenêtre 512. Ce qui est économisé est surtout le réencodage du préfixe répété ; le coût de lecture de l'attention à l'intérieur de la fenêtre reste à payer ; l'écart entre les deux est faible sur fenêtre petite.
- Le context_length du modèle est 32768 points d'échantillonnage ; cela ne veut pas dire que le flux s'arrête à 32k ; ici 128k est la longueur du flux, chaque aller-retour ne voyant que la window. Un flux long possible n'implique pas un champ de réception infini pour le modèle, ni que l'information des pages anciennes soit encore exacte.

Conclusion : SWA convient à un compromis qualité/latence explicitement choisi, et ne remplace pas par défaut le recalcul exact. Utiliser le chemin natif quand la précision par défaut est requise ; pour un budget contraint, on peut employer BLAS mono thread ou limiter les threads Eigen par `threads`, mais le backend et le nombre de threads ne changent pas l'erreur d'approximation ci-dessus. Conserver le KV du premier token comme attention sink est implémenté selon l'interface expérimentale et le balayage est terminé (voir les notes locales de fenêtre glissante) : sur un petit signal stationnaire, il restaure partiellement les ancrages de phase à longue distance, sans coût mesurable additionnel au décodage, mais ne peut pas empêcher la dégradation du cache gelé sous dérive statistique ; non intégré au chemin par défaut, l'interface reste expérimentale.

## Expérience : RoPE temps continu

Désactivé par défaut, à définir avant la construction de l'instance :

~~~cpp
settings.continuousRope = true;
settings.ropeReferenceInterval = 0.01; // intervalle d'échantillonnage de référence, en secondes
~~~

Si `ropeReferenceInterval` vaut 0, on adopte l'intervalle d'échantillonnage résolu de la requête en cours ; une valeur positive fixe permet de conserver la même échelle de temps physique entre différents rythmes d'échantillonnage ; quand on change l'unité de temps, on doit changer en synchronie cette valeur de référence.

Chaque patch utilise la moyenne de ses temps d'échantillonnage comme centre temporel ; la coordonnée est `(center_p−center_0)/(patchSize×referenceInterval)` ; le modèle prend ensuite le premier patch observé du même groupe comme ancre ; on ne peut pas prendre directement les secondes Unix comme position de rotation. Le remplissage à gauche et les temps d'échantillonnage futurs sont extrapolés selon l'intervalle de la requête ; la coordonnée continue est passée à l'encodage, au décodage, à la rétroaction autorégressive et à la vérification des pages partagées ; l'empreinte KV contient le motif de positions, l'intervalle de référence et les positions réelles, si bien qu'un changement de temps ne réutilise pas à tort les K/V d'anciennes positions.

Sur grille régulière, si l'intervalle de référence égale l'intervalle d'échantillonnage, on retombe exactement sur les positions discrètes ; les tests associés exigent une identité bit à bit des prédictions ; un temps non uniforme ne prouve que cela entre dans le calcul, sans promettre d'amélioration de précision. Sur un exemple de signal généré selon le temps réel du 5M réel, la MAE de l'entrée discrète brute est 0.139345, RoPE continu 0.158215, après ré-échantillonnage linéaire explicite 0.0924159 ; le RoPE continu est pire sur cet exemple ; l'expérience n'a pas utilisé d'observations après le point de départ de la prédiction pour le ré-échantillonnage ; la méthode de reproduction et le RMSE sont dans les notes d'acceptation locales.

Limites de l'expérience :

- Seule l'encodage de position entre patches change ; un patch n'a toujours qu'une position ; une projection d'entrée fixe ne peut pas représenter tous les intervalles d'échantillonnage internes différents
- La sortie utilise toujours une grille régulière construite par sampleInterval ; aucune interrogation en point de temps arbitraire, et aucun ré-échantillonnage automatique
- Multicanal : les mêmes positions d'échantillonnage sont encore partagées lors du même aller-retour standard ; le chemin d'observations intercalées aligne d'abord sur une grille commune, qui est uniforme ; sur le chemin de complétion conditionnelle, le mode continu est bit à bit identique au discret
- Le mode continu n'active la pagination que pour un historique de patch entiers ; un historique non entier passe par l'allers-retours complet ; le comportement de cache discret par défaut est inchangé
- Quand la précision des horodatages est insuffisante pour exprimer l'intervalle de référence, on refuse explicitement ; on peut d'abord translater l'origine du temps ; on n'accepte pas silencieusement des écarts de temps indistinguables
- Les poids existants sont entraînés par patch discret ; c'est un chemin d'inférence expérimental, sans prétention que le modèle supporte nativement des séries temporelles irrégulières en général

Les expériences associées à la suppression de colonnes de temps vides, à la normalisation du canal de temps et au décodage de frame intégrale n'ont pas été conservées comme options utilisables ; aucune amélioration de qualité utilisable n'y a été observée.

## Arbitrages et leçons

La frontière de correctitude du cache est plus difficile à déterminer que le cache lui-même. La clé de page hache les valeurs d'entrée, le masque, la position et la configuration ; elle garantit seulement « même entrée donne même page » ; « les pages d'historique ne sont pas affectées par les entrées suivantes » doit être garanti par la structure de dépendance du modèle, un hachage ne peut pas le prouver. Le chemin standard satisfait cette condition, le chemin conditionnel non ; donc le premier est paginé, le second recalculé intégralement ; la ligne de partage est déterminée par le masque et la normalisation, pas par la commodité d'ingénierie.

Le deuxième arbitrage est : plutôt tomber sur l'intégral que préempter. Piscine de pages pleine, grille non conforme ou échec du décodage partagé, les requêtes retombent toutes sur le recalcul intégral individuel ; le résultat reste correct, juste plus lent. Le planificateur peut donc rester simple : un seul thread de travail, FCFS, exécution par bouées, sans état intermédiaire par token à migrer.

Enfin, l'expérience de fenêtre glissante montre que « le coût d'incohérence entraînement/inférence est faible » doit être mesuré signal par signal ; être proche de la référence sur un signal périodique stationnaire n'implique pas que cela tienne sur un signal à dérive. Tout chemin approximatif reste une expérience explicite et rapporte simultanément qualité et durée.
