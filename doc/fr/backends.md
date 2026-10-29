# Backends de calcul

L'implémentation CPU comprend le reference par défaut, ainsi que les backends optionnels Eigen, ytensor et BLAS maison. Ce document présente les chemins numériques, les capacités de threads et de cache ; les détails BLAS sont dans [Backend BLAS](<blas.md>), le modèle GPU dans [Backend Vulkan](<vulkan.md>).

## Pourquoi plusieurs backends

Le portage le plus direct consiste à écrire chaque opérateur de l'amont PyTorch en boucles triples. C'est exactement le backend reference : il ne dépend que de la bibliothèque standard C++17, l'ordre d'addition est simple et traçable, ce qui convient comme base de correctitude. Les durées actuelles au niveau modèle sont dans [Comparaison de référence](<../../README_fr.md#performances>).

L'approche naïve pour accélérer est d'introduire une bibliothèque matricielle et de lui confier tous les calculs. On rencontre alors deux contraintes : le chemin de déploiement par défaut doit rester sans dépendance tierce ; changer de bibliothèque modifie l'ordre d'addition flottant, les résultats ne sont plus identiques bit à bit au reference, et il faut un jeu de tolérances fixé d'avance, qui ne pourra pas être desserré a posteriori. Le traitement de ce dépôt est le suivant : le backend est choisi à la configuration par CMake (mécanisme dans [Architecture du modèle](<architecture.md>)), les backends optionnels ne remplacent que les parties à forte densité de calcul, la sémantique du modèle reste cohérente, reference et BLAS partagent l'exécuteur, Eigen et ytensor conservent chacun leur orchestration ; la précision est jugée par un oracle amont fixé et des tolérances sensibles à l'échelle.

![Hiérarchie des backends et comparaison des capacités](<../../assets/backend-layers.svg>)

## reference

reference est une sous-classe de `Transformer` qui n'implémente que les crochets numériques étroits exigés par l'orchestration du modèle : projections, RMSNorm, permutation QKV et RoPE, attention simple-tête ; voir [reference/model.cpp](<../../timestral/reference/model.cpp>). L'insertion, les masques, la pagination, la tête de quantiles et toute la logique sont dans [Transformer](<../../timestral/utils/transformer.cpp>) de `utils`, si bien que reference lui-même ne fait pas cent lignes. Les opérateurs de base sont dans [kernels.cpp](<../../timestral/reference/kernels.cpp>) : les couches linéaires font un produit scalaire sériel par sortie, puis ajoutent un biais ; l'attention calcule d'abord les scores de la ligne entière, remplit les positions masquées par moins l'infini, et multiplie par V après softmax.

reference comporte trois choix délibérés :

- Aucune fusion agressive modifiant l'ordre des opérations ; chaque étape peut être mise en correspondance avec une expression PyTorch amont, et en cas de problème on peut comparer couche par couche.
- Thread unique fixe ; `Settings::threads` n'a aucun effet sur lui.
- L'option de compilation est `/fp:precise` et fast-math n'est pas activé, ce qui préserve l'ordre du produit scalaire sériel.

Les régressions numériques des autres backends prennent toutes pour référence ; avant et après l'extraction de `Transformer`, la sortie du referenceTest sur 5M réel est identique ligne par ligne (176 lignes, diff_count=0).

## Eigen

Le backend Eigen utilise les vecteurs lignes et les vues Map d'Eigen 3.4.0 pour les projections ; l'attention emploie le noyau numérique commun ordonné.

- Les poids sont pré-transposés à la construction ; le vecteur ligne Eigen met à jour terme à terme dans l'ordre K, avec le biais ajouté à la fin. OpenMP parallélise par ligne d'entrée et ne réordonne pas la réduction à l'intérieur du produit scalaire.
- L'attention lit directement le préfixe partagé et la queue de la tour courante ; variate resserre la portée par groupe de requête. Le softmax ordonné et PV maintiennent le budget numérique figé.

RMSNorm, RoPE, softmax, SiLU, SwiGLU et la tête de quantiles appellent directement les noyaux de reference, voir [eigen/model.cpp](<../../timestral/eigen/model.cpp>). Le backend Eigen précède l'extraction de `Transformer` et conserve encore une orchestration de modèle et une implémentation de pagination complètes ; une modification de sémantique commune doit être vérifiée en synchronie.

Le nombre de threads est contrôlé par `Settings::threads` : 1 pour sériel (défaut), 0 pour la valeur automatique plafonnée à 4, N pour un nombre exact de threads. Le nombre de threads d'Eigen est un état global de niveau processus, à définir avant la construction du modèle ; plusieurs modèles d'un même processus partagent la dernière définition. Le fait que la valeur automatique ne prenne pas tous les cœurs a été mesuré : les GEMM de 5M sont petits, la suite de régression des notes fait environ 28 s à 16 threads et environ 2.9 s à 4 threads.

## ytensor

Le backend ytensor utilise le single-header 0.14 d'une source désignée par l'utilisateur ; le dépôt ne contient pas sa licence ; l'enregistrement factuel est dans [la notice 3rd](<../../3rd/README.md>). Les projections passent par le matmul strided de ytensor, en empaquetant le stockage de ce dépôt dans des vues non propriétaires. L'attention actuelle utilise le noyau commun ordonné, lit directement le préfixe partagé et la queue indépendante, et les opérateurs élément par élément suivent aussi les noyaux de reference.

- YBLAS s'occupe des projections. Les implémentations d'attention d'origine FLASH_AVX2 et MATH ne sont plus le chemin modèle par défaut pour les longues requêtes actuelles ; on ne doit pas prendre l'ancien enregistrement de configuration FA2 pour la recette de cette fois.
- Seul [ytensor/model.cpp](<../../timestral/ytensor/model.cpp>) entre dans un but OBJECT C++20 indépendant avec AVX2/FMA activés (MSVC en `/arch:AVX2`, GCC/Clang en `-mavx2 -mfma`) ; la fabrique et le cœur restent du C++17 ordinaire, avec interdiction pour IPO/unity de traverser la frontière d'instruction set. La fabrique de référence vérifie le support XMM/YMM du CPU et de l'OS avant d'appeler n'importe quelle fonction de cet objet ; l'ancien CPU est explicitement refusé, et la configuration non x86 est refusée à la configuration. L'implémentation, la validation et les échecs numériques résiduels sont dans [l'enregistrement de compatibilité](<linuxCompatibility.md>). MSVC sous `/arch:AVX2` ne définit pas `__FMA__` et ne supporte pas la syntaxe GNU `__attribute__` du single-header ; ce fichier complète par paire ces deux endroits avant d'inclure ytensor et les annule après ; les notes précisent que ces attributs ne sont que des suggestions d'inlinéation.
- La pagination incrémentale est prise en charge ; `supportsIncremental()` renvoie true. La page conserve le K/V de chaque couche time et l'état de normalisation causale ; l'ajout et le décodage partagé acceptent le contrat de tableau de pages unifié.

## Critère d'acceptation de la précision

Tous les backends sont comparés au même jeu d'oracle amont fixé : 14 cas de blocs par couche, 36 cas de prédiction (dont 5 cas de refus attendus), générés par le Python amont sur la base figée. Le mini-modèle est des paramètres aléatoires à graine fixée, pas des poids publiés. Les tolérances sont fixées à la première période et reprises par les backends suivants, sans desserrement :

| Objet | Critère | Note |
| --- | --- | --- |
| Opérateur | atol 1e-5, rtol 1e-4 | Comparaison de noyau unitaire |
| État caché | max(1e-5, 1e-4×RMS(référence)), et L2 relatif ne dépassant pas 1e-5 | embedding, sorties de chaque couche, out_norm |
| Prédiction finale | abs(a−b) ≤ 1e-4 + 1e-3×abs(référence) | Jugement par cas, avec enregistrement de l'erreur maximale et du RMSE |
| cached decode | max(1e-6, 1e-5×RMS de sortie) | Décodage paginé contre recalcul intégral de même sémantique |

L'état caché utilise un atol sensible à l'échelle, car l'amplitude du flux résiduel croît couche par couche. La sortie RMS de la couche 7 du 5M réel est d'environ 120 à 152 ; un seuil fixe et petit prendrait pour erreurs les différences normales dues à l'ordre de réduction FP32 ; l'enregistrement indique une erreur absolue maximale intracouche de 8.06e-3 sur le chemin des poids réels, L2 relatif 5.10e-6. Le seuil de cached decode est fixé séparément, pour vérifier le cache, sans mélange avec les tolérances d'oracle ; les tests de concurrence configurent la base et le service testé sur le même chemin sans cache, ne vérifiant que la cohérence de planification.

L'erreur de prédiction enregistrée lors de la phase initiale de portage est la suivante ; elle ne remplace pas la régression de l'implémentation ordonnée actuelle :

| Backend | realDefault erreur absolue max | realFullBlockThenPartial erreur absolue max |
| --- | --- | --- |
| reference | 9.54e-6 | 0.310 |
| Eigen | 1.34e-5 | 0.083 |
| ytensor (Naive forcé) | 9.53674e-6 | 0.310059, bit à bit identique au reference |

realFullBlockThenPartial est un cas d'horizon long de 1030 pas, avec RMS de sortie de référence d'environ 854 ; l'autorégression multi-blocs amplifie les différences d'ordre de réduction, si bien que son erreur absolue est nettement plus grande que les cas courts ; les trois backends sont dans les tolérances. La boucle Naive de ytensor a le même ordre d'addition que reference ; les résultats sont bit à bit identiques. Avec YBLAS activé, l'ordre d'addition change ; les notes indiquent un max_abs d'enregistrement inférieur ou égal à 3.8e-6 sur cas synthétique et de l'ordre de 1e-7 sur cas multivarié et conditionnel du 5M réel ; on passe à un jugement par tolérance sensible à l'échelle, sans exiger plus la bit à bit ; les chiffres unitaires de realDefault et semblables du chemin YBLAS n'ont pas été enregistrés séparément.

## Arbitrage de performance

Mesures côte à côte sur la même machine (AMD 7735H, MSVC 19.36, Release, mono thread, protocole stageBench et timestralBench, date des notes lors de l'acceptation de la troisième période) :

| Item | reference | Eigen | ytensor |
| --- | ---: | ---: | ---: |
| forward T=256 | 32.10 ms | 6.83 ms | 6.77 ms |
| forward T=4096 | 362.21 ms | 70.64 ms | 55.35 ms |
| forward T=32768 | 4424.15 ms | 961.20 ms | 1185.57 ms |
| append | 20.98 ms | 8.46 ms | non pris en charge |
| decode | 11.79 ms | 3.59 ms | non pris en charge |
| batch | 15.7 req/s | 88.2 req/s | 123.1 req/s |
| stream | 102.7 ms | 21.4 ms | 28.0 ms |

Quelques lectures :

- Le forward simple-ligne est d'un facteur environ 3.7 à 6.5 de Eigen et ytensor par rapport à reference sur les trois longueurs ; Eigen est plus rapide à T=32768, ytensor est plus rapide à T=4096.
- Le protocole stream est l'ajout progressif de données de la même origine ; Eigen a un cache paginé, ytensor recalcule intégralement à chaque fois ; donc bien que le forward unique de ytensor soit plus rapide à T=4096, son stream reste plus lent qu'Eigen. Si le cache paginé est utilisable a plus d'impact sur le scénario en flux que la vitesse d'un GEMM unitaire.
- Les gains multithread sont limités. Eigen à 4 threads sur stream est le point optimal global mesuré, environ 7.4 fois reference ; 8 et 16 threads se dégradent au contraire à cause du coût de planification OpenMP des petits GEMM. Le YBLAS de ytensor n'a qu'environ 1.4 fois de gain à 4 threads sur contexte 32768 ; court contexte plus threads est une perte nette. D'où le défaut `threads=1`.
- Chaque groupe de configuration rapporte une tour de statistiques de protocole (5 échauffements, 25 mesures) ; pas de verrouillage de fréquence ; des variations nettes existent entre les tours ; les chiffres de la table servent à comparer les ordres de grandeur et ne sont pas une promesse de latence.

## Vue d'ensemble du backend BLAS

Le BLAS maison ne lie ni le BLAS système ni une bibliothèque matricielle externe ; il réutilise l'orchestration Transformer et couvre les crochets numériques des projections, de l'attention, de la porte et de SwiGLU. Les poids sont pré-empaquetés au chargement ; le modèle actuel maintient des opérations FP32 ordonnées par K ; au moment de l'exécution les micro-noyaux sont choisis selon les capacités AVX2, SSE4.2 et scalaire. Settings.threads contrôle le pool de threads fixe propre à chaque modèle, supporte 1, 2, 4 threads et semblables, 0 choisit automatiquement au maximum 4, et n'ignore plus le paramètre. La parallélisation se fait par blocs de sortie complets, sans scinder la réduction du produit scalaire, et supporte la pagination, le prefill KV-only et le préfixe partagé. La performance actuelle est dans [les données de référence](<benchmark.md>), la conception dans [Backend BLAS](<blas.md>).

## Arbitrages et leçons

La différence la plus sous-estimée entre backends n'est pas la vitesse, mais l'ordre d'addition. Dès qu'on change d'implémentation de GEMM, les résultats ne sont plus bit à bit égaux au reference ; il faut alors déjà avoir un jeu de tolérances défini d'avance, donné selon l'échelle numérique, sinon on est facilement tenté de desserrer les seuils en découvrant les écarts. Ce dépôt a fixé les tolérances à la première période, les backends suivants les reprennent tous, et en cas de dépassement on retombe sur l'optimisation plutôt que de changer le seuil ; l'exemple du retour en arrière du noyau BLAS multi-chaînes à entrelacement étroit tient à cela.

La deuxième leçon est de laisser au backend ce qui est son travail. Eigen et ytensor ont chacun copié une orchestration de modèle ; BLAS n'implémente que les crochets ; le premier se connecte vite, mais chaque modification de masque ou de pagination exige trois vérifications ; le second concentre la sémantique dans `utils`, et le nouveau backend a moins de travail et de surface d'erreur. Si un nouveau backend doit être connecté plus tard, la manière de BLAS doit être suivie.
