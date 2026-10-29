# Arbre d'optimisation en profondeur du backend Vulkan

## Chemin de précision du modèle actuel

Les formes courantes du modèle utilisent l'attention ordonnée, la réduction RMS et les non-linéarités FP32 compensées ; le préfixe partagé n'est plus copié par requête. L'accélération de projection G1 est conservée ; les noyaux de réduction en ligne A2 et semblables restent utilisés dans les benchmarks indépendants et les retours hors plage. Le déploiement et les limites numériques des 52 modules actuels sont dans [la conception portable](<vulkan-portability.md>), et le protocole des figures de performance et de la table du long decode est dans [la notice de référence](<benchmark.md>). Les résultats historiques de l'arbre d'optimisation ci-dessous ne représentent pas le dispatch par défaut actuel.

Les enregistrements, journaux et rejets cités dans ce document sont dans l'espace de travail local, non publiés avec le dépôt.

## Direction la plus récente : portabilité Intel et budget GPU (2026-10-25)

L'objectif actuel est mis à jour : viser les iGPU Intel de petite taille des générations 13 à 15, sans réglage spécial pour le 680M ; le decode n'est plus validé à 1 % MFU, mais regarde la latence de même tâche, le copy et la bande passante. Les chiffres 300/30 d'avant ne sont conservés que comme enregistrement de développement historique, et non comme seuil de déploiement Intel.

Le chemin critique local ggml fourni par l'utilisateur a été lu ; le commit distant est inconnu ; la portée et le SHA256 d'une partie de la revue sont dans [la conception portable](<vulkan-portability.md>). Les documents AI Max395/RDNA ont été recherchés, mais le corps externe est toujours bloqué par la vérification non-IP publique ; on ne prétend pas à une revue de code complète. Pas de subagent, pas de noyau spécial RDNA copié.

Timestral::Settings::gpuThreads a été ajouté : 0 n'importe pas ; un non-0 doit être un multiple de 256 dans la plage uint32, et contraint les invocations logiques de chaque fenêtre de dispatch sérielle, ce n'est pas une dotation physique EU. Le local_size fixe, la légalité et la taille des variantes sont vérifiés sur l'hôte ; un budget inférieur à 512 recule le pipeline en option de 512 threads ; quand le périphérique signale des subgroups 8/16, les noyaux rapides 32/64 seulement ne sont pas créés. Le paquet complet actuel compte 52 modules ; les 21 sources/liaisons/push d'origine ne changent pas.

L'ancien M≤2048 de la fusion résiduelle était une garde expérimentale du 680M, retiré du fondement par défaut ; maintenant désactivé par défaut, l'activation explicite de l'expérience ne conserve pas cette borne de M élevé. On ne proclame pas que la faible SLM attention, le nouveau panneau GEMM ou le miroir KV UMA sont déjà implémentés. L'interface complète, la frontière de sécurité, la validation et le matériel Intel non couvert sont dans [la nouvelle conception](<vulkan-portability.md>).

## Historique : 300/30 GFLOP/s et résultats multivariés (début de 2026-10-25)

Cette section est l'état d'adoption et la mesure réelle avant l'ajustement de l'objectif de cette tour ; la stratégie par défaut actuelle suit la conception portable du haut. On n'a pas atteint les 300 GF/s de prefill 680M et les 30 GF/s de decode H1 proposés par l'utilisateur, et la sixième période n'est pas marquée terminée. Pas de subagent, sans changement de l'entrée publique ni de la disposition poids/Tensor, sans rétablissement de l'arena/push par décalage.

### Mesure réelle de la stratégie par défaut finale

`Matrice complète finale` correspond à la source actuelle, plus un `résumé structuré`. 32768 échantillons par canal, données différentes au sein du même groupe, non un batch indépendant ; tous les targets prédisent ensemble. Préfill froid médiane de 3, decode H1 échauffé 5 fois/chronométré 25 fois en médiane, toutes les tâches sérielle et non parallèle au build. On enregistre la médiane de cette tour, pas le minimum de toutes les tours ; la fréquence/puissance en temps réel n'est pas mesurée, et on ne peut pas qualifier la variation inter-tours de problème de pilote.

| Nombre de canaux du même groupe | prefill 680M ms | 680M H1 ms | prefill GF/s | MFU nominal 680M | prefill 4060 ms | 4060 H1 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.47 | 8.48 | 256.8 | 8.56 % | 21.68 | 5.11 |
| 2 | 76.31 | 17.67 | 279.1 | 9.30 % | 42.20 | 11.60 |
| 4 | 165.99 | 28.23 | 256.7 | 8.56 % | 98.37 | 22.07 |
| 8 | 328.71 | 55.03 | 259.3 | 8.64 % | 189.49 | 48.93 |

Le decode 680M fait seulement environ 1.84 à 2.37 GF/s, soit 0.061 % à 0.079 % de MFU nominal, bien en dessous de 1 %. Le prefill monovarié est 256.8 GF/s, 8.56 %, exigeant encore une amélioration de débit d'environ 16.8 %, ou une baisse de 41.47 ms à 35.50 ms. Le prefill 2/4/8 canaux n'atteint pas non plus 300 GF/s. Le 4060 n'utilise pas le dénominateur 3 TFLOPS du 680M pour déduire son propre MFU.

Le `CPU fenêtre de référence` plus tôt de la même tour sur la même machine : BLAS encode successivement 170.85/336.81/683.07/1424.89 ms, decode 4.21/7.85/14.90/36.04 ms ; ce n'est pas une re-mesure CPU de dernière appariement GPU par item. Le prefill GPU est nettement plus rapide que cette référence, le decode reste plus lent. L'entrée totale de 8 canaux est 262144 échantillons ; on ne peut pas expliquer la croissance de durée par un retour de même charge de travail, et aucune augmentation en proportion du nombre de canaux du débit total n'est apparue.

### Covariables futures connues

Un Target plus les autres KnownCov, fournissant de vraies observations futures, nombre total de canaux du même groupe 2/4/8. L'entrée paginée d'ajout actuelle ne contient pas de rôles/futur ; ce mode est un forward conditionnel complet, échauffé 5 fois, chronométré 5 fois en médiane, non marquer comme decode en cache. Ci-dessous prend le chronométrage en régime établi de metric=full_conditioned_forward dans le journal ; le champ decode_ms de sortie de compatibilité de ce mode n'est qu'un nom de champ ancien, sans signifier de décodage incrémental.

| Condition | Nombre total de canaux | forward complet 680M ms |
| --- | ---: | ---: |
| 1 Target + 1 KnownCov | 2 | 294.44 |
| 1 Target + 3 KnownCov | 4 | 644.15 |
| 1 Target + 7 KnownCov | 8 | 1344.05 |

Ces forward renvoient des quantiles normalisés ; on ne peut pas comparer directement la précision avec la première sortie 0.50 et la dé-normalisation paginée 2.38. Les FLOPs nécessaires du double flux complet ne sont pas encore dérivés séparément ; on ne rapporte donc pas le MFU de ce scénario, et on ne prétend pas que les covariables améliorent forcément la qualité de prédiction.

### Feuilles adoptées, retours et sans gain

- G1 complète les projections d'embedding réelles (K,N)=(64,768)/(768,192)/(64,192), toujours précise par K ; 12 formes × biais bit à bit identiques à v1
- A3 utilise un noyau causal indépendant pour recouper les blocs de clés futures d'encodage prouvés entièrement vides, sans changer l'espacement réel de K/V ; le bidirectionnel KnownCov et le suffixe double flux conservent le prédicat d'origine
- A5 subgroup s'active quand la capacité compute Basic/Vote/Arithmetic et 39300 B partagés est satisfaite. La validation du shader vérifie que chaque subgroup ne contient que la même requête ; seulement size32/64 passe par le chemin rapide, sinon le groupe entier retombe ; réduction directe 64 voies, fusion partagée 32 voies, sans warp32 codé en dur. Le forward conditionnel complet passe aussi le contrôle du modèle de long contexte
- P1 le prefill KV-only s'arrête après que le K/V de la dernière time est prêt ; l'attention/porte/O de queue inutile, le variate/MLP suivant, out_norm et la relecture de hidden ne s'exécutent pas ; le forward complet, trace et decode ne sont pas tronqués. La projection QKV de la fin calcule encore trop de Q ; cette Q inutile n'est pas comptée dans la production de la tâche
- G2 actuellement est un GEMV sériel par K à 64 threads, 16 colonnes/groupe, chargement des poids fusionné, utilisé pour la grille native complète M1 à 8. Le candidat K parallèle 32 voies échoue sur le budget opérateur M1/N672/K768 et cache partagé/aller-retours complet, est retiré, sans budget desserré
- A4 à count1, time/headDim64, nombre de clés au moins 512, split par 128 clés ; chaque tranche sort max/dénominateur/PV non normalisé, puis fusion stable ; le split vide ne produit pas de NaN. Passe le contrôle du modèle de tranche de queue, manquant, rôles, multi-lignes et long préfixe H1, mais sans en proclamer que le decode entier atteint BLAS ou 30 GF/s
- F1 le bloc résiduel résident sur le périphérique est utilisé par défaut seulement pour 128≤M≤2048. La projection hidden, SiLU, output et la projection skip et addition sont dans le même lot, seule la valeur finale est relue ; la petite head n'a pas de gain net, la direction de M élevé est instable, donc retour aux trois projections d'origine. L'interrupteur TIMESTRAL_VULKAN_RESIDENT_RESIDUAL=0 ne désactive que cette feuille, à 1 il suit aussi la garde de forme ; une valeur illégale est refusée
- Rotary ne réutilise que les fréquences inverses de cet appel et les lignes de position strictement identiques, sans cache de forme inter-requêtes ajoutée ; la coordonnée continue est toujours calculée avec le std::cos/sin d'origine
- [gemmWide](<../../timestral/vulkan/shaders/gemmWide.comp>) ne garde que le diagnostic explicite : N576/1024 seulement environ 5 % d'amélioration de micro-benchmark, M128 et K512 se dégradent, sans sélection automatique. Le candidat 128 threads de [attentionPacked](<../../timestral/vulkan/shaders/attentionPacked.comp>) est correct mais plus lent que subgroup, sans sélection automatique. Le [PV en registres](<../../timestral/vulkan/shaders/attentionRegister.comp>) reste diagnostic seulement

La médiane time1024 du `contrôle d'attention du même processus` : v1 12.37 ms, A2 4.00 ms, causal 2.47 ms, subgroup 2.04 ms, packed 3.72 ms, enregistrement/soumission/fence inclus, transport non inclus, non considéré comme périphérique seulement. Le A/B indépendant monovarié de F1 passe de 46.81/46.90 ms à 40.55/41.05 ms (`R1 et segment de diagnostic explicite valide`), mais les résultats de deux paires de quatre canaux ont des directions opposées ; les données complètes sont conservées dans `première paire` et `revue unique du sens inversé`, sans prendre 169 ms comme gain stable.

La concaténation de labels PowerShell antérieure a fait des paramètres non R1 un seul label ; le journal réel a channels=1 ; ces lignes sont invalidées ; le journal d'origine ci-dessus ne prend que le segment R1 et de diagnostic explicite. Le multicanal adopte l'enregistrement vérifié après validation du nombre de paramètres, et la table finale a encore vérifié les channels/cas réels, sans prendre les lignes mal marquées comme résultats 2/4/8 canaux.

### Précision et build par défaut

`Acceptation de la stratégie finale` : AMD et 4060 chacun Tiny/Real 2/2, les anciennes lignes 1/2/3 par couche et les seuils de quantiles complets inchangés, plus les covariables futures connues de 63/513 patches cache/complet/reference H1, le hachage de page en lecture seule, 129 patches. 63 et 513 traversent respectivement la borne inférieure et supérieure de la fusion résiduelle. siluGrid ajouté couvrant la frontière de 65535 groupes, la réflexion ABI de 21 modules et la validation double GPU sont dans `acceptation préalable de la fusion`. La fin ne nettoie que l'allocation de sortie temporaire inutile avant le retour, puis le build et le fumigène Tiny passent, et la matrice finale est collectée.

Le build de bibliothèque standard pure avec OpenMP, Eigen, ytensor, BLAS, Vulkan explicitement désactivés passe réellement 24/24 (82.56 s), pas une copie du 25/25 historique. reference/BLAS conservent l'implémentation par défaut d'origine de residual, seulement les crochets numériques virtuels internes sont étendus.

Le budget de sortie existant du 8 canaux forward/predictBlock public passe, mais le nouveau diagnostic strict par couche ne passe pas. Après désactivation de tous les nouveaux noyaux, il échoue toujours au même layer6/index7459 ; le candidat de RMS par item ne l'a pas résolu non plus, est retiré. On garde l'interrupteur --strict-multi du [test](<../../tests/gpuModelTest.cpp>) pour reproduire, sans écrire le test de sortie 8 lignes par défaut comme tous les niveaux 8 lignes atteints. Cette frontière de précision reste à localiser, sans seuil desserré.

### Ligne principale restante

1. E3 évalue d'abord si les 16 relectures de K/V peuvent être lues d'un coup après une même soumission ; actuellement toujours sérielle, sans dire que 17 soumissions de prefill font 1. Il faut conserver la survie des ressources de commande à l'exception et la sémantique empoisonnée, sans d'abord faire l'arena
2. K1, après définition de l'identité du pool, de la génération de page, de l'éviction/réutilisation/invalidation de modification, implémente seulement le miroir de KV du périphérique et la lecture de préfixe segmentée, réduisant à chaque H1 l'assemblage/ré-upload CPU d'environ 12 Mio et les 16 mergeKv ; pas encore implémenté
3. Le diagnostic decode a environ 2.5 à 3.4 ms hors API CPU, l'étendue de soumission GPU environ 4.6 à 5.4 ms, les deux dépassent séparément l'objectif de 0.543 ms ; le temps se chevauche et ne s'additionne pas ; on ne peut pas seulement prendre deux soumissions de moins comme solution 1 % MFU
4. L'écart strict des valeurs intermédiaires de 8 canaux, le budget de fréquence/puissance/videomémoire en temps réel restent à traiter. La récupération du corps ggml est de nouveau refusée par la vérification non-IP publique, la revue du commit figé n'est pas terminée, la source n'est pas copiée

## Plan avant l'essai de résident sur le périphérique du bloc de projection résiduel (suite de 2026-10-24)

Le prefill monovarié 680M après complément de mesure fait 45.7700 à 51.8650 ms, seulement 205.32 à 232.66 GF/s, sans atteindre 300 ; le decode H1 9.2596 à 9.3902 ms, seulement 1.73 à 1.76 GF/s. Les données sont écrites dans `matrice finale`, sans suivre la sortie d'arrière-plan perdue après redémarrage de l'hôte.

La prochaine étape n'essaie qu'une frontière avec attribution de support existante : le résiduel de patch/head fait actuellement trois project indépendants, chacun avec upload/download, SiLU et addition sur le CPU, environ 12 ms hors API CPU du diagnostic. Transformer::residual interne devient un crochet numérique protégé avec implémentation par défaut ; l'ordre par défaut de reference/BLAS et semblables est inchangé ; Vulkan accomplit la projection hidden, SiLU, la projection output, la projection skip et l'addition en un seul lot, ne relisant que la valeur finale. Les autres API de GpuTransformer ne s'activent pas par défaut, Vulkan est contrôlable séparément, sans exiger de transformation d'arena/push par décalage.

On conserve l'ordre par K, le biais et l'addition des trois projections d'origine. On ajoute siluGrid correct en grille 2D, évitant le mauvais usage de l'ancien noyau de validation ne regardant que X après qu'un forward complet de 8 canaux dépasse 65535 groupes. Il y a une différence entre l'exp GPU et la libm CPU ; il faut d'abord passer le budget existant opérateur/par couche/tous quantiles/cache ; si non atteint, on retire l'activation par défaut, sans desserrer le budget. Après passage, on fait un A/B indépendant avec TIMESTRAL_VULKAN_RESIDENT_RESIDUAL=0/1, sans changer d'autres optimisations en même temps. Si conservé, l'en-tête public et la disposition des poids ne changent pas, mais le changement du crochet virtuel interne exige une régression std-only par défaut une fois.

## Avance 300/30 GFLOP/s et multivarié (soir du 2026-10-24)

L'utilisateur exige de continuer à explorer le prefill 680M 10 % (300 GF/s) et le decode de tâche 1 % (30 GF/s), en complétant les performances 2/4/8 canaux, sans subagent. L'objectif n'est pas une garantie ; il faut rapporter l'écart bout en bout mesuré, sans remplacer par le pic pur opérateur. L'ancienne interface de restauration de Goal refuse toujours ; on avance selon l'autorisation directe de cette tour, sans prétendre que le Goal automatique est restauré.

### Protocole et quantité de tâche nécessaire

Extension du backendCompare existant : CHANNELS=1/2/4/8, 32768 échantillons par canal, même groupId mais données de canaux différentes, non un batch d'échantillons indépendants. Les targets sont une prédiction commune de tous les canaux, selon le préfill froid d'origine médiane 3, H=1 échauffé 5/chronométré 25 ; known est un Target plus R-1 KnownCov fournissant de vraies observations futures ; l'entrée paginée publique actuelle ne contient pas de rôles/futur, sans passer pour un toucher de cache, et rapporte séparément le forward conditionnel complet (échauffé 5/chronométré 5). La performance n'est pas une amélioration de qualité de prédiction ; la validation numérique est faite séparément.

Depuis cette tour, on fixe les FLOPs de tâche KV-only nécessaires, sans passer l'attention/MLP final inutilisable de Vulkan pour un produit. Le prefill 32k monovarié est 10.648682496 G, le decode H=1 est 0.016289280 G, exigeant respectivement ≤35.496 ms, ≤0.543 ms pour atteindre les objectifs ci-dessus. L'ancien 12.260475 G est la charge de travail principale du graphe d'exécution GPU non recoupé, conservé comme historique mais ne pouvant pas gonfler le MFU recoupé. La partie time multivariée s'étend par R, la partie variate est comptée par R², sans seulement multiplier les FLOPs monovariés par R. Le masque double flux complet de known n'a pas de charge de travail nécessaire dérivée ; on ne rapporte pas de MFU fictif.

### Ordre de mise en œuvre de ce lot et seuils d'entrée

1. Enregistrer sériellement et mesurer le cached 1/2/4/8 canaux et le BLAS de la même fenêtre, puis mesurer le forward conditionnel KnownCov, la fenêtre de performance ne chevauche pas la compilation de cette session
2. G1 complète les formes d'embedding réelles N768/K64, N192/K768 et N192/K64, en gardant le précis par K ; dispatch après ajout des formes de correctitude
3. Revoir les blocs vides de causal : recoupe seulement la zone d'encodage time, quand toutes les requêtes du bloc sont avant encoded et non bidirectionnel KnownCov, en conservant l'espacement de ligne réel de keyCount. C'est tranché cette tour sous relecture cached et horodatage ; le decode n'emporte pas cette branche de prefill, et utilise un shader indépendant si nécessaire pour éviter la dégradation registre/branche
4. Le prefill GPU KV-only s'arrête après achèvement du K/V de l'axe time final, sans exécuter l'attention, la porte/O, le variate suivant, le MLP, out_norm/relecture de hidden sans effet sur le contenu des pages ; le forward complet/trace/decode n'est pas recoupé. La première version calcule encore le QKV complet sans compter la Q inutile dans les FLOPs de tâche, et on considère ensuite la projection KV indépendante
5. G2 adopte 32×4 threads, chaque groupe 32 voies réduit une colonne de point K, quatre colonnes/groupe, dispatché par ligne, visant M≤8, les poids chargés en continu par K. Cela change l'ordre de réduction, tranché par l'atol/rtol d'origine et le budget relL2 hidden/tous quantiles, sans exiger l'identité d'octet, et sans desserrer le budget
6. A4 le long KV d'une requête est découpé par 128 clés, plusieurs groupes de travail par head/ligne, 64 threads chargeant K ensemble, PV conservant la continuité de la dimension de voie, sortant le max/dénominateur/PV non normalisé de chaque split, puis fusion FP32 indépendante. Sans hypothèse de largeur de warp, sans passer le résultat split en moyenne pour de l'attention ; le split masqué/vide doit avoir un traitement stable de dénominateur nul. Dispatch seulement à count1, time/headDim64 et long KV
7. Ajouter un candidat de réduction subgroup gated par capacité du périphérique : requête compute Basic/Vote/Arithmetic, le shader vérifie d'abord si chaque subgroup ne contient que la même requête et size32/64 ; sinon le groupe entier passe par la réduction d'origine. Si conforme, utiliser subgroupMax/Add plus un peu de fusion partagée, réduisant le barrier de groupe par 64 clés, sans dépendre de l'ordre des voies ni codé warp32
8. Contrôler gemmWide 64×64 tuile de sortie, 256 threads, registres 4×4 par thread, bloc K 16, partagé 8256 B, toujours précis par K. Activation seulement si le micro-benchmark résident et le bout en bout des grandes formes M du modèle principal en profitent tous les deux ; M128 et autres petites grilles gardent 32×32. Rotary ne fait que la réutilisation précise des fréquences inverses du même appel et des lignes de position identiques, sans cache inter-requêtes ajouté
9. Visant le fait que l'attention occupe toujours le premier rang côté périphérique, réévaluer un bloc 8 requêtes×64 clés de 128 threads : quatre accumulateurs QK et quatre dimensions PV par thread, réduction partagée 16 voies, 37632 B partagés ; éviter la chaîne sérielle d'une voie un score, toujours selon le masque et le softmax en ligne d'origine. L'ordre de réduction change ; il faut un budget figé et un contrôle du modèle, dispatch après passage
10. Chaque nouveau chemin numérique d'abord un petit oracle/contrôle du modèle puis la performance, en ne répétant qu'une fois le contrôle ayant un différend directionnel. Enregistrer le mode et l'écart réel monovarié/multivarié et KnownCov ; un objectif non atteint n'est pas écrit comme atteint

### État d'obtention de la référence ggml

Cette tour a de nouveau essayé le module figé de raw.githubusercontent.com et l'information de commit de api.github.com, tous refusés par la vérification non-IP publique. La recherche trouve [l'image figée mul_mm.comp](<https://tangled.org/okami.mom/llama-cpp-bonsai/blob/01fd9521c92e7882a3fa1083bb933e3bd9305bef/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp>) et [la notice du backend](<https://deepwiki.com/ggml-org/ggml/3.3-vulkan-backend>) ; les deux domaines refusent aussi, le corps n'est pas lu, l'implémentation n'est pas copiée, sans prétendre à la revue du micro-noyau ggml. Les candidats ci-dessus sont des implémentations basées sur le shader lu localement et le principe d'accès mémoire/réduction GPU ; après obtention de la source, on vérifiera la tuile et le dispatch du périphérique, sans laisser l'attente réseau bloquer les autres travaux indépendants.

## Attribution dédiée 680M : MFU faible et choix d'en-tête de relecture erroné (2026-10-24)

L'utilisateur pointe que le 680M fait environ 3 TFLOPS et que le bout en bout n'utilise que quelques pour cent. Ce jugement tient ; la tour précédente ne prouve que l'accélération locale du shader, sans terminer l'attribution du temps d'exécution ; appeler la soumission et le transport globalement goulot principal manquait aussi de preuve. Cette section remplace ces conjectures, sans passer la meilleure valeur d'une seule tour nouvellement apparue pour un optimisation terminée.

### Recalcul du MFU selon la portée d'exécution actuelle

Selon le pic nominal FP32 spécifié par l'utilisateur 3000 GF/s, sans utiliser le dénominateur 134.4 GF/s CPU pour expliquer l'efficacité GPU, et sans prétendre avoir mesuré la fréquence réelle. Un Target, toutes observations, 1024 patches, D192, FFN512, 8 axes time et 2 axes variate :

- Les projections Transformer sont 8.606711808 GFLOPs, les trois projections du résiduel de patch sont 0.427819008 GFLOPs
- L'attention effective des paires visibles time/variate est 3.225944064 GFLOPs, au total 12.260474880 GFLOPs de travail effectif principal
- Le decode de préfixe complet H=1 contient les trois projections de la tête ; le travail effectif principal est 0.016289280 GFLOPs, sans réutiliser la charge de travail de encode
- RMS, exp, RoPE, élément par élément, gestion de cache, copie et initialisation mémoire ne comptent pas de FLOPs, mais le temps compte dans le bout en bout. Vulkan fait aussi un travail partiellement inefficace sur les PV dont le masque est nul ; la quantité d'instructions réellement exécutées n'est pas égale aux FLOPs de modèle effectifs

Par conséquent l'encode précédent de 143.5030 ms équivaut seulement à 85.44 GF/s, MFU nominal environ 2.85 %. Les 3 TFLOPS sont un dénominateur hypothétique figé, pas la fréquence GPU de cette tour ni un compteur d'activité du matériel. La borne inférieure de calcul pur idéal du graphe actuel est environ 4.09 ms, non une latence bout en bout promissible ; une implémentation de dizaines à centaines de millisecondes a clairement encore un grand écart. L'ancien forward complet/formule J=4 ne peut pas s'appliquer directement au prefill KV-only ou au decode H=1.

### Outil d'attribution et frontières du chronométrage

Ajout de l'outil d'attribution vulkanProfile (outil local, non versionné) : chronométrage exclusif de l'API par la couche externe de GpuExecutor, chronométrage de soumission/attente/récupération à l'intérieur de l'exécuteur, et horodatage de requête Vulkan comptant chaque noyau et le copy. Le diagnostic du modèle par défaut est désactivé ; activé seulement dans l'outil explicite, sans changer le shader, l'ABI, le barrier de synchronisation ou la frontière de soumission. La requête utilise timestampValidBits de la file de calcul réelle, traite le retour de compteur ; la capacité de pool insuffisante signale dropped_events, sans fabriquer de durée nulle.

L'outil encode 3 fois avec la même entrée 32k et un pool de pages froid, decode échauffé 5 fois puis chronométré 5 fois en diagnostic, et enregistre les statistiques complètes selon la médiane wall de cette tour, sans prendre séparément la médiane de chaque colonne pour assembler une exécution. Les événements GPU sont marqués à la même étape compute ou transfer ; l'étendue de soumission va de haut en bas ; la somme des événements, l'étendue GPU et l'attente de fence CPU se chevauchent mutuellement et ne s'additionnent pas. Les horodatages peuvent changer la planification et la variation de durée GPU de cette machine est marquée ; le temps de diagnostic ne remplace pas le benchmark 5/25 de backendCompare sans diagnostic. Les deux journaux précoces à 11.832656 G ne comptaient que le graphe Transformer ; la formule finale a complété les projections patch/head comme 12.260475 G/0.016289 G ci-dessus.

### Goulot hôte d'encode confirmé

Dans `l'enregistrement avant subdivision`, un encode wall 176.3228 ms, étendue de soumission GPU 85.96935 ms ; allocation CPU 0.0342 ms, enregistrement de dispatch 0.5251 ms, récupération 0.2878 ms, aucun n'est l'item dominant. Le copy upload+download GPU fait environ 1.92 ms ; on ne peut pas expliquer tout l'écart par « le transfert CPU vers GPU est lent ».

`L'enregistrement après subdivision` confirme les données suivantes, comparées à `l'enregistrement corrigé` :

| Même item de diagnostic | Avant correction | Après relecture cached |
| --- | ---: | ---: |
| encode wall ms | 228.0989 | 143.5917 |
| étendue de soumission GPU ms | 120.22921 | 120.94282 |
| CPU map+lecture+unmap ms | 77.5258 | 1.5239 |
| octets relus | 17.25 Mio | 17.25 Mio |
| drapeaux mémoire de relecture réels | 6, Visible+Coherent | 14, Visible+Coherent+Cached |

Le type1 signalé par le 680M a flags6/heap1, le type3 a flags14/heap1. L'implémentation d'origine choisissait toujours le premier type host-visible cohérent, et prenait aussi le staging adapté à l'écriture CPU pour la lecture CPU. La lecture CPU sans HOST_CACHED est très lente ; la mémoire physique unifiée ne signifie pas que toutes les mappings de mémoire Vulkan sont aussi adaptées à l'accès CPU. Ce n'est pas ici une explication de transfert PCIe de GPU dédié, encore moins un pic de GEMM du périphérique insuffisant.

Corrigé : le pool de réutilisation de relecture indépendant choisit en priorité HOST_VISIBLE|HOST_COHERENT|HOST_CACHED, sans mélanger avec le staging d'upload ; sans type cohérent cached, retour au type cohérent d'origine, sans introduire de sémantique flush/invalidate non traitée. La relecture attend encore la fence et la visibilité transfer→hôte ; toutes les données sont réellement relues ; l'API publique et la précision sont inchangées.

### A/B bout en bout sans diagnostic

Toutes les tâches sérielle, sans chevauchement avec le build de cette session ni d'autres mesures GPU. `Première tour` l'ancienne version 228.5427/10.8646 ms, BLAS 181.9911/4.7145 ms, prouve clairement que l'ancienne version était plus lente que BLAS à ce moment. Puis avec le programme ancien sauvegardé et la version cached dans deux ordres, protocole toujours préfill froid médiane 3, decode échauffé 5 chronométré 25 :

| Bouée | encode/decode ms ancienne version | encode/decode ms version cached | encode/decode ms BLAS |
| --- | --- | --- | --- |
| cached d'abord, ancienne version après | 185.1625 / 16.9553 | 88.0076 / 11.9052 | 182.7768 / 4.8753 |
| ancienne version d'abord, cached après | 196.6105 / 11.9873 | 85.2186 / 16.8519 | 186.6453 / 4.6018 |

L'enregistrement inversé brut est dans `final-ab`. L'encode des deux tours s'améliore d'environ 2.10 à 2.31 fois, la version cached est environ 2.08 à 2.19 fois plus rapide que le BLAS de la même fenêtre, mais le MFU n'est qu'environ 4.64 % à 4.80 %, encore loin d'une optimisation suffisante. La direction du decode n'est pas cohérente et reste nettement plus lente que BLAS ; cette tour ne proclame pas d'accélération du decode, sans continuer à mesurer en boucle pour obtenir un résultat satisfaisant. Le temps GPU a aussi une variation marquée ; la fréquence, la puissance ou d'autres charges du système ne sont pas collectées ; on ne peut pas attribuer de son propre chef au pilote ou au throttling.

### Priorités d'optimisation après correction

1. L'erreur d'attribut de cache de la grande relecture est corrigée. La prochaine étape évalue si le CPU a vraiment besoin de chaque valeur intermédiaire, et si les 20 soumissions d'encode (embedding trois fois, graphe entier une fois, KV seize fois) peuvent être réduites ; pas réécrire d'abord le pool de descripteurs
2. L'attention est le premier point chaud GPU : le diagnostic après correction time attention huit fois au total 72.11564 ms, environ 60 % de l'étendue GPU ; une autre tour 47.58 ms, le classement est cohérent mais les valeurs absolues ne sont pas mêlées. Le bloc actuel de 512 threads/39168 B partagés a encore beaucoup de barrier de groupe et de produit scalaire 64 dimensions sériel. On devrait comparer sous horodatage GPU une tuile/réduction plus appropriée et les blocs vides de causal, plutôt que de regarder seulement le bruit bout en bout masqué par la relecture
3. Le dispatch G1 a manqué les grandes formes d'embedding réelles : M1024/N768/K64 et M1024/N192/K768 utilisent encore v1 ; le diagnostic peut atteindre 3.81 et 12.51 ms respectivement ; le micro-benchmark et le contrôle de budget de ces formes passent avant l'optimisation des noyaux élément par élément déjà rapides
4. Le decode a besoin d'un vrai GEMV et d'une attention de long KV dédiée. La projection M=1 utilise encore le GEMM 16×16 ; 15/16 lignes de sortie sont inefficaces ; la forme principale fait seulement environ 2 à 8 GF/s. A2 à count=1 n'a que trois groupes de travail de tête ; sept emplacements de requête sont inefficaces et ne remplissent pas les 12 CU ; il faut un partage de voie séparé et un split-K/fusion partielle, sans copier le noyau de prefill
5. Le decode a encore 12 Mio d'upload de préfixe explicite et seize fusions ; environ 3 ms hors API CPU incluant la vérification des pages, l'assemblage du préfixe et le numérique commun. D'abord plus d'attribution, puis miroir de page/lecture directe, sans passer tout le temps CPU inconnu pour une copie PCIe

L'ancien A3a « gain bout en bout instable » ne signifie pas que le gain du noyau GPU est prouvé nul : la relecture CPU d'ancienne génération 80 ms et la variation GPU masquent les différences côté périphérique. Conserver le résultat historique, mais réexaminer le noyau de prefill indépendamment après correction de mémoire et horodatage disponibles, sans emporter la branche de recoupe inconditionnellement dans le decode. Le pic FMA et le débit de la multiplication-addition précise séparés ont aussi une granularité différente, mais même en reconnaissant la contrainte de la borne de calcul, cela n'explique pas l'efficacité globale de quelques pour cent actuelle, sans être un motif d'arrêt de l'optimisation.

Cette tour n'a pas implémenté de GEMV, de split-K ni de nouvelle tuile d'attention, et n'a pas desserré le budget numérique. Ajout de cohérence d'octets de l'interrupteur de diagnostic, de tests de comptage et de relecture cohérente, validation double GPU et passage ciblée Tiny/Real. Le code CPU et le shader par défaut ne changent pas, sans relancer tous les tests CPU en boucle pour le diagnostic.

## État et frontières de cette tour

Le 2026-10-24, redessiné à la demande de l'utilisateur, la planification précède l'implémentation, et après achèvement de la planification on avance selon l'arbre, sans subagent. Les anciens L1 à L6 ne sont plus l'ordre de mise en œuvre ; les enregistrements d'échec restent à la fin de ce document. Le point de retour v1 et la méthode de restauration sont dans la notice de sauvegarde ; ce répertoire de sauvegarde appartient aux fichiers de travail locaux, non publiés avec le dépôt.

La priorité est la précision, la facilité d'usage, le gain de performance mesurable. L'API publique, les fichiers de modèle et la disposition de données CPU ne changent pas ; le build std-only par défaut n'introduit pas de Vulkan ni de BLAS système. L'optimisation par défaut n'utilise que FP32, en conservant la multiplication-addition par K et la sémantique du masque ; FMA, réduction parallèle et basse précision sont validés séparément, sans les activer en desserrant le budget.

## Ce que les échecs changent

- Ne plus changer ensemble l'arena, le cache de descripteurs, l'upload des poids et la disposition push des dix shaders. Conserver les buffers indépendants v1 et abiVersion=1, en étendant par de nouveaux kernels plutôt que de remplacer tout le protocole.
- L'optimisation de l'exécuteur et l'optimisation des opérateurs sont des branches indépendantes. La réutilisation de pool n'exige pas de décalage de tenseur, de SSBO dynamique ni de migration de poids ; le blocage de mémoire partagée n'exige pas non plus de changer l'exécuteur.
- L'exception AMD n'a pas de conclusion de racine. L'absence d'erreur de validation ne signifie pas que l'application est correcte ; le passage 4060 ne remplace pas l'acceptation AMD. Tout échec de périphérique ne désactive que cette branche, sans bloquer d'autres optimisations indépendantes.
- Ne présupposer pas que chaque couche a un gain. D'abord écrire l'hypothèse de coût, faire un contrôle de petite portée ; atteinte des exigences numériques mais gain bout en bout nul, enregistrer comme candidat ou refuser, sans accumuler de complexité par défaut.
- Ne pas relancer en boucle les tests de tout le dépôt ni créer un nouvel arbre de build pour une seule expérience. Réutiliser l'arbre Release et les tests existants, valider seulement les frontières touchées ; à la phase formelle, refermer en exécutant une fois l'arbre GPU et la régression du build par défaut.

## Base et hypothèses de goulot

Base historique 32k de la cinquième période (non A/B même période de cette tour) :

| Périphérique | encode ms | decode ms | Première valeur de sortie |
| --- | ---: | ---: | ---: |
| RTX 4060 Laptop | 120.8538 | 7.7143 | 2.3805 |
| iGPU AMD gfx1035 | 236.9136 | 11.8549 | 2.3806 |

Le checksum de backendCompare (outil local, non versionné) est la première valeur de sortie, non une validation numérique complète. encode est un préfill de pool de pages froid de 32768 points d'échantillonnage, 1024 patches, médiane de 3 ; decode est un préfixe complet H=1, échauffé 5 fois, chronométré 25 fois en médiane. Le gain de ré-optimisation est tranché par A/B sériel de même périphérique, même fenêtre avec le programme v1 sauvegardé de cette tour, sans calculer l'accélération avec la valeur la plus rapide historique.

Coûts confirmés dans le code et hypothèses encore à mesurer :

- [Executor](<../../timestral/vulkan/executor.cpp>) crée un pool de descripteurs et un ensemble de descripteurs par dispatch ; c'est un coût CPU/pilote certain, mais la part est inconnue.
- [gemm](<../../timestral/vulkan/shaders/gemm.comp>) un thread calcule une sortie ; un groupe 16×16 à M=1 n'a que 16 threads écrivant ; la boucle K met des barrier répétés ; le chargement global de W n'est pas fusionné dans la direction K. La tuile de registres et le chargement W fusionné ont chacun une valeur indépendante.
- [attentionFused](<../../timestral/vulkan/shaders/attentionFused.comp>) un ensemble de 64 threads par requête/tête ; chaque voie calcule sériellement le QK d'une clé ; chaque bloc réécrit la somme partielle PV dans la sortie globale puis la relit. La lecture de V est continue le long de j, sans l'appeler simplement lecture dispersée ; pas de K/V partagé inter-requêtes. 1024×1024×192×4 environ 0.805 Go est la borne supérieure de lecture logique de V d'une couche sans masque, non le 8 Go de flux de videomémoire réel prétendu par l'ancien texte ; le toucher de cache et l'optimisation du compilateur changent aussi le flux réel.
- Le decode paginé assemble le préfixe sur l'hôte, upload K/V et fusionne par couche time. La résidence inter-appels peut supprimer le transport redondant, mais il faut d'abord résoudre l'identité des pages, l'éviction et le cycle de vie du modèle, sans prendre le pageId CPU seul comme clé stable.

## Arbre d'optimisation R

La figure ci-dessous est l'espace d'optimisation et l'ordre de mise en œuvre historique ; l'état est dans l'enregistrement de mise en œuvre. Chaque feuille conserve un retour v1 indépendant ; après mesure 680M, la priorité est ajustée en relecture cached, attribution horodatée de l'attention, les formes d'embedding manquantes, le vrai GEMV et l'attention de decode, puis entre miroir paginé et fusion selon le gain, sans mettre plus le pool de descripteurs en avant de la ligne principale.

~~~text
R  Optimisation en profondeur FP32, racine en buffers indépendants v1/abiVersion=1
├── R0  Base stable et observation légère
│   ├── Sauvegarder le programme de base v1, A/B sériel double GPU
│   └── Statistiques seulement selon le besoin de dispatch, soumission, transport et temps de point chaud
├── E   Coût CPU/pilote de l'exécuteur, branche indépendante des opérateurs
│   ├── E1  Réutilisation de pool de descripteurs multi-set, reset après fin de fence
│   ├── E2  Préallocation de set par emplacement, ou cache par buffer physique/étendue
│   ├── E3  Relecture/soumission en lot, conserver le contrat de synchronisation et empoisonné
│   └── E4  Pool séparé upload/relecture, relecture privilégiant HOST_CACHED (corrigé)
├── G   Projection FP32, d'abord conserver l'ordre par K
│   ├── G1  Tuile de sortie 32×32, 16×16 threads, accumulation 2×2 par thread
│   ├── G2  Noyau étroit M≤4, voie le long de la colonne de sortie, K sériel conservant la précision
│   └── G3  Optionnel GEMV K parallèle, FMA, disposition de panneau des poids
├── A   Attention, time et variate dispatchés séparément
│   ├── A1  PV en registres headDim=64, évitant l'aller-retours de sortie par bloc de clés
│   ├── A2  Bloc de K/V partagé multi-requête FP32, softmax en ligne
│   ├── A3  Recoupe des blocs Full/vides, masque strict double flux
│   └── A4  Decode split-K et noyaux dédiés variate petits
├── F   Fusion et réutilisation de données statiques
│   ├── F1  residual+RMSNorm, épilogue porte+projection
│   └── F2  Réutilisation de rotary d'un appel, puis réévaluer un cache borné inter-appels
├── K   Résidence des données paginées, sans changer la sémantique du pool de pages public
│   ├── K1  D'abord mesurer la part d'assemblage/upload hôte et de fusion
│   ├── K2  Identité de page stable + génération + contrat d'inactivation, miroir du périphérique
│   └── K3  Attention en lecture directe du tableau de pages, annuler la fusion/upload du préfixe
└── X   Spécialisation de périphérique optionnelle, non une dépendance du chemin par défaut
    ├── Réduction subgroup : requêter la capacité subgroup, sans codé warp=32
    └── Cooperative matrix : détection de caractéristique par composant + budget de précision indépendant
~~~

## Conception des branches et conditions d'entrée

### R0 et E : d'abord éliminer la création de pool, sans changer l'adressage

E1 chaque pool contient 256 sets, au maximum 1280 descripteurs de buffer de stockage, en augmentant le pool à la demande. Chaque dispatch alloue toujours un set indépendant et écrit le buffer/étendue réel, et n'est pas modifiable après enregistrement jusqu'à la fin de fence. La soumission réussie libère d'abord le command buffer puis reset les pools utilisés, pour réutilisation par la soumission suivante. Le même lot après download peut continuer l'enregistrement ; les poignées scratch après end restent invalides. Le pool possède les poignées DescriptorSet ordinaires, sans construire de UniqueDescriptorSet incomplet, sans libération par set, donc sans besoin de eFreeDescriptorSet.

E1 ne change que l'exécuteur et la régression de cycle de vie, sans changer le shader, le manifest ni l'interface GPU partagée. La validation couvre le premier dispatch du même cb, la traversée de la frontière 256, d'abord upload puis dispatch, continuation après download, réutilisation inter-lots ; les deux périphériques doivent passer. Après erreur, avant vidange le pool ne peut pas être reset, suivant la sémantique empoisonnée.

E2 ne démarre qu'après E1 si l'allocation/mise à jour de set reste un point chaud. Si le cache est adopté, la clé comprend la disposition, l'identité du buffer réel, le décalage, l'étendue et la génération, sans cache avec des poignées logiques qui s'invalident. La mise à jour d'un set déjà référencé est interdite avant l'achèvement de la commande. E3 considère d'abord la relecture d'un coup de plusieurs résultats, sans supprimer les fences, barrier nécessaires ni changer le contrat d'interface existant par retour asynchrone.

### G : d'abord corriger l'accès mémoire et la réutilisation de registres, puis changer l'algorithme numérique

La tuile de sortie de G1 est 32×32, le groupe de travail est 16×16 soit 256 threads, non 32×32 threads. Bloc K 16, A partagé 32×16, W transposé en 16×33 (avec rembourrage), au total 4160 B. Chaque thread charge deux A et deux W, W continu le long de K, calcule un résultat de registres 2×2 ; chaque sortie conserve le même ordre K que v1, la multiplication-addition précise, le biais en fin. D'abord charger fusionné scalaire pour éviter les risques d'alignement vec4 et de bloc de queue. La ligne/colonne de queue et le K non entier 16 sont remplis par zéro et participent à tous les barrier.

G1 n'est activé que si M et N sont assez grands ; les formes principales du modèle sont K=192/512, N=192/576/1024 ; la borne finale est tranchée par le micro-benchmark résident et le bout en bout, sans prendre par défaut M>4 tous par le nouveau noyau. Les formes non mesurées ou dégradées gardent gemm.

G2 à M=1 à 4 essaie d'abord les voies voisines responsables des colonnes de sortie voisines, K sériel par voie, bloc d'entrée partagé, évitant l'inactivité de 15/16 lignes. Si la bande passante de la disposition W est insuffisante, on essaie ensuite la transposition au chargement ou le pré-empaquetage de panneau. La réduction K parallèle et le FMA de G3 changent l'ordre d'addition ; il faut une vérification d'erreur séparée, sans lier à G1. La basse précision des poids n'appartient pas à cette branche.

### A : d'abord réduire les allers-retours de sortie, puis faire de vrais blocs multi-requêtes

A1 ajoute un noyau de même ABI pour headDim=64, un registre PV par voie ; les blocs QK et softmax avancent toujours par 64 clés ; on conserve la réduction et le masque v1, en écrivant le résultat à la fin. Interdiction de présupposer que la sortie globale n'est certainement pas optimisée en registres par le compilateur ; le gain doit être mesuré. Les autres headDim passent par v1.

Le premier candidat d'A2 est limité à l'axe time, headDim=64 ; le bloc de sortie est 8 requêtes × 64 dimensions. On adopte 64×8 soit 512 threads ; les 64 voies de chaque requête calculent toujours chacune le QK d'une clé, 64 voies du même groupe réduisent le softmax, chaque voie accumule une dimension de sortie. Le bloc K/V 64 clés est partagé ; l'adressage K transposé utilise 65 colonnes de rembourrage, plus Q et scores/réduction indépendants par requête ; le total partagé est 39168 B, inférieur à la borne commune 48 Ko des deux périphériques. Le choix de 512 threads plutôt que 128 threads du premier jet est pour conserver d'abord l'arbre de réduction 64 clés d'origine et l'ordre par K/PV, sans introduire en même temps un autre partage/arrondi de variation ; si le périphérique n'atteint pas 512 threads ou la borne partagée, on garde v1. Les différents emplacements de requête de l'axe variate correspondent à différents K/V, sans réutiliser le même bloc partagé ; on continue l'optimisation indépendante. On choisit d'abord ce candidat de tuile, sans énumération complète des paramètres. La réutilisation du bloc K/V, la recalibrage du maximum/dénominateur en ligne reste FP32 ; les lignes entièrement masquées sortent 0 ; la requête/clé de queue ne sort pas avant le barrier. Le blocage et l'ordre de réduction du softmax en ligne changent ; on ne promet pas d'avance la bit à bit.

A3 suit la classification d'origine timeAllowed/variateAllowed. La première feuille étroite A3a vise seulement le cas où toutes les requêtes du même bloc time sont inférieures à encoded et n'appartiennent pas au rôle bidirectionnel KnownCov : à ce moment timeAllowed exige clé≤requête, donc tous les blocs suivants de 64 clés dépassant le plus grand emplacement de requête du bloc sont vides, et on peut terminer uniformément la boucle K/V. On conserve toujours le keyCount d'origine comme espacement de ligne, le masque de clé de queue d'origine et le prédicat par paire intra-bloc, sans raccourcir l'espacement de ligne réel K/V. Les autres cas ne sont pas recoupés ; activation seulement après passage du contrôle bit à bit et du gain bout en bout. Les autres classifications Full/vides sont traitées indépendamment plus tard ; known-future, valeurs manquantes, groupIds différents ne peuvent pas dégénérer en causal pur. A4 est fait seulement si l'attention de decode est vraiment le goulot ; le split-K a besoin de buffers partiels et de fusion supplémentaires ; les petites requêtes ne bénéficient pas naturellement. Le petit variate évite la boucle de 64 clés générique et le gros bloc partagé quand keyCount est très petit.

### F : la fusion s'articule autour de valeurs intermédiaires stables, sans copier le flux du modèle

F1 d'abord la sortie résiduelle et RMSNorm dans le même noyau, en gardant strictement l'ordre d'origine de y=x+delta, de la somme de carrés et de (y*inverse)*scale ; les appels exigeant une trace ou deux sorties peuvent encore lire y. Sans porter la sémantique CPU dans le backend. La fusion de porte dans l'épilogue de projection n'est activée que si numériquement identique et sans calcul redondant ajouté.

La relecture du graphe de couches v1 confirme que rotary est déjà uploadé une seule fois hors de la boucle de couches ; le jugement de l'ancienne L5 « upload répété par couche » ne tient pas ; la partie d'un appel de F2 est donc directement notée existante, sans implémentation répétée. On n'ajoute un cache que si le coût d'upload inter-appels de rotary mesuré mérite l'optimisation ; la clé doit inclure le contenu réel de rotary ou tous les paramètres de position, sans seulement count/contextPatches, car continuousRope et le décalage de position changent les données. Le cache a une borne de capacité, sans remplacer la gestion des ressources par un upload persistent illimité.

### K et X : items à haut risque retardés avec conditions de sortie claires

K2 avant de concevoir l'interface du miroir de page, lire le comportement de génération, de référence et d'éviction du pool de pages ; la clé distingue au moins le modèle, l'identité du pool, le pageId et la génération de contenu. scratch, éviction LRU, abort, géométrie de canaux différente, préfixe partagé, FIFO doivent tous être inactivés correctement. Le pool de pages CPU reste la source d'autorité ; la relecture des nouvelles pages est conservée, sans changer le hachage de page ni la sémantique de lecture seule partagée. Une dépassement de capacité peut jeter le miroir et retomber sur l'upload, sans exiger que chaque page CPU réside en même temps sur le GPU. Sans part de transport claire ni contrat de cycle de vie, K3 n'est pas implémenté.

X ne promet pas que l'existence de VK_KHR_cooperative_matrix permet d'exécuter une certaine combinaison FP16/BF16 ; il faut requêter les dimensions de matrice réellement supportées, les types de composant et la capacité d'accumulation FP32. TF32 n'est pas un nom d'interrupteur de BF16. Cette tour n'ajoute pas d'abord d'option basse précision ; si le budget de précision n'est pas fixé, il reste non implémenté.

## Acceptation concise et efficace

1. Chaque feuille ne fait tourner que les frontières touchées : un changement de shader fait glslc+spirv-val+réflexion ABI et un petit jeu oracle du nouveau noyau ; une pure réutilisation de pool ne recompile pas dix shaders non changés.
2. Après connexion au modèle, exécuter les gpuModelTiny/Real existants, une fois sur chaque périphérique, incluant par couche, tous quantiles et pagination ; en cas d'échec, d'abord localiser, sans compter sur des dizaines de re-fours intégrales.
3. Le candidat opérateur ne mesure que les formes principales du modèle et les blocs de queue. Le micro-benchmark résident met l'upload/relecture hors du chronométrage, ou le marque explicitement comme contenu d'enregistrement/soumission hôte ; l'ancien benchmark de transport par opérateur ne peut pas servir de débit pur shader.
4. La fenêtre de performance est sérielle avec GPU exclusif, en A/B avec le v1 sauvegardé ou le dernier nœud accepté. Un gain nettement nul arrête directement ce candidat ; près du bruit, au plus un A/B inversé supplémentaire. Si encore incertain après deux tours, on garde le défaut v1, sans ajuster les seuils pour courir après la meilleure valeur d'une seule fois.
5. Seuil d'activation par défaut : la correctitude passe tout, le chemin cible s'améliore stablement et l'autre chemin principal n'a pas de dégradation claire. Un écart directionnel dépassant environ 3 % sert de valeur déclenchant une re-mesure, non une garantie de précision ou de signification statistique absolue. Le dispatch par périphérique/forme doit être centralisé, sans avaler silencieusement d'erreurs puis changer de résultat.
6. Le budget numérique suit les tests existants, sans remplacer par le checksum de la première sortie : opérateur atol=1e-5, rtol=1e-4 ; hidden atol sensible à l'échelle et relL2≤1e-5 ; quantiles finaux atol=1e-4, rtol=1e-3 ; la cohérence de cache suit les règles figées existantes. Un changement pur d'exécuteur attend une sortie bit à bit identique.
7. Une phase significative refermée fait une seule fois l'arbre GPU complet, la régression std-only par défaut et la vérification de style, et met à jour ce document et l'enregistrement de progression local. Le dénominateur MFU nominal CPU 134.4 GF/s ne sert pas à passer pour un taux d'utilisation du matériel GPU.

## Enregistrement de mise en œuvre

| Nœud | État | Correctitude | Performance de cette tour |
| --- | --- | --- | --- |
| R0 | Terminé | Programme v1 de cette tour sauvegardé | Initial 4060 126.6051/7.7350 ms, AMD 247.1702/12.1213 ms |
| E1 | Expérimenté, retiré | Validation double GPU, frontière 257/traversée de soumission/premier dispatch, Tiny/Real passés | 4060 decode légère amélioration, AMD encode dégradé, non activé globalement |
| G1 | Accepté et activé par forme limitée | Double GPU 9 formes × interrupteur de biais bit à bit identique à v1, petite oracle CPU de bloc de queue passée, Tiny/Real incluant long contexte 129 patches passé | Projection 1024 lignes principale environ 2.27 à 2.32 fois sur 4060, environ 2.34 à 4.04 fois sur AMD |
| A1 | Évalué, seulement le noyau de diagnostic explicite conservé, sans sélection automatique | Double GPU bit à bit identique à v1 | PV en registres seul sans gain stable et significatif |
| A2 | Accepté, activé quand time/headDim64/nombre de clés≥128 et capacité satisfaite | Requête/clé de queue, grille 2D, multi-lignes multi-têtes, manquant/rôle/base/encoded bit à bit identique à v1 ; modèle double GPU passé | Micro-benchmark time 1024 requêtes environ 2.96 fois sur 4060, environ 3.23 fois sur AMD, voir ci-dessous |
| A3a | Correctitude passée, gain instable, retiré | Double GPU bit à bit passé | AMD encode amélioré, 4060 deux fois directions incohérentes et decode dégradé, non activé par défaut pour l'instant |
| G2, autres items d'A3, F, K, X | Planifié | Non implémenté | Non mesuré |

### Expérience E1 sans gain net

La pure réutilisation de pool ne change aucun shader ni chemin d'upload ; la correctitude et la validation ciblées double GPU passent toutes. Après sauvegarder séparément le programme E1, deux contrôles 32k inversés :

| Périphérique/tour | encode/decode ms v1 | encode/decode ms E1 |
| --- | ---: | ---: |
| 4060 tour 1 | 128.4104 / 8.2558 | 127.9981 / 7.6962 |
| 4060 inversé | 126.7515 / 8.0593 | 123.7811 / 7.8064 |
| AMD tour 1 | 248.1191 / 12.0481 | 275.5350 / 12.0973 |
| AMD inversé | 265.5561 / 12.0808 | 275.2834 / 12.0375 |

4060 decode amélioration environ 3 % à 7 %, AMD decode sans amélioration directionnelle et encode dégradé environ 4 % à 11 %. Retrait d'E1 selon le seuil, sans ajouter de cas particulier de fabricant pour conserver un gain faible ; l'exécuteur reste toujours un pool indépendant par dispatch. La régression de frontière de pool est conservée comme test de cycle de vie de descripteurs de grande quantité ; l'ancien problème d'arena n'est pas reproduit dans E1, mais la racine reste inconnue. Si E2 est de nouveau étudié à l'avenir, il faut reprendre une preuve de gain indépendante.

### Micro-benchmark résident G1

Ajout d'un mode resident au vulkanBench existant (outil local, non versionné) : 5 échauffements, 25 chronométrés, 8 dispatch consécutifs à chaque fois, chronométré par dispatch, incluant enregistrement hôte, barrier, soumission et fence, sans upload/relecture, non appelé temps GPU pur. Les médianes de cette tour ci-dessous, toutes les sorties bit à bit identiques à v1 :

| M/N/K | 4060 v1→G1 ms | AMD v1→G1 ms |
| --- | ---: | ---: |
| 1/576/192 | 0.020112→0.020788 | 0.037187→0.035425 |
| 128/192/192 | 0.030113→0.022238 | 0.065200→0.039800 |
| 1024/576/192 | 0.382188→0.165175 | 1.123037→0.277750 |
| 1024/1024/192 | 0.619525→0.266575 | 1.923963→0.822250 |
| 1024/192/512 | 0.327550→0.144425 | 1.069613→0.301212 |

Le choix de noyau dans le modèle se concentre sur Vulkan Executor, sans changer le contrat GpuTransformer inter-API. G1 n'est activé que pour la grille gemm native complète de M≥128 et (K,N) de (192,192/576/1024) ou (512,192) ; M=1 et les formes non mesurées gardent v1. L'Executor indépendant interne désactive par défaut le choix automatique de noyau, pour appeler clairement les deux noyaux par oracle et benchmark ; le modèle l'active par défaut, TIMESTRAL_VULKAN_OPTIMIZATIONS=0 peut désactiver globalement le choix automatique de noyau, 1 active, une valeur illégale est explicitement refusée. Cet interrupteur ne change pas l'ensemble de fichiers du paquet de shaders compilé.

### Expérimentation d'attention A1/A2

Le micro-benchmark résident d'attention fait 4 dispatch à chaque fois, 5 échauffements, 25 chronométrés, incluant enregistrement/barrier/soumission/fence et sans transport ; le masque est causal ordinaire, trois têtes, headDim64. A1 non activé automatiquement, A2 seulement pour l'axe time, en conservant le masque double flux complet.

| Périphérique/forme | ms v1 | ms A1 | ms A2 |
| --- | ---: | ---: | ---: |
| 4060 count1024/base0 | 2.804500 | 2.660175 | 0.946100 |
| 4060 count1/base1024 | 0.226275 | 0.222725 | 0.088600 |
| AMD count1024/base0 | 13.108300 | 13.059775 | 4.052275 |
| AMD count1/base1024 | 0.448200 | 0.444650 | 0.189950 |

A1 une autre tour 4060 long prefill précédemment 2.654600→2.644050 ms, AMD 12.581050→13.006950 ms, sans grand gain stable, sans noter la réécriture en registres elle-même comme accélération principale. L'amélioration d'A2 comprend aussi le chargement K fusionné et la lecture partagée ; une seule requête bénéficie sans réutilisation de requête ; le prix est le grand groupe de travail et l'occupation partagée 39168 B, donc sans sélection automatique pour les clés courtes, variate ou autres headDim.

### 32k bout en bout et variation

G1 seul première fois 4060 123.3685/8.4713 ms contre v1 130.5715/8.1300 ms, AMD 248.7492/14.8189 ms contre 314.2363/14.9267 ms. Cela montre que le facteur du micro-benchmark de projection ne peut pas s'extrapoler directement au modèle entier, et que le decode n'est pas le chemin cible de G1.

G1+A2 première fois A/B d'interrupteur du même programme 4060 110.7752/7.6756 ms contre v1 131.2195/8.2253 ms, AMD 199.1684/12.4841 ms contre 264.6728/12.0622 ms. Le decode de AMD cette tour est légèrement plus lent ; la re-mesure du protocole suivant change beaucoup ; on ne rapporte pas seulement la meilleure tour.

Dernière fenêtre inversée, en adoptant le même programme et le shader A2, encode/decode sont respectivement ; une mesure du candidat A3 est insérée pendant le contrôle v1 voisin, tous sériels, sans build ni tâche GPU de cette session :

| Périphérique | ms v1 | ms G1+A2 acceptés | rapport encode/decode | Première valeur de sortie |
| --- | --- | --- | --- | --- |
| RTX4060 Laptop | 125.9646 / 8.0188 | 109.8991 / 6.7479 | 1.146 / 1.188 | 2.3805 |
| iGPU AMD | 263.0007 / 12.9188 | 143.5030 / 10.5634 | 1.833 / 1.223 | 2.3806 |

L'encode AMD inter-tours varie de 143 à 199 ms, decode de 10.6 à 12.5 ms ; on ne peut pas prendre la dernière tour comme niveau fixe du matériel. Le bout en bout ne contient pas seulement le shader, mais aussi la normalisation CPU, l'embedding/tête, le pool de pages, l'enregistrement et la relecture. Cette tour ne met pas à jour tous les backends CPU, sans calculer un nouveau rapport d'accélération déterminé avec une seule valeur CPU ancienne. La première valeur de sortie sert seulement de fingerprint rapide ; les conclusions numériques viennent de la comparaison bit à bit des opérateurs et des tests par couche/tous quantiles/pagination.

### A3a correct mais instable, retiré

A3a recoupe seulement les blocs de clés futures de zone d'encodage prouvés vides ; le contrôle bit à bit double GPU passe. Première tour A2→A3 AMD 149.7078/10.6403→130.7113/10.2023 ms, 4060 109.1403/6.8161→123.8143/7.6405 ms. Une tour inversée AMD 143.5030/10.5634→132.8497/10.3990 ms, 4060 109.8991/6.7479→100.9962/7.2785 ms.

L'encode AMD s'améliore sur deux tours, l'encode 4060 a une direction incohérente, le decode se dégrade sur deux tours. Selon le plan, sans continuer à mesurer en tour jusqu'à satisfaction, A3a est retiré ; le code source du candidat et le jeu de shaders avant recoupe sont conservés dans un répertoire temporaire local, et l'attentionTile officiel est A2. On pourra plus tard séparer un noyau de prefill dédié indépendant et réévaluer, plutôt qu'emporter inconditionnellement la branche de recoupe dans le decode.

### Fermeture de cette phase

Le build incrémental final Release sans avertissement, ctest GPU complet 29/29 (AMD, 81.88 s), modèle ciblée 4060 2/2 et validation de l'exécuteur passés ; std-only par défaut complet 25/25 (77.77 s). Le contrôle d'opérateurs renforcé utilise memcmp pour vérifier octet par octet. Le cas de deux lignes de 129 patches ajoutés sur 5M réel, les budgets par couche et de tous quantiles sont inchangés ; sans passer la courte régression d'origine de seulement 3 patches pour une couverture du nouveau dispatch. Le spv final d'A2 et le hachage de module du benchmark inversé ci-dessus sont identiques.

C'est la fermeture de la première série de nœuds acceptés, non l'achèvement de tout l'arbre R ; la projection étroite G2, la fusion suivante, le miroir de KV et la spécialisation de périphérique doivent continuer selon les conditions d'entrée.

## Sources de recherche et documents non encore obtenus

Cette tour 2026-10-24 a de nouveau essayé d'obtenir le commit figé et le corps du shader ggml ; api.github.com et github.com sont tous refusés par la vérification d'adresse non publique de web_fetch. La recherche ne trouve que les chemins de [flash_attn_cm2.comp](<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp>) et du [générateur de shaders](<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp>), sans commit figé ni corps lu, sans copie de l'implémentation. Cet item peut être complété en lecture indépendante, sans bloquer l'optimisation FP32 d'origine ; après récupération du réseau, d'abord fixer le commit, revoir les dépendances et la licence, puis citer l'algorithme concret, sans passer le noyau spécial de matrice coopérative directement pour un noyau Vulkan1.1 générique.

Les bornes de périphérique connues sont AMD push128 B/partagé64 Ko, 4060 push256 B/partagé48 Ko ; les deux sont conçus ensemble selon push128 B/partagé48 Ko ; l'ABI reste v1. La source est la mesure de périphérique de la cinquième période et le shader/exécuteur actuel ; la méthode FA2 fait référence à la pensée par blocs et de softmax en ligne de [l'article FlashAttention-2](<https://arxiv.org/abs/2307.08691>), sans prétendre avoir atteint une performance de niveau article.

## Essai d'implémentation L1 et enregistrement de retour (2026-10-24)

L1 a été implémenté complètement selon la conception ci-dessus (double arena, cache de set de descripteurs, push par décalage de abiVersion=2, reconstruction incrémentale de l'arena des poids), la chaîne complète 4060 passe, mais l'iGPU AMD expose un problème de comportement du premier dispatch non résolu. Par décision de l'utilisateur, cette implémentation est retournée entièrement à la sauvegarde v1 ; les anciens L1 à L6 sont abrogés, remplacés par l'arbre R de ce document. Enregistrement ci-dessous, pour référence de planification de la prochaine tour.

### Pièges localisés et corrigés

1. La construction à un paramètre `vk::UniqueDescriptorSet(rawHandle)` ne remplit pas le pointeur de périphérique ; le destructeur `vkFreeDescriptorSets(nullptr, set)` déclenche le périphérique invalide du loader et le crash /GS de l'ICD (c0000409, le déchargement de crash WER est dans le vulkan-1.dll système). L'usage correct est de déplacer dans le conteneur après `allocateDescriptorSetsUnique` ; toute écriture construisant un wrapper unique par poignée brute est interdite dans ce projet.
2. Le pool de libération explicite de set de descripteurs doit porter `vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet` (VUID-vkFreeDescriptorSets-descriptorPool-00312, la couche de validation le signale). La combinaison de réutilisation résidente et de libération par invalidation explicite oublie facilement.
3. Le moment d'enregistrement de la copie des poids : si on enregistre une seule fois à la création du command buffer, dans le même lot d'abord upload puis dispatch (l'arena n'est construit qu'à l'intérieur du dispatch), la copie n'est jamais enregistrée, et l'arena reste tout zéro sans aucune erreur. Le point d'enregistrement doit être accroché à chaque dispatch ou déclenché à la demande avant le téléchargement, ou changé en soumission indépendante dès l'upload.

### Non résolu : comportement anormal du premier dispatch AMD (RADV)

Phénomène : le premier dispatch dans le command buffer lit l'arena de poids tout zéro ; le second dispatch de même genre dans le même buffer est parfaitement correct. 4060 sans aucun problème. Le noyau touché n'est pas limité à l'attention (sans rapport avec la mémoire partagée) ; gemm et silu comme premier dispatch du cb sont aussi faux (maxErr respectivement 15.8 et 3.29, 4060 dans le budget).

Hypothèses exclues (toutes reproduites de manière contrôlée, les rejets de lignes de source et de données sont conservés dans l'espace de travail local) :

- Copie non exécutée : après soumission préliminaire indépendante et attente de fence de la copie des poids, la sonde après fence confirme toujours la présence des données de l'arena, et le premier dispatch lit toujours zéro.
- Manquant de barrier de visibilité : le barrier explicite transfer vers compute est inefficace.
- Démarrage à froid du pipeline : la soumission d'échauffement silu est inefficace ; l'échauffement par dispatch attention fait au contraire la sortie erronée devenir des valeurs nulles non correspondantes à aucune ligne d'entrée (cohérent avec la combinaison des données antérieures de l'arena), ce qui montre que la forme de l'erreur est liée à l'historique des dispatch.
- Problème de descripteur et d'objet mémoire : la couche de validation ne signale pas d'erreur, mais cela ne suffit pas à exclure une erreur de cycle de vie, de liaison ou de synchronisation du côté application.

À ce moment on soupçonnait un défaut de RADV sur les constantes push du premier dispatch ou la première lecture du premier buffer lié, mais sans localiser la racine, sans pouvoir affirmer une erreur du pilote à partir de là. Il faut ensuite prendre la manière de liaison invariante v1 comme contrôle, en changeant un à un le pool, les descripteurs, les décalages et l'ordre d'upload ; sans refactorer en même temps puis masquer l'erreur par un dispatch d'échauffement. Cette tour passe sans adopter ce plan d'arena/ABI par décalage ; AMD reste le périphérique à vérifier pour chaque branche activée.

### Opération de retour et validation

Selon la méthode de restauration de BACKUP.md, 19 fichiers sont restaurés octet par octet depuis la sauvegarde (dix .comp, kernels, executor, contrat gpuExecutor, gpuTransformer, deux tests, manifest), suppression du trace.hpp temporaire, suppression et reconstruction de build-gpu-vulkan (zéro avertissement /W4). Acceptation : ctest 29/29 (périphérique AMD 0), sur 4060 vulkanKernels, vulkanExecutor, vulkanShaderAbi et le modèle complet gpuModel passent tous. L'arbre à l'achèvement de ce retour est v1 d'origine, abiVersion retourne à 1 ; l'arbre R de cette tour ajoute des noyaux d'optimisation sur cette base, la version d'ABI reste 1.
