# Backend GPU Vulkan

La cinquième période a livré la bibliothèque SPIR-V de dix opérateurs et le backend modèle résident sur périphérique ; la sixième période a ajouté des noyaux optimisés FP32 sans changer les liaisons v1 ni la disposition push. Vulkan est la première API GPU connectée au backend sélectionné dans le dépôt : avec TIMESTRAL_USE_VULKAN=ON, loadBackendModel renvoie un modèle GPU, les poids sont résidents sur le périphérique et le graphe de couches s'exécute en lot ; le build par défaut ne lie aucun code Vulkan, et l'API publique, la disposition des tenseurs et le choix du backend CPU sont inchangés. Le modèle GPU ne conserve que Vulkan ; les composants expérimentaux OpenGL et OpenCL ont été supprimés à la demande de l'utilisateur ; les manques de migration et les frontières historiques sont dans [l'enregistrement d'arbitrage](<gpuScope.md>).

## Les deux entrées de build

| Option | Effet |
| --- | --- |
| TIMESTRAL_USE_VULKAN | Le backend modèle sélectionné devient le modèle résident Vulkan, exclusif avec Eigen, ytensor et BLAS ; activer deux quelconques en même temps produit une erreur au stade de la configuration |
| TIMESTRAL_BUILD_VULKAN_KERNELS | Construit seulement le composant de validation d'opérateurs indépendants timestralVulkanKernels, sans fournir de backend modèle |
| TIMESTRAL_VULKAN_SPV_DIR | Quand il pointe vers un répertoire complet de 52 modules SPIR-V Vulkan1.1, aucun compilateur shader n'est nécessaire ; le manquant de modules ordinaires/budget ou un en-tête invalide produit une erreur au stade de la configuration |
| TIMESTRAL_VULKAN_TEST_DEVICE | Numéro de périphérique Vulkan énuméré utilisé par les tests d'opérateurs, par défaut 0 |
| TIMESTRAL_VULKAN_TEST_VALIDATION | Les tests d'opérateurs exigent les couches de validation Khronos ; l'absence d'installation produit un échec et non un saut |
| TIMESTRAL_BUILD_GPU_BENCHMARKS | Construit vulkanBench, par défaut avec le mode de transport ; les modes resident/attention chronomètrent l'enregistrement résident et la soumission, par défaut ON |

Arbre de build du backend modèle (exige les en-têtes et le loader du SDK Vulkan) :

~~~powershell
cmake -S . -B build-gpu-vulkan -A x64 -DTIMESTRAL_USE_VULKAN=ON "-DVulkan_INCLUDE_DIR=$env:VULKAN_SDK/Include" "-DVulkan_LIBRARY=$env:VULKAN_SDK/Lib/vulkan-1.lib"
cmake --build build-gpu-vulkan --config Release --parallel
ctest --test-dir build-gpu-vulkan -C Release --output-on-failure
~~~

La configuration d'exécution commune est dans [la configuration d'exécution](<../../timestral/utils/gpuSettings.hpp>) ; le modèle Vulkan fournit en plus un interrupteur de retour optimisé :

| Variable | Par défaut | Description |
| --- | --- | --- |
| TIMESTRAL_GPU_SHADER_DIR | Répertoire des shaders gravé à la construction | Surchage l'emplacement des spv précompilés |
| TIMESTRAL_GPU_DEVICE | 0 | Numéro de périphérique Vulkan énuméré, strictement décimal |
| TIMESTRAL_GPU_VALIDATION | 0 | À 1, exige les couches de validation Khronos ; l'absence d'installation produit un échec |
| TIMESTRAL_VULKAN_OPTIMIZATIONS | 1 | À 1 active l'accélération de projection validée, à 0 utilise la projection générique sans fusion ; une valeur illégale est refusée. Les deux partagent le chemin de précision ordonnée, la relecture en cache et l'orchestration KV-only ; 0 n'est pas un retour vers l'ancien code source |
| TIMESTRAL_VULKAN_RESIDENT_RESIDUAL | 0 | La fusion résiduelle n'est qu'une expérience explicite, M≥128 ; la borne supérieure de M élevé exportée 680M est révoquée, sans stratégie par défaut Intel établie à partir de là ; une valeur illégale est refusée |

L'ordre d'énumération des périphériques ne garantit pas la priorité au GPU dédié ; il peut changer après un changement de pilote ; les tests et les benchmarks affichent tous le nom réel du périphérique. Le périphérique 0 mesuré cette tour est AMD Radeon(TM) Graphics (API1.3.217), le périphérique 1 est NVIDIA GeForce RTX 4060 Laptop GPU (API1.4.325).

## Budget de parallélisme GPU

Le [Settings public](<../../timestral/timestral.hpp>) ajoute gpuThreads : 0 n'importe pas le dispatch de cette bibliothèque, un non-0 doit être un multiple de 256/dans la plage u32, et n'a d'effet que sur Vulkan. Il limite les invocations logiques de chaque fenêtre de dispatch sérielle, ce n'est pas un EU/CU physique, des canaux nominaux 768 ou un pourcentage d'utilisation du GPU ; un budget très petit peut être nettement plus lent. Le local_size fixe ne change pas avec la valeur utilisateur ; un multiple illégal est refusé avant la création de l'objet GPU ; un budget de 256 ne choisit pas un pipeline en option de 512 threads.

~~~cpp
timestral::Timestral::Settings options;
options.gpuThreads = 8192; // budget d'exemple, pas une valeur recommandée pour un GPU donné
timestral::Timestral model("model/timestral-5m", options);
~~~

Les contraintes entières détaillées, la grille normale/la fenêtre rectangulaire, le natif 2D, la ligne de queue exacte, le JIT de premier pipeline et le manquant de validation Intel sont dans [la conception portable](<vulkan-portability.md>). Les autres processus/modèles, les copies de pilote et la bande passante DRAM partagée ne sont pas garantis par ce budget ; on ne peut pas en promettre la réservation de certains EU ni l'absence définitive de device-lost.

## Bibliothèque d'opérateurs et ABI

Le [répertoire source](<../../timestral/vulkan/shaders/>) contient 26 noyaux de calcul et des aides mathématiques précises partagées ; les variantes Budget de chacune sont générées hors ligne ; le paquet complet compte 52 modules. Les liaisons, push et localSize des 21 noyaux d'origine sont inchangés ; les 5 noyaux nouveaux sont le préfixe partagé, l'attention intra-groupe et le chemin de précision ordonnée. Les variantes Budget conservent l'ABI du noyau d'origine correspondant. Le noyau rapide subgroup n'est créé que si size32/64 est signalé et que la capacité est satisfaite, sans forçage sur Intel8/16.

| Opérateur | Calcul | Groupe de travail | push |
| --- | --- | --- | --- |
| [gemm](<../../timestral/vulkan/shaders/gemm.comp>) | A[M,K] fois W[N,K] transposé, biais optionnel | Blocage K partagé 16×16 | 16 octets |
| [rmsNorm](<../../timestral/vulkan/shaders/rmsNorm.comp>) | Réduction de somme de carrés par ligne, (x*inverse)*scale | 64 threads par ligne | 12 octets |
| [softmax](<../../timestral/vulkan/shaders/softmax.comp>) | Réduction de maximum et de somme d'exponentielles par ligne, masque à moins l'infini | 64 threads par ligne | 8 octets |
| [silu](<../../timestral/vulkan/shaders/silu.comp>) | x/(1+exp(-x)) | 64 threads, vérification de queue | 4 octets |
| [prepareQkv](<../../timestral/vulkan/shaders/prepareQkv.comp>) | QKV fusionné réordonné par tête plus RoPE partiel | 64 threads | 20 octets |
| [mergeKv](<../../timestral/vulkan/shaders/mergeKv.comp>) | Concaténation par ligne du K/V de préfixe et du K/V nouveau | 64 threads | 16 octets |
| [elementwise](<../../timestral/vulkan/shaders/elementwise.comp>) | Trois modes : addition résiduelle, porte sigmoid, SwiGLU | 64 threads | 12 octets |
| [attentionScores](<../../timestral/vulkan/shaders/attentionScores.comp>) | Scores QK du masque à deux flux et softmax en ligne | 64 threads | 36 octets |
| [attentionValues](<../../timestral/vulkan/shaders/attentionValues.comp>) | Réduction PV masquée | 64 threads | 36 octets |
| [attentionFused](<../../timestral/vulkan/shaders/attentionFused.comp>) | Scores et PV achevés dans le même workgroup, réduction partagée inter-warp | 64 threads | 36 octets |
| [gemmTile](<../../timestral/vulkan/shaders/gemmTile.comp>) | Tuile de sortie 32×32, blocage de registres 2×2, chargement W fusionné, précis par K | 16×16 threads, 4160 octets partagés | 16 octets |
| [attentionTile](<../../timestral/vulkan/shaders/attentionTile.comp>) | 8 requêtes de l'axe time partageant le K/V, softmax en ligne à 64 clés | 64×8 threads, 39168 octets partagés, retour v1 si le périphérique ne satisfait pas | 36 octets |
| [attentionRegister](<../../timestral/vulkan/shaders/attentionRegister.comp>) | PV en registres headDim64, diagnostic seulement | 64 threads | 36 octets |
| [gemv](<../../timestral/vulkan/shaders/gemv.comp>) | Projection précise par K à M étroit, chargement des poids fusionné par K | 64 threads, 16 colonnes/groupe | 16 octets |
| [attentionCausal](<../../timestral/vulkan/shaders/attentionCausal.comp>) | Recoupe seulement les blocs de clés futures d'encodage prouvés entièrement vides | 64×8, 39168 octets partagés | 36 octets |
| [attentionSubgroup](<../../timestral/vulkan/shaders/attentionSubgroup.comp>) | Attention subgroup avec vérification d'alignement des requêtes et retour de réduction d'origine | 64×8, 39300 octets partagés, Basic/Vote/Arithmetic | 36 octets |
| [attentionDecode](<../../timestral/vulkan/shaders/attentionDecode.comp>) | Long KV H1 tranché par 128 clés, conserve max/dénominateur/PV | 64 threads, 17408 octets partagés | 40 octets |
| [attentionDecodeMerge](<../../timestral/vulkan/shaders/attentionDecodeMerge.comp>) | Fusion stable des tranches, sans moyennage des résultats normalisés locaux | 64 threads | 12 octets |
| [siluGrid](<../../timestral/vulkan/shaders/siluGrid.comp>) | SiLU supportant le déploiement de grille 2D, pour la fusion résiduelle | 64 threads | 4 octets |
| [gemmWide](<../../timestral/vulkan/shaders/gemmWide.comp>) | 64×64 sortie, registres 4×4, diagnostic seulement | 16×16, 8256 octets partagés | 16 octets |
| [attentionPacked](<../../timestral/vulkan/shaders/attentionPacked.comp>) | Candidat quatre scores/quatre PV en registres, diagnostic seulement | 16×8, 37632 octets partagés | 36 octets |

[shaderManifest.json](<../../timestral/vulkan/shaderManifest.json>) est l'ABI lisible par machine avec abiVersion=1 : environnement cible vulkan1.1, entrée main/GLCompute, localSize de chaque shader, numéro de liaison et direction d'accès, nom des membres de la constante push, type et décalage, limite minimale du périphérique et règle de dispatch 2D. Les kernels linéaires de grande grille résidente sont déployés par x=min(G,65535), y=ceil(G/x) ; l'index d'élément est d'abord aplati par y ; le GEMM utilise une grille 2D native. L'ancien noyau de validation silu ne regarde que X ; le chemin résident de grande grille utilise siluGrid, sans transformer l'ancien noyau directement en dispatch 2D.

Le glslc du SDK local 1.3.283 écrit le mot de version SPIR-V 1.3 comme 0x00010300 ; le spirv-dis du même SDK le résout en 1.3 et spirv-val passe. À la construction, AUTO privilégie glslc, et utilise glslangValidator (ou glslang) en cas d'absence ; le premier avec --target-env=vulkan1.1, le second avec -V --target-env vulkan1.1. Le résultat de détection automatique n'est pas réécrit forcé dans l'item de surcharge utilisateur ; si spirv-val est trouvable, le binaire est vérifié dans le même environnement cible ; le processus d'inférence ne compile pas de GLSL.

[Le test de réflexion ABI](<../../tests/vulkanShaderAbiTest.cpp>) analyse le produit de compilation sans dépendre du GPU : le point d'entrée unique et main/GLCompute, LocalSize cohérent avec le manifest, l'ensemble des descriptor set et des liaisons des storage buffer strictement égal, les décalages/types/longueur totale des membres de la structure push cohérents, et refuse quatre sortes de modules corrompus : magie corrompue, troncation à l'intérieur d'instruction, wordCount nul et wordCount hors portée. Il transforme la dérive entre manifest et spv en échec à la construction, sans plus dépendre d'une non-conformité silencieuse à l'exécution.

## Exécuteur résident

[GpuExecutor](<../../timestral/utils/gpuExecutor.hpp>) est le contrat d'exécution indépendant du backend : begin, upload (marquable persistent), allocate, dispatch, download, end. L'implémentation du [Vulkan exécuteur](<../../timestral/vulkan/executor.cpp>) :

- Les poids sont uploadés une fois en persistent à la construction du modèle ; staging utilise un buffer hôte-visible cohérent ; staging et zone de travail du périphérique ont chacun un pool de réutilisation à haut niveau
- Tout le graphe de couches d'un runLayers est enregistré dans un primary command buffer ; un barrier de visibilité est inséré avant chaque dispatch ; la soumission a lieu au premier point de relecture ; les relectures de pages K/V suivantes restent en soumission/attente sérielle, sans relecture centralisée implémentée
- Les poignées du lot deviennent invalides à end ; l'usage inter-lots lève out_of_range, sans réutilisation des adresses scratch du lot précédent

Sémantique des exceptions : submit marque le lot courant comme empoisonné et conserve toutes les ressources en cas d'échec de création de fence, de queue.submit ou d'attente ; end() libère après vidange par device waitIdle ; le begin() suivant refuse explicitement au lieu de réentrer avec une maladie. L'assistant de nettoyage de [GpuTransformer](<../../timestral/utils/gpuTransformer.cpp>) n'appelle end() que quand une exception se propage déjà, et un échec de nettoyage n'écrase pas l'erreur de périphérique d'origine. Cette sémantique corrige un masquage d'erreur découvert lors d'une revue : l'ancien chemin, après que submit lève une erreur, appelait encore end sur un command buffer non terminé, avalant l'erreur d'origine et bloquant les lots suivants.

## Backend modèle

[GpuTransformer](<../../timestral/utils/gpuTransformer.hpp>) hérite de Transformer et réécrit le graphe de couches et les crochets numériques. Le bloc résiduel résident sur le périphérique devient une expérience explicite ; par défaut les trois projections passent encore par le CPU ; le post-traitement, la planification et la gestion de pages restent sur l'hôte. On suit les prédicats à deux flux time/variate ; RoPE est précalculé par l'hôte avec la même libm que reference, en ne réutilisant précisément que les fréquences inverses de cet appel et les lignes de position strictement identiques. Le prefill pur KV s'arrête après que le K/V de la dernière time est prêt, mais le forward complet, trace et decode ne sont pas tronqués. Le K/V de préfixe est toujours assemblé/uploadé depuis le pool de pages CPU, concaténé par mergeKv, sans miroir de KV sur le périphérique.

Matrice de tests :

| Test | Couverture |
| --- | --- |
| gpuModelTiny / gpuModelReal | Mini et 5M réel contre reference indépendant, état caché par couche max(1e-5,1e-4×RMS) plus relL2≤1e-5, quantiles de sortie 1e-4+1e-3×|ref|, cohérence cache et intégral, lignes 1/2/3, isolement de groupe, rôles, clés manquantes, coordonnées continues, pages froides/ajoutées/partagées, hachage de page inchangé |
| vulkanExecutor | G1 douze formes × biais bit à bit ; GEMV dix formes × biais budget figé ; causal bit à bit, subgroup/packed budget figé ; split-K tranche vide/tranche de queue/rôles ; siluGrid dépassant 65535 groupes ; l'ancien test de graine et de cycle de vie de l'attention |
| vulkanKernels | Quatre opérateurs, 30 groupes de formes GEMM, blocs de queue, bords, concurrence, refus de spv corrompu |
| vulkanShaderAbi | Comparaison de réflexion ABI entre manifest et spv plus refus de quatre sortes de corruption |
| time/condition/multivariate Real, sharedDecode | Régression du 5M réel et decode partagé paginé passant par le backend GPU |

Le 8 canaux par défaut n'augmente que le budget de sortie du forward/predictBlock public ; --strict-multi conserve le diagnostic strict par couche, et le choix de noyau v1 échoue aussi actuellement, sans localisation encore. Les bornes inférieure et supérieure du dispatch inter-résiduel H1 de 63/513 patches, le KnownCov de 129 patches, le hachage de page en lecture seule sont passés sur double GPU, sans écrire le passage de la sortie 8 canaux comme passage de toutes les couches 8 canaux.

## Granularité de performance et fenêtre historique

Le déploiement final vise la iGPU Intel ; 680M/4060 ne font que de la validation. Le decode n'est plus validé à 1 % MFU, mais par latence de même tâche, copy et bande passante ; le coût du budget le plus récent et le matériel non couvert sont dans [la conception actuelle](<vulkan-portability.md>). Les données ci-dessous comme 41.47/8.48 ms appartiennent à la stratégie historique qui activait la fenêtre M résiduelle, et ne sont pas les valeurs par défaut actuelles, sans inférence de la latence Intel.

Prefill froid et decode H=1 de 32768 points d'échantillonnage (1024 patches), protocole backendCompare (outil local, non versionné), kind=paged, médiane de 3 encodes, decode échauffé 5 fois et chronométré 25 fois en médiane, machine sans autre build ni tâche GPU :

| Périphérique | encode ms | decode ms | checksum |
| --- | ---: | ---: | --- |
| RTX4060 Laptop (périphérique 1) | 120.8538 | 7.7143 | 2.3805 |
| iGPU AMD (périphérique 0) | 236.9136 | 11.8549 | 2.3806 |

Re-mesure CPU historique de la même tour (voir les notes d'acceptation locales) : blas-avx2 181.15/4.39 ms, reference 4769.09/8.16 ms, checksum tous 2.3806. Cela ne montre que la performance bout en bout de cette fenêtre, et ne prouve pas une utilisation GPU suffisante. L'ancien texte appliquait la formule du forward complet au prefill paginé, et discutait l'efficacité GPU avec un dénominateur de 134.4 GF/s CPU ; cette granularité est désormais retirée ; le jugement « H=1 dominé par la soumission et l'orchestration » sans attribution est aussi retiré. Le 680M est recalculé selon les 3 TFLOPS donnés par l'utilisateur et la portée d'exécution réelle ; l'analyse est dans [l'attribution dédiée 680M](<vulkan-opt.md>) ; le MFU nominal CPU sert toujours le CPU seulement.

Le tableau ci-dessus est la base historique de la cinquième période, et ne représente pas la version optimisée actuelle. Le rotary et les metadata du même appel sont uploadés une seule fois hors de la boucle de couches ; le jugement de l'ancien texte « précalcul/upload par couche » est corrigé. Actuellement il y a encore un pool de descripteurs indépendant par dispatch, une relecture de pages sérielle, et le pool scratch ne conserve que le haut niveau sans budget.

Après la reconstruction de la sixième période, on accepte la projection G1 et l'attention de K/V partagé A2 ; v1→version optimisée de la même fenêtre finale : 4060 encode 125.9646→109.8991 ms, decode 8.0188→6.7479 ms ; AMD encode 263.0007→143.5030 ms, decode 12.9188→10.5634 ms, les valeurs de sortie initiales restent 2.3805/2.3806. AMD a des variations plus grandes entre les tours ; toutes les tours, les micro-benchmarks d'opérateurs, la réutilisation de pool non acceptée et la coupe de blocs vides sont dans [l'arbre d'optimisation](<vulkan-opt.md>), sans promesse de facteur fixe.

Le choix de noyau actuel se concentre sur Vulkan Executor, l'orchestration n'ajoute que l'arrêt KV-only validé et la fusion résiduelle de la fenêtre M. G1 couvre les formes d'embedding mesurées ; M1 à 8 peut utiliser le GEMV par K ; la longue attention time choisit causal/subgroup selon la capacité ; le long KV H1 peut faire du split-K. FP16, FMA de projection et réduction K parallèle de projection ne sont pas activés ; l'arenne n'est pas rétablie ; la réordonnancement de la réduction d'attention est validé selon le budget numérique d'origine.

## Correction de relecture et diagnostic 680M

Les horodatages 680M dédiés et le chronométrage CPU ont révélé : l'ancienne relecture utilisait une mémoire Visible+Coherent mais non Cached ; le map/lecture/unmap CPU de 17.25 Mio prenait environ 77.53 ms, alors que le copy d'exécution GPU prenait environ 2 ms. La relecture actuelle a un pool de réutilisation indépendant privilégiant HOST_CACHED|HOST_COHERENT, et conserve le retour cohérent existant sans ce type ; le staging d'upload est inchangé, sans suppression de fence ni de barrier de visibilité. Après correction, la lecture CPU correspondante est environ 1.52 ms.

Ci-dessus est l'attribution historique au moment de la correction cached ; la fenêtre ancienne encode 85 à 88 ms, 12.260475 G selon la figure avant recoupage. Le MFU actuel fixe la quantité de tâche nécessaire 10.648682496 G, sans passer les calculs inutiles de queue ultérieurement supprimés pour des produits. Les chiffres les plus récents, les variations par tour et les items non atteints sont dans [l'arbre d'optimisation](<vulkan-opt.md>).

L'outil d'attribution vulkanProfile (outil local, non versionné) n'est construit que si TIMESTRAL_USE_VULKAN et les benchmarks GPU sont activés, sans changer le comportement par défaut du modèle. Usage :

~~~powershell
.\build-gpu-vulkan\timestral\vulkan\Release\vulkanProfile.exe build-gpu-vulkan/timestral/vulkan/shaders model/timestral-5m 0 1 1 0 8192
~~~

Les cinq derniers paramètres sont le numéro de périphérique, diagnostic 0/1, optimisation 0/1, résiduel expérimental 0/1, budget GPU, les deux derniers par défaut 0. L'outil rapporte l'API hôte, l'étendue GPU, le copy, le nombre réel de fenêtres et les invocations maximales de la fenêtre, en supprimant le 3 TFLOPS/MFU codé en dur ; le débit est décrit en GF/s de tâche nécessaire. Le decode diagnostique n'a que 5 fois ; l'attente CPU et le temps GPU ne s'additionnent pas ; la latence utilise un protocole figé avec diagnostic désactivé. Le miroir de KV n'est toujours pas implémenté.

## Limites connues

- Chaque dispatch lie un buffer à un SSBO unique, contraint par maxStorageBufferRange ; le chemin 32k de timestral-5m est bien en dessous de la borne typique 128 Mio, mais un modèle plus grand ou un lot plus grand peut atteindre le plafond
- Le pool de réutilisation scratch ne conserve que selon le haut niveau, sans récupération ni budget mémoire
- La fusion résiduelle est désactivée par défaut, expérience explicite seulement, sans stratégie par défaut Intel décidée selon la borne M2048 du 680M
- FP32 seulement, sans exiger de FP16/cooperative matrix ; subgroup est une optimisation optionnelle avec retour en cas de capacité manquante
- Les opérateurs de forme petite sont plus lents que le CPU avec le transport, voir la section précédente ; la soumission de lot résidente a seulement éliminé la synchronisation redondante entre couches, sans éliminer le coût d'allocation et de soumission au sein d'un dispatch
## Frontière SPIR-V inter-API

Cette bibliothèque ne publie que le Shader SPIR-V ciblant Vulkan1.1, sans promettre sa consommation par d'autres API. L'expérience historique OpenGL manquait d'attentionFused et d'un contexte inter-plateformes ; l'expérience OpenCL n'avait que quatre opérateurs mesurés, sans validation du modèle résident, la version de IL du périphérique était vide et la file AMD avait échoué ; on ne peut pas migrer en changeant l'extension. Les deux ensembles de composants sont désormais supprimés, sans conserver d'entrée d'activation, voir [l'arbitrage du backend GPU](<gpuScope.md>).

## Publication binaire

~~~powershell
cmake --install build-gpu-vulkan --config Release --component VulkanShaders --prefix .agents/data/vulkan-package
~~~

Le composant d'installation place les 52 spv des 26 ordinaires et des 26 Budget, [shaderManifest.json](<../../timestral/vulkan/shaderManifest.json>) et [gpuKernels.md](<../../timestral/utils/gpuKernels.md>) dans share/timestral/vulkan. Il suffit de pointer TIMESTRAL_VULKAN_SPV_DIR vers ce répertoire pour construire hors ligne, sans chercher de glslc ni de glslangValidator. Le choix explicite de l'outil peut définir le chemin TIMESTRAL_VULKAN_SHADER_COMPILER et TIMESTRAL_VULKAN_SHADER_COMPILER_KIND=AUTO/GLSLC/GLSLANGVALIDATOR. L'entrée précompilée doit être un fichier de confiance, validé et dont la disposition des descripteurs et de push est conforme ; l'exécution ne fait qu'une vérification de base de l'en-tête SPIR-V, sans se charger de prouver la sécurité de n'importe quel shader externe.
