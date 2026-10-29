# Corrections de compatibilité Linux et frontières de vérification

Les comptes de tests de ce document enregistrent les corrections de compatibilité de l'époque. L'utilisateur a ensuite décidé de supprimer les composants OpenGL/OpenCL ; le code, la construction et les tâches CI associés sont désormais supprimés ; l'état actuel des plateformes suit [la matrice de la notice principale](<../../README_fr.md#plates-formes-et-backends>), et l'arbitrage est dans [Portée du backend GPU](<gpuScope.md>).

Cette bouée corrige point par point selon le rapport d'audit les problèmes de construction confirmables. Le déploiement par défaut reste du C++17, sans ajout de dépendance tierce d'exécution, sans modification de l'algorithme de prédiction, des poids, du pas de décodage par défaut ni du budget d'erreur figé. La machine actuelle est Windows/MSVC, sans distribution Linux, GCC ou Clang disponibles ; on ne peut pas écrire le résultat local du tableau ci-dessous comme un passage mesuré sur Linux.

## Tableau des corrections

| Problème signalé | Traitement actuel | Entrée de preuve |
| --- | --- | --- |
| Paramètres par défaut de Settings imbriqués bloquant GCC/Clang | Suppression des arguments par défaut dans la classe, ajout d'une construction par délégation à un paramètre et d'une surcharge de loadEngine ; Settings reste agrégé, sans constructeur d'utilisateur | [Déclaration publique](<../../timestral/timestral.hpp>), [chargement par défaut](<../../timestral/utils/engine.cpp>) |
| Threads apparaissant seulement dans les branches test/BLAS | La configuration racine découvre Threads inconditionnellement, timestral et les noyaux Vulkan lient chacun Threads::Threads en PUBLIC ; l'ancien composant OpenCL est désormais supprimé | [Configuration racine](<../../CMakeLists.txt>), [test descendant indépendant](<../../tests/embedding/CMakeLists.txt>) |
| Les COMPILE_FEATURES des fichiers source ytensor n'upgrade pas réellement la norme | Utilisation d'un but OBJECT C++20 indépendant, le cœur et la fabrique continuent en C++17 ; la CI affirme les compile_commands réels | [Configuration de build](<../../CMakeLists.txt>), [CI Linux](<../../.github/workflows/linux.yml>) |
| L'AVX2/FMA de l'unité entière ytensor donne des instructions illégales sur l'ancien CPU | Seuls les objets du modèle activent AVX2/FMA ; la fabrique de référence vérifie d'abord CPUID, OSXSAVE et l'état XMM/YMM, et refuse explicitement en cas de non-conformité ; IPO/unity désactivé des deux côtés pour éviter que l'optimisation ne traverse la frontière de sécurité | [Vérification CPU](<../../timestral/utils/cpu.cpp>), [fabrique](<../../timestral/ytensor/backend.cpp>) |
| Vulkan ne cherchant que glslc | AUTO privilégie glslc, et utilise glslangValidator/glslang si absent ; prise en charge de la surcharge explicite du chemin et de l'espèce ; les deux ensembles de commandes visent Vulkan1.1 | [Choix du compilateur](<../../timestral/vulkan/shaderCompiler.cmake>) |
| Pollution de cache du compilateur shader NOTFOUND | Les items de surcharge utilisateur et le résultat de détection automatique sont séparés ; le résultat automatique n'est pas réécrit forcé dans l'item utilisateur ; les tests couvrent le manquant, l'apparition ultérieure et le forçage de changement d'espèce | [Test de choix répétable](<../../tests/shaderCompilerTest.cmake>) |
| SPIR-V hors ligne incomplet/non conforme | Vérification un par un de 26 modules ordinaires + 26 modules de budget, vérification de la magie SPIR-V, validation à la configuration si spirv-val existe ; un répertoire hors ligne complet ne cherche pas de compilateur | [Build Vulkan](<../../timestral/vulkan/CMakeLists.txt>) |
| Échec de concaténation de chemin char8_t C++20 | Extraction unifiée des bytes UTF-8, sans rétrécissement par code page locale ; l'implémentation et le fixture de test partagent la conversion ; le test JSON UTF-8 brut ne traite plus char8_t comme char | [Conversion de chemin](<../../timestral/utils/path.hpp>), [régression double norme](<../../tests/portabilityTest.cpp>) |
| Avertissement de string temporaire de boucle de portée de config.cpp | L'itération de littéraux par const char*, std::string explicite à la concaténation d'erreurs, le sens du message est inchangé | [Analyse de configuration](<../../timestral/utils/config.cpp>) |
| Frontière de plateforme OpenGL/OpenCL floue | L'évaluation ultérieure ne fait pas une migration complète à faible coût ; selon la décision de l'utilisateur, on supprime le code expérimental et les entrées de build/CI, sans le conserver comme backend | [Fondement de l'arbitrage](<gpuScope.md>) |

Le choix de ytensor exige toujours x86 AVX2+FMA et le support de l'OS. Refus à la configuration sur non x86 ; l'ancien x86 lève une exception explicable avant d'entrer dans le modèle optimisé, en suggérant à l'utilisateur de choisir reference ou Eigen ; pas de retour de backend qui passe silencieusement pour ytensor. Le dispatch multi-ISA d'exécution existant de BLAS est inchangé.

## Vérification réelle sur la machine locale

| Configuration/test | Résultat et portée |
| --- | --- |
| Ensemble reference | 29/29 passés |
| Ensemble BLAS | 39/39 passés |
| Ensemble Eigen | 30/30 passés |
| Build ytensor et nouveaux cas de compatibilité | Build passé ; les tests en-têtes C++17/C++20, initialisation agrégée, chemin UTF-8, bits de capacité CPU et choix shader passés |
| Ensemble ytensor | 30/31, parallelDecodeReal a toujours un échec de budget numérique figé, sans écriture tout vert |
| Tout le cœur construit en C++20 | Build passé ; les régressions ciblées double norme des en-têtes publics, JSON et safetensors passées |
| Consommateur indépendant avec tests/examples désactivés | Lié seulement timestral, vérifie la dépendance transitive de threads, compilation/lien/exécution passés |
| glslc | Génération de 52 modules et spirv-val, 52 items d'ABI shader et vrai dispatch des quatre petits opérateurs passés |
| glslangValidator | Les mêmes 52 modules, validation/ABI et vrai dispatch à faible charge passés |
| SPIR-V hors ligne | Le répertoire complet de 52 modules construit et passe l'ABI même en fournissant sciemment un chemin de compilateur invalide ; le manquant de modules ordinaires/budget est refusé à la configuration |
| Contrat AUTO/cache | Sans équipement Vulkan requis ; le retour manquant, la restauration NOTFOUND, le changement d'espèce explicite et les paramètres illégaux passés, exécutable de manière répétable |

Le vrai dispatch Vulkan ici provient du AMD Radeon(TM) Graphics du périphérique 0 de la machine locale, et non un résultat Linux llvmpipe ou iGPU Intel. Seul 2×2 linear/rmsNorm, 3 éléments softmax/silu sont appelés ; aucun test de pression GPU ni benchmark de modèle complet n'a été exécuté.

### Échec numérique que ytensor conserve encore

L'endroit de l'échec est l'index 13712 de single-frame/cached-full : actual=2.294824, expected=2.294836, limite d'origine environ 0.000011. Ce budget n'a pas été desserré, aucune assertion supprimée ni cas exclu de CTest. Pour distinguer l'isolement de build du problème numérique existant, on a cette bouée seulement dans une copie d'archivage temporaire rétabli l'objet de modèle ytensor d'origine sauvegardé le 25 octobre à la place du nouvel objet de modèle ; la même régression échoue toujours au même index, avec les mêmes actual/expected/limit. La bibliothèque d'origine n'a pas été remplacée, et le backend officiel n'a pas été changé en Naive/reference.

Par conséquent, cette bouée peut montrer que le blocage de build du rapport a été traité, mais ne peut pas montrer que tous les scénarios numériques réels de ytensor passent. Les portées non passées antérieures d'autres longues prédictions, de formes GPU en plus et de périphérique 1 ne disparaissent pas non plus par cette correction ; voir les notes d'acceptation numériques locales.

## Régression à exécuter sur Linux

[Le workflow Linux](<../../.github/workflows/linux.yml>) fixe désormais GCC11 et Clang14 sur Ubuntu22.04, couvrant reference, tout le cœur C++20, BLAS, Eigen, ytensor ; les commandes réelles de ytensor doivent ne faire apparaître -std=c++20/-mavx2/-mfma que dans l'unité modèle, et la fabrique de référence/vérification CPU ne doit pas porter AVX2. En plus :

- Le processus QEMU Nehalem exécute réellement [le test de protection sans AVX2](<../../tests/ytensorGuardTest.cpp>), et ne se contente pas de simuler les bits CPUID
- Compilation/lien du consommateur indépendant avec tests désactivés, sans dépendre des buts de test pour ajouter pthread
- Le build croisé RISC-V doit réellement sélectionner Eigen, et supprimer les anciennes commandes masquant les échecs
- glslc/glslangValidator construisent chacun 52 modules et vérifient par spirv-val, utilisant llvmpipe ICD logicielle pour exécuter seulement l'ABI et un petit smoke
- Le répertoire hors ligne complet ne dépend pas du compilateur ; le répertoire incomplet doit échouer
- La tâche de build du composant statique OpenCL d'origine est supprimée avec le composant, plus inscrite à la CI ni au plan de support

La syntaxe du workflow a été analysée et vérifiée sur la machine locale, mais pas encore déclenchée sur la CI distante ; on n'a pas remplacé le résultat Linux par « CI ajoutée ».

~~~bash
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-11
cmake --build build-linux --parallel 2
ctest --test-dir build-linux --output-on-failure --parallel 2

cmake -S . -B build-vk -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_VULKAN_KERNELS=ON -DTIMESTRAL_VULKAN_SHADER_COMPILER_KIND=GLSLANGVALIDATOR
cmake --build build-vk --target vulkanShaderAbiTest vulkanSmokeTest --parallel 2
ctest --test-dir build-vk -R '^(vulkanShaderAbi|vulkanSmoke)$' --output-on-failure
~~~

Le smoke et la bibliothèque statique doivent être construits par la même configuration CMake. Certains types de Vulkan-Hpp sont affectés par NDEBUG ; on ne doit pas mélanger les en-têtes/objets Debug et la bibliothèque statique Release ; la correction actuelle ne promet pas cette compatibilité d'ABI inter-configurations, et n'utilise pas de wrapper manuellement non conforme pour vérifier le GPU.

La performance n'a pas été re-mesurée cette bouée. Les figures et CSV existants conservent les enregistrements de programme/environnement au moment de la mesure d'origine ; on ne peut pas réinjecter le SHA du binaire ou la latence après correction de compatibilité dans les résultats anciens.
