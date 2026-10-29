# Documentation de Conception

[中文](<../zh/README.md>) | [English](<../en/README.md>)

Ce répertoire explique la conception de la bibliothèque d'inférence Timestral C++ dans l'ordre « ce que le modèle calcule, comment l'inférence économise du travail, comment les backends calculent ». L'utilisation et la compilation sont dans le [README du projet](<../../README_fr.md>), les paramètres d'exemples et les instructions d'exécution dans le [guide d'exemples](<../../example/README.md>), et les preuves numériques dans les enregistrements de validation locaux (non commités).

## Historique des mises à jour

L'[historique par version](<changelog.md>) résume les changements par version. Ajouter les nouvelles versions à la fin.

## Ordre de Lecture Suggéré

1. [Architecture du modèle et modules du dépôt](<architecture.md>) : introduit l'incorporation de patch, la normalisation causale, les axes d'attention time et variate, les masques, RoPE partiel, l'attention avec porte, SwiGLU et la tête de quantiles en utilisant les dimensions réelles de Timestral-5M, puis explique les limites des modules du dépôt et la sélection du backend à la liaison.
2. [Conception du côté inférence](<inference.md>) : comment une requête passe par l'ordonnancement, l'auto-régression par blocs et le post-traitement ; pourquoi la mise en cache KV paginée fonctionne sur le chemin standard mais pas sur le chemin conditionnel ; et l'expérience de fenêtre glissante à préfixe gelé au niveau des exemples.
3. [Backends de calcul](<backends.md>) : approche d'implémentation de reference, Eigen et ytensor, critères de tolérance pour l'acceptation de la précision, et compromis de performance mesurés.
4. [Backend BLAS](<blas.md>) : l'arbre d'optimisation du niveau requête au micro-noyau, dérivant le pré-empaquetage, le blocage cache, le softmax en ligne, les masques par bloc et exp vectoriel 2^x, avec les critères de performance et les expériences échouées.

Les concepts des documents antérieurs sont utilisés directement dans les documents suivants ; par exemple, les conditions de cache du document d'inférence dépendent des règles de masque du document d'architecture. Si vous ne vous intéressez qu'à un backend, lisez d'abord la section modules du document d'architecture, puis passez directement au document correspondant.

Pour les paramètres d'interface, valeurs de retour, exceptions et politiques d'ordonnancement, voir la [Référence API](<api.md>) directement.

## Constructions Multiplateformes

[Corrections de compatibilité Linux et limites de validation](<linuxCompatibility.md>) enregistre les corrections point par point des en-têtes publics, des dépendances de threads, des gardes C++20/ISA locaux de ytensor, des chemins UTF-8 et des double compilateurs Vulkan, ainsi que les résultats locaux et la CI Linux en attente. Les échecs numériques connus ne sont pas ignorés parce qu'une compilation passe.

## Jalonnes de Développement GPU

[Backend GPU Vulkan](<vulkan.md>) couvre la bibliothèque SPIR-V à dix opérateurs, le manifeste ABI et les tests de réflexion, l'exécuteur résident, le backend modèle et les performances mesurées à 32k. Le backend GPU actuel est Vulkan uniquement ; voir [Périmètre du backend GPU](<gpuScope.md>) pour la raison de la suppression des anciens composants OpenGL/OpenCL.

[Portabilité Vulkan et budget de parallélisme](<vulkan-portability.md>) enregistre la cible iGPU Intel, la revue partielle locale de ggml, la configuration de la fenêtre logique GPU, le contrat de dispatch 52 modules, le coût du budget strict et la validation matériel Intel en attente. Le décodage est accepté sur latence/debit, pas sur le dénominateur 3 TFLOPS de l'appareil de développement.

## Reproduction des Performances et de la Qualité

[Protocole de référence et données brutes](<benchmark.md>) enregistre 16 recettes et les résultats de pré-remplissage par nombre de variables. Le README principal ne conserve que le tableau à variable unique, le graphique de vitesse et un ensemble RMSE ; les tableaux multivariables et la vérification de la étape par défaut AR sont concentrés ici. Les données ont été mesurées dans des conditions autorisant une légère charge quotidienne sans charge externe lourde ; l'étendue des échecs des cas étendus est listée séparément.

## Figures

Les figures sont dans le répertoire [assets](<../../assets>) à la racine du dépôt. Les SVG avec les préfixes arch-, infer- et backend- correspondent aux trois premiers documents ; les SVG avec le préfixe blas- correspondent au tutoriel BLAS ; les graphiques de balayage SWA et les CSV bruts proviennent de l'expérience de fenêtre glissante.
