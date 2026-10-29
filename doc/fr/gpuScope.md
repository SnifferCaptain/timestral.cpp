# Périmètre du backend GPU

## Décision actuelle

Le backend GPU modèle complet actuel est Vulkan uniquement. OpenGL et OpenCL sont tous deux ❌, supprimés à la demande de l'utilisateur, et ne sont plus des composants optionnels ni des promesses de migration future. Les expériences passées d'opérateurs ne valent pas un backend complet livré ; les observations historiques sont conservées dans ce document et ne sont pas réécrites comme un support actuel.

## Pourquoi ce n'est pas une petite migration

| Élément | Ancien composant OpenGL | Ancien composant OpenCL |
| --- | --- | --- |
| Limite mesurée | Windows WGL, neuf noyaux de base et chaîne résidente partielle | Quatre opérateurs OpenCL C 1.2 indépendants, exécuteur résident sauvegardé non accepté |
| Manque actuel du modèle partagé | attentionFused manquant, contrat de dispatch incomplet avec le modèle existant | attentionFused et noyaux résidents suivants manquants, usine sauvegardée non branchée sur la compilation |
| Environnement d'exécution | Cible OpenGL4.5 / SPIR-V 1.0, liaison UBO/SSBO et contexte, pas un binaire cible Vulkan | L'environnement d'exécution des kernels diffère de celui des shaders Vulkan, impossible de réutiliser directement le SPIR-V Vulkan généré par GLSL |
| Faits pilotes | Seul le chemin WGL AMD a des expériences anciennes, aucune livraison Linux/EGL ni d'acceptation du modèle 5M complet | Les deux appareils avaient des chaînes de version IL vides à l'époque ; la création de file AMD a échoué, impossible de revendiquer la portabilité juste parce que le loader a des points d'entrée IL |
| Travail encore nécessaire | Noyaux critiques nouveaux, contexte plateforme, adaptation ressources/synchronisation et acceptation numérique et performance complète du modèle/cache/quantiles | Noyaux critiques nouveaux, chemin compilation/IL, problèmes ressources/synchronisation/pilotes et acceptation complète équivalente |

Le Transformer partagé et une partie des maths d'opérateurs peuvent être réutilisés, mais changer simplement les noms d'API ou un ensemble de paramètres de compilation de shaders ne donne pas un backend complet. Revenir à des opérateurs d'attention séparés pour éviter le manque, repli CPU silencieux ou revendiquer le support du modèle complet après n'avoir exécuté que quatre opérateurs ne sont pas des options de livraison.

## Périmètre supprimé

- L'intégralité des host, loaders, exécuteurs, usines sauvegardées, shaders/noyaux intégrés, docs de répertoire et CMake des deux répertoires de composants expérimentaux
- Deux tests d'opérateurs dédiés, sans toucher aux tests d'autres backends ni aux budgets gelés
- Les deux entrées de composants optionnels dans la compilation racine et l'entrée de l'exécuteur expérimental OpenCL
- La tâche de compilation du composant statique OpenCL dans le workflow Linux
- Les liens et instructions d'activation dans la documentation officielle pointant vers les composants supprimés

Aucun répertoire formel distinct de « backend expérimental » n'a été créé, et aucune usine vide ni fausse implémentation n'est conservée. Les anciens caches de compilation qui activent explicitement les options supprimées échouent à la configuration avec un message d'utiliser Vulkan, plutôt qu'ignorer silencieusement l'ancienne option et laisser l'utilisateur croire que cette API est toujours sélectionnée.

Vulkan et les contrats GPU partagés, les quatre backends CPU et les données de modèle/mesure d'origine ne sont pas supprimés. Les graphiques de performances ne sont pas remplis à nouveau en raison des choix de périmètre des composants. La matrice de support actuelle est dans le [README principal](<../../README_fr.md#plates-formes-et-backends>) ; les échecs numériques de stress connus pour les GPU existants restent dans les enregistrements numériques locaux.
