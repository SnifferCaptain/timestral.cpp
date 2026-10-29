# Optimisation portable Vulkan et budget parallèle GPU

## Objectifs et limites

Le déploiement final vise les GPU intégrés Intel des 13e à 15e générations, de plus petite taille, décrits par l'utilisateur ; 680M et 4060 ne servent qu'à la validation en développement et leurs meilleurs résultats ne déterminent pas la stratégie par défaut pour Intel. EU, Xe-core, CU AMD et canaux d'arithmétique ne sont pas la même unité ; un appareil qui nomme 768 canaux d'arithmétique ne signifie pas 768 threads Vulkan allocables. La compatibilité de version Vulkan n'implique pas non plus des extensions subgroup, workgroup, SLM, cache ou matrices optionnelles identiques.

Le decode abandonne l'acceptation par 1 % de MFU et se tourne vers la latence en régime établi de la même tâche, le copy explicite, l'estimation du trafic de données logique et la bande passante. CPU/GPU, même en partageant la DRAM, ont des caches et des chemins de réplication différents, de sorte qu'une égalité de latence ne peut pas être promise sur cette base. L'outil d'attribution a retiré les 3TFLOPS et le pourcentage de MFU codés en dur ; il conserve les GFLOPs/s de la tâche requise comme description du débit, pas comme seuil de tâche mémoire.

## Références locales vérifiées

La snapshot ggml fournie par l'utilisateur est directement lisible ; la snapshot et les fichiers de contrôle appartiennent aux fichiers de travail locaux et ne sont pas publiés avec le dépôt. Aucun commit distant n'est fixé, aucun noyau entier n'a été copié et aucune revue complète du dépôt n'est revendiquée. Le fichier de contrôle enregistre le SHA256 complet des fichiers pertinents ; le périmètre de revue effectif est le suivant :

| Chemin | Portion lue et informations utiles | Traitement dans ce projet |
| --- | --- | --- |
| `mul_mm.comp` | Définitions de blocage/alignement et boucle principale, BM/BN/BK, tuile warp, chargement partagé et accumulation en registres | Évaluer ensuite des panneaux FP32, des chargements vec4 alignés et la réutilisation ; ne pas importer directement les exigences FP16/cooperative |
| `mul_mat_vec_base.glsl`, `mul_mat_vec.comp` | Définitions de section initiale et structure de réduction, parallélisme ligne/colonne, accès vectoriel, chemins subgroup/partagé | La réduction K-parallèle doit toujours passer le budget gelé ; ne pas relâcher l'erreur pour la bande passante |
| `flash_attn_decode_phase_1.comp` | Attention en phases et réduction locale de la section initiale ; ce fichier n'est pas entièrement relu | Emprunter l'idée de découpage/merge, sans copier l'extension matrices ni un agencement de lanes spécifique |
| `ggml-vulkan.cpp:1468–1478` | La surcharge partagée inclut chargements, index de ligne, stage coopmat et LUT, filtrés selon la limite de l'appareil | Regarder seulement la taille des données de tuile ne suffit pas ; ne pas élargir les tuiles à l'aveugle |
| `Paramètres de decode Xe` | Paramètre 8 pour la branche Xe1, 16 pour les autres chemins Xe, taille de groupe bornée par q_len de 64 à 256 | Les noyaux de sous-groupe rapides 32/64 de cette bibliothèque ne sont plus créés quand l'appareil signale 8/16 ; la réduction générique est conservée ; la largeur de lane n'est pas devinée d'après le code CPU |
| `Création UMA`, `buffer hôte mis en cache` | L'UMA préfère le device-local mappable, l'emprunt est contraint par l'allocation réelle ; le cache hôte a un fallback | Ne pas passer une simple pointeur std::vector pour un buffer GPU ; ne pas traiter HostCached comme un taux de hit L1 GPU |

La recherche en ligne a trouvé le [guide Intel SLM](<https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/shared-local-memory.html>), la [question de capacité AMDVLK liée à l'AI Max395](<https://github.com/GPUOpen-Drivers/AMDVLK/issues/420>) et la [branche d'inférence Vulkan RDNA](<https://github.com/SelfRef/llama.cpp-rdna/blob/rdna/README.md>). web_fetch renvoie toujours un refus pour IP non publique pour ces domaines ; il n'existe actuellement que des résumés de recherche, et leurs implémentations ou chiffres de performance spécifiques ne sont pas traités comme des faits vérifiés. L'optimisation spécifique RDNA n'est pas le défaut pour Intel ; ce qui est réellement vérifié est le code local ci-dessus.

## Utilisation de gpuThreads

La configuration est ajoutée à la fin des [Settings publics](<../../timestral/timestral.hpp>), lue à la construction, ne prend effet que sur Vulkan, est ignorée par les autres backend et ne modifie pas le comportement du threads CPU. Après l'extension de la structure de configuration, l'application et la bibliothèque doivent être re-compilées ; les poids et l'agencement Tensor sont inchangés.

~~~cpp
timestral::Timestral::Settings options;
options.threads = 1;
options.gpuThreads = 8192; // exemple de borne supérieure, pas une recommandation appareil

timestral::Timestral model("model/timestral-5m", options);
~~~

- 0 : ne pas limiter le dispatch original de cette bibliothèque, valeur par défaut
- Non nul : multiple positif de 256 et dans la plage uint32 ; 256/512/768/4096/8192 sont tous légaux, 384 et 257 sont illégaux ; ce n'est pas « puissances de deux uniquement »
- 256 est la quantification d'implémentation du workgroup GEMM actuellement requis, pas une largeur de warp commune à tout le matériel Vulkan
- Chaque kernel conserve son local_size vérifié ; la valeur utilisateur n'est pas écrite dans local_size ; un budget 256 ne crée pas de pipeline d'attention à 512 threads et prend automatiquement le chemin original légal
- Un budget illégal lève invalid_argument avant que l'exécuteur ne crée l'instance/l'appareil Vulkan ; le LocalSize fixe, les invocations par axe/total de l'appareil et les tailles de variante sont aussi validées, évitant un pas nul ou des modules mélangés

**C'est une borne supérieure d'invocations de shader logique pour une fenêtre de dispatch sérial unique, pas un quota physique EU/CU.** Un budget 768 autorise au plus trois workgroups pour un kernel à 256 threads et au plus un pour un kernel à 512 threads ; un demi-workgroup n'est pas dispatché. Les 768 canaux nominaux de l'appareil ne peuvent pas être utilisés pour expliquer 512 comme une occupation de 66 % ; il peut y avoir bien plus de threads logiques en vol que de canaux d'arithmétique. Un très petit budget abaisse gravement le parallélisme et augmente la surcharge de commandes.

La limite ne s'applique qu'aux compute shaders émis par l'exécuteur de ce modèle ; elle ne limite pas les autres processus/modèles, les threads CPU du pilote, le copy/migration interne du pilote, la bande passante DRAM partagée, et ne promet ni un pourcentage d'utilisation ni de consommation. Quand une réservation d'EU dure ou un duty cycle fixe est nécessaire, cette option ne peut pas être traitée comme un contrôle matériel global implémenté. Un paramètre légal ne constitue pas non plus une garantie qu'aucun pilote ne fera jamais TDR/device-lost.

## Planification, indexation et ABI

L'[exécuteur](<../../timestral/vulkan/executor.cpp>) utilise le dispatchBase Vulkan1.1, et le pipeline porte le drapeau DISPATCH_BASE. Il existe des dépendances d'exécution compute/transfer entre les fenêtres, évitant le chevauchement entre la fenêtre suivante de cette bibliothèque et la précédente ; les lignes complètes peuvent être fusionnées en rectangle dans le budget, sans sérialisation inconditionnelle ligne par ligne. windowX × windowY × localInvocations fixes est toujours ≤ gpuThreads.

Le groupCount ne peut pas être simplement réduit : les kernels aplatis dépendent de gl_NumWorkGroups.x et calculeraient une mauvaise adresse pour Y non nul. Le [générateur hors ligne](<../../timestral/vulkan/budgetShader.cmake>) génère 26 modules Budget à partir des 26 kernels compute, en conservant le SSBO/push correspondant et l'ordre des opérations flottantes, et fournit la largeur aplatie via la constante de spécialisation 255. Les 21 GLSL d'origine et leur ABI ne sont pas modifiés ; de nouveaux noyaux de préfixe partagé, d'attention intra-groupe et de précision ordonnée sont ajoutés, toujours abiVersion=1. Le [manifeste complet](<../../timestral/vulkan/shaderManifest.json>) compte 52 modules au total ; le répertoire pré-compilation/édition doit être entièrement mis à jour.

Les kernels aplatis, selon le contrat existant, n'utilisent NumWorkgroups que comme pas aplati, identifié à l'exécution par le marqueur Builtin du module d'origine. La grille de budget est normalisée en W fixe = min(limite de groupes X de l'appareil, 65535) ; la dernière ligne ne dispatche que la couverture d'origine et un dispatch partiel ne s'étend pas ; les GEMM/GEMV/attention scindés natifs conservent les coordonnées 2D d'origine. Chaque opérateur met en cache un seul pipeline de budget ; aucune spécialisation/JIT n'accumule avec la longueur de la requête, et le pipeline n'est pas libéré pendant l'enregistrement ou tant que les commandes en vol le référencent. La première utilisation peut encore engendrer une surcharge de JIT de pipeline ; la médiane du protocole fixe ne représente pas la première requête à froid.

## Politique par défaut et prochaines étapes

La borne supérieure de M 2048 pour le residual de la round précédente, resserrée sur 680M, est retirée de la justification par défaut. La valeur par défaut de TIMESTRAL_VULKAN_RESIDENT_RESIDUAL passe à 0 ; un 1 explicite n'est qu'une expérience, conservant la condition d'amortissement de petit dispatch M≥128, sans traiter les résultats grand-M de 680M comme une règle Intel. Les autres agencements de modèle par dispatch et la légalité des capacités sont séparés ; aucun cas particulier par marque GPU n'a été créé.

Les formes courantes actuelles utilisent une attention ordonnée à 64 threads et 8200 o de mémoire partagée, au plus 2048 clés, sans dépendre du subgroup ni du FP64. Le préfixe partagé est téléversé par groupe de canaux, et le GPU lit directement un préfixe et une fin indépendants. L'accélération de projection G1 est conservée ; les noyaux d'origine A2 et autres réductions en ligne sont conservés comme candidats ou repli hors plage ; la sélection de kernel historique ne doit pas être traitée comme le dispatch par défaut de cette round. La plage numérique et les limites de requête longue sont dans le [protocole de benchmark](<benchmark.md>).

La capacité physique des buffers croît géométriquement ; les descripteurs, le copy et le download n'utilisent que des octets logiques. Les pools device/staging/readback inactifs sont réutilisés après la fin du fence, limités par le nombre d'allocations de l'appareil, évitant des milliers d'allocations dues aux AR qui grandissent par patch.

Priorités suivantes :

1. **Attention à faible SLM et chemins 8/16-wide** : le chemin existant à 512 threads/~39Ko n'est pas adapté à tous les petits GPU intégrés ; de nouveaux candidats de petits blocs génériques et un budget gelé sont nécessaires, pas un remplacement direct par des noyaux RDNA spécifiques
2. **Accès GEMM** : d'abord des chargements vectoriels alignés / une réutilisation de panneau qui ne réordonnent pas l'accumulation K, puis regarder l'ordre de parcours M/N ; L1/L2, les banques SLM et la pression de registres nécessitent une vérification avec les outils du matériel Intel cible ; HostCached ou les horodatages Vulkan ne peuvent pas être passés pour un taux de hit
3. **Copy pur et chemin KV** : sous un budget strict, les nombreuses petites fenêtres de mergeKv sont coûteuses ; le copy de transfer avec vérification de plage/aliasing et le préfixe segmenté sont déjà utilisés pour la requête actuelle ; le préfixe est encore assemblé et téléversé par couche depuis la page CPU. L'identité du pool, la génération et l'invalidation de modification/éjection doivent être définies plus tard avant un miroir KV entre appels ; le zero-copy ou la résidence KV sur appareil n'est pas implémenté
4. **Acceptation du decode** : latence CPU/GPU de la même tâche, charge du copy et estimation de lecture/écriture logique ; la charge du copy n'est pas le trafic DRAM réel ; la bande passante réelle est aussi affectée par les hits de cache de poids/KV et les copies supplémentaires ; aucun seuil de 1 % de MFU n'est fixé

## Vérifications de cette round

- La réflexion ABI des 42 modules passe, y compris la détection de la constante de budget 255 ; les modules d'origine et Budget ne peuvent pas être mélangés
- La validation double GPU et la comparaison d'opérateurs passent : multiple illégal/débordement uint32, 256/512/768/4096, rectangle natif, flat2D, row2D, dépassement de 65535 groupes, fin de couverture partielle non réécrite, attention à 512 threads ; le même kernel est comparé au bit près
- Le chemin des Settings publics est identique au bit près entre la valeur par défaut et la prédiction 256 ; le 1 illégal est rejeté ; le Tiny/Real d'origine double GPU 2/2 ; les comparaisons de modèle réel à budget 256 et 4096 incluent les coordonnées continues, 129 patchs, le préfixe H1 63/513 et les hachages de pages en lecture seule
- La bibliothèque standard pure avec OpenMP etc. désactivés 24/24 (77,38 s) ; la compilation incrémentielle de l'usine BLAS passe. La normalisation/la fenêtre rectangle ultérieures n'ont modifié que l'indexation/la planification internes GPU ; la passe CPU complète non concernée n'a pas été répétée
- Le budget de sortie publique à 8 canaux passe, mais l'écart de valeurs intermédiaires du vieux chemin `--strict-multi` n'est pas encore localisé ; il n'est pas écrit que les 8 canaux passent tous par couche
- `base et modèle`, `bibliothèque standard`, `grille normalisée`, `fenêtre rectangle` conservent les journaux complets
- Pas encore de machine Intel réelle ; les ci-dessus sont une validation en développement AMD/NVIDIA, pas une certification de performance Intel. Le budget optimal n'a pas été choisi d'après les résultats

## Observation de la surcharge de budget

`observation finale` ne sélectionne que 0/512/8192 pour montrer le coût des budgets logiques stricts et plus larges ; 5M réel à un canal 32k, préfixe prefill page froide médiane de 3, decode réchauffement 5 / chronométré 25, residual expérimental désactivé, sans chevauchement avec la compilation. Ce n'est pas de l'auto-réglage ni une prédiction de vitesse Intel. Le chronométrage exclut le chargement du modèle ; le JIT à froid n'est pas chronométré séparément ; le profiling prend 5 à part pour l'attribution de fenêtre/trafic, sans remplacer la base de latence.

| Appareil de développement | Budget logique | prefill ms | H1 decode ms |
| --- | ---: | ---: | ---: |
| 680M | 0 | 44,80 | 9,06 |
| 680M | 512 | 461,64 | 21,55 |
| 680M | 8192 | 79,63 | 9,88 |
| 4060 | 0 | 27,40 | 5,59 |
| 4060 | 512 | 291,72 | 24,09 |
| 4060 | 8192 | 39,40 | 6,88 |

Le budget 512 est très strict ; le coût est loin de « utiliser un tiers de GPU en moins » ; 8192 n'est qu'une borne supérieure plus large, pas une recommandation. Les statistiques de fenêtre mesurent un maximum n'excédant pas la valeur réglée, mais l'occupation/fréquence/consommation réelle du GPU n'a pas été mesurée. Le shader de copy pur de mergeKv forme de nombreuses petites fenêtres sous un budget strict ; la priorité suivante est le chemin de transfer à plage sûre, sans traiter ces chiffres comme des conclusions de réglage Intel.
