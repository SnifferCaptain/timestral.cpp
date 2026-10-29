# Référence API

L'interface publique n'a qu'un seul en-tête d'entrée [timestral/timestral.hpp](<../../timestral/timestral.hpp>) ; il suffit de lier le but CMake timestral ; le build par défaut ne dépend que de la bibliothèque standard C++17.

~~~cpp
#include "timestral/timestral.hpp"
~~~

## timestral::Sample

Un point d'échantillonnage, temps en secondes, origine arbitraire.

**Membres**
| Variable membre | Type | Description |
| --- | --- | --- |
| time | double | Horodatage, en secondes |
| value | double | Relevé ; NaN indique une valeur manquante à cet instant |

## timestral::Timestral

Moteur d'inférence Timestral local : charge le modèle une fois, supporte la soumission concurrente, le cache de préfixe partagé est adressé par contenu, et les pages sans référence sont évictées en LRU.

### Timestral::Timestral()

Charge la configuration et les poids depuis le répertoire local du modèle, puis démarre le thread de planification.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| modelDirectory | const std::filesystem::path& | - | Répertoire contenant config.json et les fichiers de poids |
| settings | Settings | Initialisation agrégée par défaut | Configuration d'instance figée après construction |
| policy | Policy | Policy::Fifo | Stratégie de planification initiale |

**Lève**
| Exception | Condition |
| --- | --- |
| std::runtime_error | Répertoire, configuration ou poids manquants, champ illégal |

**Exemple**
~~~cpp
timestral::Timestral model{"model/timestral-5m"};
~~~

Pas de construction ni d'assignation par copie ; le destructeur arrête le thread de planification, et le futur des requêtes non encore démarrées se termine par std::runtime_error.

### Timestral::predict()

Soumet une prédiction et attend le résultat, équivalent à un get immédiat après submit.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| request | Request | - | Contenu de la requête, voir la table des membres Request |

**Renvoie**
| Type | Description |
| --- | --- |
| Result | Résultat de la prédiction |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | Paramètres de requête illégaux |
| RequestSuperseded | Invalidation par une nouvelle requête de même source ou par abort sous stratégie Mailbox |

**Exemple**
~~~cpp
timestral::Timestral model{"model/timestral-5m"};
std::vector<timestral::Sample> history;
for (std::size_t t = 0; t < 128; ++t)
    history.push_back({static_cast<double>(t) * 0.01, std::sin(static_cast<double>(t) * 0.05)});
timestral::Timestral::Request request;
request.variables = {history};
request.sampleInterval = 0.01;
request.horizonSeconds = 0.32;
auto result = model.predict(request);
for (std::size_t t = 0; t < result.steps(); ++t)
    std::cout << result.time(t) << ',' << result.median(t) << '\n';
~~~

### Timestral::submit()

Soumet une prédiction et retourne immédiatement ; la validation des paramètres et l'ajout de session se font dans le thread de soumission.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| request | Request | - | Un streamId vide est traité comme requête anonyme, le service alloue l'identifiant en interne |

**Renvoie**
| Type | Description |
| --- | --- |
| std::future<Result> | Le résultat est obtenu une fois prêt ; get() lève RequestSuperseded en cas d'invalidation, std::runtime_error en cas d'échec du moteur |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | Lève dans le thread de soumission si les paramètres de la requête sont illégaux, si la session n'existe pas ou si le nouveau relevé n'est pas plus tard que la fin de la session ; la requête n'entre pas dans la file et la session est inchangée |

### Timestral::abort()

Termine une origine : invalide ses requêtes non encore démarrées et libère les références aux pages de préfixe qu'elle détient.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| streamId | const std::string& | - | Identifiant d'origine ; le futur des requêtes invalidées lève RequestSuperseded |

**Note**

Les pages encore référencées par d'autres origines continuent d'être partagées ; les pages sans référence ont immédiatement leur clé de page détachée et sont libérées.

### Timestral::close()

Supprime l'historique conservé par une session et invalide les requêtes non encore démarrées de cette session.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| session | const std::string& | - | Identifiant de session, pris depuis Result::session ; ignoré s'il n'existe pas |

**Note**

Les pages KV utilisées par la session ne sont pas libérées immédiatement ; elles restent dans le LRU et sont évictées à la demande.

### Timestral::setPolicy()

Change de stratégie de planification ; n'affecte que les requêtes non encore planifiées.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| policy | Policy | - | En passant à Mailbox, on ne conserve que la dernière requête non démarrée de même source dans la file |

### Timestral::policy()

Stratégie de planification actuelle.

**Renvoie**
| Type | Description |
| --- | --- |
| Policy | Stratégie actuelle |

### Timestral::pendingRequests()

Nombre de requêtes en file non encore démarrées.

**Renvoie**
| Type | Description |
| --- | --- |
| std::size_t | Nombre de requêtes en attente d'exécution |

### Timestral::patchSize()

Longueur de patch du modèle, 32 pour le modèle 5M.

**Renvoie**
| Type | Description |
| --- | --- |
| std::size_t | Nombre de fentes d'échantillonnage contenues dans un patch |

### Timestral::contextLength()

Contexte maximal pris en charge par le modèle.

**Renvoie**
| Type | Description |
| --- | --- |
| std::size_t | Plafond du contexte, compté en pas |

### Timestral::Policy

Énumération des stratégies de planification.

| Valeur | Description |
| --- | --- |
| Fifo | Exécution dans l'ordre de soumission |
| Mailbox | Conserve la plus récente requête en attente de même source, invalide les requêtes antérieures de même source |
| Continuous | Lot automatique, contraint par Settings::maxBatch et maxVariables |

### Timestral::Settings

Configuration d'instance, lue à la construction, immuable après construction. Tout aux valeurs par défaut, c'est une attention complète identique à l'amont, sans approximation par fenêtre glissante.

**Membres**
| Variable membre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| maxContext | std::size_t | 0 | Plafond du nombre d'échantillons du contexte, 0 prend le contextLength du modèle, la partie excédentaire jette les relevés les plus anciens |
| patchBatch | std::size_t | 16 | Nombre de patches par tour de l'AR par blocs, par exemple 1/2/4/8/16/32, borné à blockSize à la construction |
| maxBatch | std::size_t | 4 | Nombre de fentes de requêtes par tour |
| maxVariables | std::size_t | 8 | Plafond du total de cibles et de covariables par requête |
| cachePages | std::size_t | 2048 | Plafond du nombre de pages KV par nombre de canaux, une page stocke un patch, allocation à la demande, 0 désactive le cache |
| maxSessions | std::size_t | 16 | Nombre de sessions conservant l'historique, au-delà on évince la moins récemment utilisée, 0 ne crée pas de sessions |
| fullPrefillThreshold | std::size_t | 512 | Si l'historique conservé de la session cette tour ne dépasse pas ce nombre de points, on utilise tous les nouveaux relevés, autorisant un nouveau prefill ; 0 fixe toujours la frontière |
| threads | std::size_t | 1 | Nombre de threads BLAS/Eigen/ytensor, par défaut 1 sériel, 0 automatique (au plus 4), N pour un nombre exact de threads |
| continuousRope | bool | false | Position de temps de patch continue expérimentale |
| ropeReferenceInterval | double | 0.0 | Intervalle d'échantillonnage de référence, 0 adopte l'intervalle de la requête |
| gpuThreads | std::size_t | 0 | Budget de fenêtre de dispatch logique Vulkan, 0 illimité, un entier positif doit être un multiple de 256, ce n'est pas une dotation de CU physiques |
| attentionMask | std::function<bool(std::size_t, std::size_t)> | vide | Prédicat de visibilité additionnelle de l'axe time, paramètres (query, key), vide pour le masque natif |

**Note**

Les paramètres de attentionMask sont des indices de token de l'axe time : un token est patchSize points d'échantillonnage, comptés depuis le patch le plus ancien de la grille de contexte du modèle, les patches de prédiction continuent le compte. La paire (query, key) renvoyant false est masquée en plus du masque causal natif ; un token est toujours visible par lui-même ; le prédicat est appelé en concurrence et doit être réentrant. Seuls les backends reference et BLAS l'exécutent nativement ; les backends Eigen, ytensor et Vulkan, une fois défini, repassent par le calcul du modèle reference, avec une vitesse nettement en baisse. Comme attentionMask est un std::function, Settings reste agrégé, mais ne peut plus servir en constexpr.

### Timestral::Request

Une requête de prédiction. Quand les fins ne sont pas alignées, complétion segmentée selon la disponibilité des observations ; l'horizon est compté à partir de la dernière observation valide ; tous les canaux partagent une même axe temporelle, et le conditionnel et la cible sont déterminés par les positions d'observation propres aux variables. Si session est vide, variables est l'historique complet ; sinon variables ne contient que les nouveaux relevés après la fin de chaque canal de cette session, et peut être vide.

**Membres**
| Variable membre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| session | std::string | chaîne vide | Identifiant de session à reprendre, pris depuis le dernier Result::session |
| streamId | std::string | chaîne vide | Identifiant d'origine ; chaîne vide pour requête anonyme ; en reprenant une session, déduplication par session |
| variables | std::vector<std::vector<Sample>> | - | Relevés de chaque canal ordonnés par temps croissant ; en reprenant, le nombre de canaux doit correspondre à la session |
| horizonSeconds | double | 0.0 | Étendue de la prédiction, en secondes, 0 prend patchBatch patches (5M par défaut 512 points), un négatif est illégal |
| sampleInterval | double | 0.0 | Intervalle d'échantillonnage, en secondes, 0 signifie déduire des horodatages |
| symmetricAveraging | bool | true | Ajoute un aller-retours miroir du contexte négatif et moyenne |
| clipNonnegative | bool | true | Tronque au non négatif les trajectoires dont les observations sont toutes non négatives |

La session de canaux synchronisés utilise par défaut tous les nouveaux relevés en phase d'historique court : si l'historique conservé cette tour après rognage par maxContext ne dépasse pas fullPrefillThreshold, un nouveau prefill est autorisé du fait de la réalignment des patches, bufferedSamples vaut 0. Le seuil par défaut est 512 points d'échantillonnage, identique à la longueur d'une tour de décodage par défaut de 5M, mais ne change pas avec patchBatch ou l'horizon de cette tour. Au-delà du seuil, on conserve la frontière de patch et le remplissage à gauche du dernier accueil complet des relevés, et en reprenant on n'envoie au modèle que les patches complétés ; les relevés de queue non complétés sont conservés dans la session et ne participent pas à la prédiction de cette tour. Le prochain ajout d'abord se concatène avec la queue, et les pages anciennes complètes ne sont pas réencodées à cause du changement du reste du nombre de points ajoutés. Cette règle d'entrée ne dépend pas de l'activation du cache KV ; un seuil à 0 fixe toujours la frontière. La première requête utilise toujours tous les relevés, conservant la sémantique de remplissage à gauche de l'amont.

Pendant la mise en attente, la longueur interne de prédiction est le nombre de pas requis plus le nombre de points en attente, puis on saute l'étendue correspondante au début. Ainsi Result::origin reste toujours le temps du dernier relevé, et le nombre de pas retourné et l'horizon sont inchangés, mais la prédiction n'a pas utilisé les relevés en attente et ne peut pas être considérée comme équivalente à une requête sans état sur toutes les observations les plus récentes. Result::bufferedSamples donne le nombre de points mis en attente par canal, dans la plage de 0 à patchSize moins 1. Par exemple avec la configuration par défaut de 5M, quand l'historique atteint 512 points et qu'on ajoute 31, on réutilise 16 pages et met 31 en attente ; ce n'est qu'à l'ajout suivant de 1 point qu'une nouvelle page est créée ; 160 points plus 31 ajoutés restent en phase d'historique court, 191 points participent tous à la prédiction.

Au-delà de maxContext, la fenêtre se déplace par patch entiers ; la normalisation de l'historique et le contexte changent, et il faut reconstruire le KV de la fenêtre ; ici aucune fenêtre glissante approximative à KV gelé n'est activée. La complétion conditionnelle de canaux non synchronisés passe toujours par le chemin intégral par étapes, sans appliquer la règle de mise en attente de queue ; la session conserve l'historique nécessaire aux canaux auxiliaires pour couvrir la position à prédire la plus ancienne, ainsi que le futur connu, sans rognage indépendant par fin de canal. Quand il faut utiliser tous les derniers relevés à chaque fois, on soumet l'historique complet sans session.

### Timestral::Result

La sortie d'une prédiction, temps en secondes, les quantiles sont stockés en ligne principale par (canal, pas, niveau).

**Membres**
| Variable membre | Type | Description |
| --- | --- | --- |
| channels | std::size_t | Nombre de canaux, dans le même ordre que l'entrée |
| sampleInterval | double | Intervalle d'échantillonnage effectif |
| origin | double | Instant précédent la fenêtre de prédiction |
| levels | std::vector<double> | Points de quantile du modèle |
| quantiles | std::vector<double> | (channels, steps, levels) en ligne principale |
| observed | std::vector<std::uint8_t> | (channels, steps), 1 pour observation d'origine |
| prefillSeconds | double | Durée d'encodage des pages de préfixe |
| decodeSeconds | double | Durée de décodage |
| totalSeconds | double | Durée totale d'exécution |
| reusedPages | std::size_t | Nombre de pages de préfixe touchées, hors pages du pass miroir de moyenne symétrique |
| computedPages | std::size_t | Nombre de pages nouvellement calculées |
| bufferedSamples | std::size_t | Nombre de relevés de queue mis en attente par canal, n'ayant pas encore participé à la prédiction de cette tour, inférieur à patchSize |
| session | std::string | Session conservant l'historique complet de cette tour, vide si maxSessions vaut 0 |

#### Result::steps()

Nombre de pas de la fenêtre de prédiction de chaque canal.

**Renvoie**
| Type | Description |
| --- | --- |
| std::size_t | Nombre de pas |

#### Result::time()

Horodatage du pas step.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | Index de pas à partir de 0, doit être inférieur à steps() |

**Renvoie**
| Type | Description |
| --- | --- |
| double | origin plus (step+1) intervalles d'échantillonnage, en secondes |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step supérieur ou égal à steps() |

#### Result::median()

Médiane du pas step, c'est-à-dire le quantile 0.5.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | Index de pas à partir de 0, doit être inférieur à steps() |
| channel | std::size_t | 0 | Index de canal |

**Renvoie**
| Type | Description |
| --- | --- |
| double | Valeur à l'échelle d'origine de la médiane de ce pas |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step ou channel hors plage |

#### Result::quantile()

Quantile du pas step à un niveau arbitraire.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | Index de pas à partir de 0, doit être inférieur à steps() |
| level | double | - | Niveau cible, valeur dans (0,1), prend l'extrémité de la grille si hors de la grille des points de quantile du modèle |
| channel | std::size_t | 0 | Index de canal |

**Renvoie**
| Type | Description |
| --- | --- |
| double | Quantile de ce pas à level, à l'échelle d'origine |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step ou channel hors plage, ou level hors de (0,1) |

**Exemple**
~~~cpp
// quantiles 0.10 et 0.90
double p10 = result.quantile(t, 0.10);
double p90 = result.quantile(t, 0.90);
~~~

#### Result::lower()

Borne inférieure de l'intervalle central.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | Index de pas à partir de 0, doit être inférieur à steps() |
| level | double | - | Couverture de l'intervalle, valeur dans (0,1), level=0.9 prend le quantile 0.05 |
| channel | std::size_t | 0 | Index de canal |

**Renvoie**
| Type | Description |
| --- | --- |
| double | Borne inférieure, à l'échelle d'origine |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step ou channel hors plage, ou level hors de (0,1) |

#### Result::upper()

Borne supérieure de l'intervalle central.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | Index de pas à partir de 0, doit être inférieur à steps() |
| level | double | - | Couverture de l'intervalle, valeur dans (0,1), level=0.9 prend le quantile 0.95 |
| channel | std::size_t | 0 | Index de canal |

**Renvoie**
| Type | Description |
| --- | --- |
| double | Borne supérieure, à l'échelle d'origine |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step ou channel hors plage, ou level hors de (0,1) |

#### Result::isObserved()

Si ce point est une observation d'origine ; une position connue ne représente pas la distribution de prédiction du modèle.

**Paramètres**
| Paramètre | Type | Valeur par défaut | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | Index de pas à partir de 0 |
| channel | std::size_t | 0 | Index de canal |

**Renvoie**
| Type | Description |
| --- | --- |
| bool | true pour observation d'origine, false pour prédiction |

**Lève**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step ou channel hors plage |

#### Result::cacheHitRate()

Taux de toucher du cache paginé.

**Renvoie**
| Type | Description |
| --- | --- |
| double | reusedPages/(reusedPages+computedPages), renvoie 0 s'il n'y a pas de statistique de pages |

### Timestral::RequestSuperseded

Quand la requête est invalidée par une nouvelle requête de même source de la stratégie Mailbox ou par abort, son futur lève cette exception.

**Hérite de**

std::runtime_error
