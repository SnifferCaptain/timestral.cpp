# Tutoriel de conception du backend BLAS dédié

Ce document part du volume de calcul d'un forward pour déplier de haut en bas tout l'arbre d'optimisation du backend BLAS dédié de Timestral : d'abord décider quels calculs ne sont tout simplement pas faits, puis confier les calculs indispensables à des noyaux adaptés au cache et aux registres. Le backend BLAS ne dépend d'aucune bibliothèque externe ; il est activé par l'option CMake `TIMESTRAL_USE_BLAS`, est adopté par défaut sur x86 si aucun autre backend n'est choisi, et le build par défaut des autres architectures est le reference de la bibliothèque standard pure. Les expériences roofline existantes ci-dessous sont des enregistrements AMD 7735H mono thread, MSVC 19.36 ; la comparaison actuelle des modèles 1/2/4 threads est dans [les données de référence](<benchmark.md>) ; la capacité de cache et le pic sont des estimations et non des mesures par compteurs matériels ; les enregistrements bruts sont dans les notes d'acceptation locales.

## Threads et isolation de sortie

Settings.threads contrôle désormais le pool de threads fixe propre au modèle : 1 pour sériel, 0 pour choix automatique au maximum 4, N pour désigner le nombre de threads de calcul, le thread appelant participe aussi. Les projections sont dispatchées par ligne ou panneau de 16 colonnes, l'attention par têtes et groupes de canaux indépendants, les activations par sorties indépendantes ; les threads d'arrière-plan sont réutilisés entre opérateurs. Chaque tâche détient son propre espace de travail empaqueté ; l'addition sur la dimension K n'est pas scindée entre threads ; les exceptions sont renvoyées à l'appelant après la fin de toutes les tâches. L'imbrication dans le même pool s'exécute sériellement sur le thread courant ; les multiples appelants externes isolent les tâches par une verrou de soumission.

[Les tests de parallélisme](<../../tests/blasParallelTest.cpp>) vérifient la concurrence réelle à deux threads, les multiples appelants, l'imbrication, la récupération d'exception et la cohérence bit à bit à 2/4 threads ; ils ne se contentent pas du nombre de threads retourné en validation d'exécution réelle.

## Chemin du modèle actuel

Le modèle utilise par défaut les FP32 ordonnées par K avec AVX2, l'addition résiduelle après le point et les non-linéarités standard. Une requête unique lit directement le KV segmenté, évitant le ré-empaquetage redondant du préfixe entier. L'attention par lots est découpée en blocs de 32 requêtes, utilisant un SGEMM empaqueté pour QK/PV, en conservant le softmax à deux passes et l'ordre d'addition par clé pour chaque dimension, sans retomber sur le calcul scalaire en segment entier, ni matérialiser la matrice de scores complète. Ce chemin satisfait le budget numérique figé des longues requêtes RoPE. Le préfixe partagé est toujours lu par groupe de canaux ; l'empaquetage préalable des poids, la fusion RMS empaquetée et la disposition QKV sont conservés. Le mode Auto des micro-benchmarks de bas niveau peut toujours tester FMA et réduction en ligne ; les enregistrements historiques de débit et de fusion ne peuvent pas être considérés directement comme la configuration du modèle actuelle. La nouvelle comparaison au niveau modèle suit [le protocole de référence](<benchmark.md>).

## D'où vient le problème

### Combien calcule-t-on par forward

Timestral-5M découpe chaque 32 échantillons en un patch, D=192, 3 têtes de 64 dimensions chacune, largeur cachée du MLP F=512, 8 blocs au total avec 8 sous-couches d'axe time et 2 sous-couches d'axe variate. L'entrée de T échantillons correspond à C=T/32 patches de contexte, plus J patches de prédiction ; la longueur de séquence de l'attention est C+J et non T.

Chaque couche linéaire est comptée en 2MNK FLOPs. En accumulant par slot d'unité : le QKV, la porte et le wO d'une sous-couche d'attention font 368640 FLOPs, 10 sous-couches au total 3686400, 8 MLP au total 4718592, les couches linéaires principales du transformer font 8404992 FLOPs par slot. À T=32768, C=1024, les couches linéaires principales font environ 9.07 GFLOPs, les paires autorisées de l'attention d'axe time ajoutent environ 3.25 GFLOPs. Le backend reference met 4424 ms à terminer ce forward, soit environ 2.8 GFLOP/s selon les FLOPs de matrice effectifs ci-dessus.

### Comment calcule-t-on le MFU

Chaque cœur Zen3 peut émettre 2 FMA 256 bits par cycle, soit 2×8×2=32 FLOPs FP32. Estimation à 4.2 GHz, la borne nominale par cœur est 134.4 GFLOP/s ; les MFU nominaux de ce document sont tous « FLOP/s effectif mesuré divisé par 134.4 ». Ici on estime à 4.2 GHz désigné par l'utilisateur, sans verrouillage de fréquence ni mesure de fréquence effective réelle ; on ne doit pas déduire d'un résultat supérieur à 90 % qu'un boost de fréquence a eu lieu. Si le résultat dépasse 100 %, cela ne veut dire que le dénominateur de fréquence, le comptage des FLOPs ou le chronométrage ne correspondent pas aux hypothèses, au moins sur un point ; on doit d'abord vérifier avant de proclamer une rupture de la borne matérielle.

Le numérateur du MFU au niveau modèle ne compte que les FLOPs de matrice de la formule ; RMSNorm, exp, RoPE, empaquetage, biais et transfert mémoire ne sont pas comptés ; c'est donc un « MFU d'algorithme effectif », pas un taux d'utilisation du matériel.

## Vue d'ensemble de l'arbre d'optimisation

![Arbre d'optimisation BLAS](<../../assets/blas-tree.svg>)

L'optimisation se déploie par couches de haut en bas. La couche requête et la couche d'exécution du modèle décident quels calculs peuvent être évités ou fusionnés ; la couche opérateur décide de la structure avec laquelle le calcul est organisé ; les couches noyau et jeu d'instructions décident de l'efficacité de chaque instruction. Une optimisation de couche basse aussi aboutie soit-elle ne peut pas rattraper ce que la couche haute a trop calculé ; inversement, le calcul économisé par la couche haute, s'il est confié à un noyau inefficace, voit son gain mangé. On déploie ci-dessous couche par couche selon cet arbre.

## Couche requête : d'abord décider ce qui n'est pas calculé

La couche requête reprend le lot FCFS, la pagination par groupe de V canaux et l'adressage par contenu de préfixe du Service commun, sans créer un autre planificateur. Le backend BLAS ne fait ici que deux choses.

Premièrement, le préfixe partagé est diffusé par row%V. Quand plusieurs requêtes du lot touchent le même segment de préfixe déjà en cache, la vue de calcul référence directement la même KV en lecture seule, au lieu de matérialiser une KV complète par requête ; la nouvelle queue reste indépendante par requête.

Deuxièmement, le prefill pur est terminé en avance. Un prefill qui n'écrit le cache que pour un decode ultérieur n'a pas besoin de tête de sortie ; on peut retourner après que la dernière sous-couche d'axe time ait écrit le K/V. Cette coupure ne peut pas être avancée, et ne peut pas servir un forward complet exigeant trace ou sortie.

## Couche d'exécution du modèle : fusion et crochets étroits

### Pourquoi ne pas copier un modèle complet

Le flux de prédiction complet comprend la normalisation de patch, le masque conditionnel, la pagination, le multivarié et le post-traitement des quantiles ; en copier un pour BLAS ferait diverger progressivement les deux chemins. La manière est d'extraire l'orchestration du modèle de reference dans un exécuteur Transformer commun de utils ; le backend ne fournit que des crochets numériques étroits : `project`, `normalize`, `prepareQkv`, `attention`, `projectNormalized`, `projectAttention`, `prepareCacheKv`, `applyGate`, `activate` et `projectResidual`. reference ne couvre que les quatre premiers ; les implémentations par défaut des autres crochets sont identiques au code inline d'origine ; après migration, la sortie du 5M réel est inchangée ligne par ligne.

### Liste de fusion

- RMSNorm fusionné dans l'empaquetage A : l'inverse RMS de chaque ligne est calculé une seule fois ; à l'empaquetage on écrit dans le micro-panneau selon (input×inverse)×scale, sans matérialiser la matrice normalisée. On n'a pas choisi d'absorber le scale dans les poids, car cela changerait l'ordre d'arrondi
- QKV et gate fusionnés en un seul GEMM : les poids fusionnés sont assemblés au chargement ; la sortie conserve la largeur de ligne 4D ; la lecture ultérieure se fait directement par décalage, évitant une copie de décomposition
- RoPE partage sin/cos : Q et K ne calculent les fonctions trigonométriques qu'une fois ; la rotation des dimensions partielles garde l'appariement j et j+headDim/2 de l'amont ; pour 5M, les dimensions réellement rotées sont 0 à 15 et 32 à 47
- Le résidu accumulé avec beta=1 : wO et la projection basse du MLP accumulent directement dans le tenseur résiduel, économisant un tenseur temporaire et un parcours
- La porte sigmoid est faite in situ sur la sortie d'attention ; SwiGLU écrit directement la sortie compacte exigée par la projection basse ; les deux utilisent l'exp vectoriel du bas

La fusion n'est valable que si elle réduit l'allocation et les parcours mémoire ; les fonctions non linéaires ne traversent pas la linéarisation, et l'ordre d'addition n'est pas changé pour la vitesse.

## GEMM : des boucles triples à BLIS

### Pourquoi la boucle triple naïve est lente

La boucle triple naïve i, j, k met à jour un élément de C à la fois ; l'intérieur suit K. Si la matrice logique B est rangée en ligne principale, B[k,j] saute ligne par ligne ; chaque ligne de cache n'utilise qu'un float dans ce point ; échanger les boucles rend B continu, mais si C est relu et réécrit sans cesse ou s'il n'y a pas de blocage de registres, les chargements, les écritures et les dépendances de boucle peuvent encore bloquer le débit FMA. L'essentiel est de faire réutiliser plusieurs fois les A et B lus dans les registres avant d'écrire un bloc de C, et non simplement de changer l'ordre des boucles dans le code source.

### Blocage de registres : micro-noyau 6×16

![Micro-noyau 6×16](<../../assets/blas-microkernel.svg>)

Le micro-noyau calcule à la fois 6 lignes × 16 colonnes de C. À chaque pas k, il lit une ligne de 16 floats de B (2 YMM), diffuse un à un les 6 valeurs de A, et fait 12 FMA. AVX2 a 16 registres YMM au total : 12 accumulateurs de C, 2 vecteurs de B, 1 diffusion de A, il en reste 1 au planificateur du compilateur.

Ainsi chaque pas k ne fait que 2 chargements vectoriels et 6 diffusions, mais accomplit 192 FLOPs ; le ratio chargement/FMA passe de 1:1 à environ 1:1.5 ; estimé seulement avec 2 FMA/cycle, 12 FMA exigent au moins 6 cycles. Estimation avec une latence de dépendance de 4 à 5 cycles, il faut 8 à 10 chaînes d'accumulation indépendantes pour masquer la latence ; 12 en laissent de la marge ; l'occupation exacte des ports et les latences restent à la documentation des instructions du CPU cible et aux mesures. C'est la raison du choix 6×16, pas une preuve que d'autres formes sont forcément pires.

Préchargement des 6 lignes de C avant la boucle K, pour améliorer la probabilité de toucher L1 à l'écriture finale ; l'indice de préchargement lui-même ne garantit pas le toucher. Dans la comparaison historique, les formes partielles de modèle M=12 ont ainsi gagné environ 15 %. On a essayé un déploiement K de 4 pas avec préchargement de B ; MSVC a chassé les accumulateurs des registres ; le débit est passé de 105 à 66 GF/s ; c'est été annulé.

### L'empaquetage : faire au micro-noyau que des lectures ordonnées

![Disposition de l'empaquetage](<../../assets/blas-packing.svg>)

La figure utilise d'abord le B(K,N) en ligne principale générique à titre d'illustration ; le micro-benchmark utilise aussi cette entrée. Le fichier de modèle réel stocke W(N,K) en ligne principale ; au chargement, `rowStride=1` et `columnStride=K` remettent la vue B=Wᵀ au même empaquetage, sans avoir transposé d'abord un B complet. Le micro-noyau espère que les 16 valeurs de B et les 6 valeurs de A lues à chaque pas k sont voisines en mémoire. L'empaquetage consiste à réorganiser la matrice d'origine dans cet ordre : B est découpé en panneaux de 16 colonnes, chaque panneau K lignes × 16 colonnes stocké contiguement ; A a un panneau de 6 lignes, entrelacé par [k][6]. La queue qui ne fait pas 16 colonnes ou 6 lignes est complétée par des zéros ; le micro-noyau n'a pas à traiter les bords ; à l'écriture il n'écrit que la zone valide.

L'empaquetage lui-même est un gather de O(K·N) ; pour une multiplication de matrice à M grand, il peut être étalé sur le calcul de O(M·K·N) ; pour une projection de modèle à M petit, non : à M=12, N=1024, K=192, le chemin générique avec empaquetage B ne fait que 16.6 GF/s, le chemin pré-empaqueté 130 GF/s. Les poids ne changent pas pendant l'inférence ; au chargement du modèle on empaquette donc tous les poids de projection en `PackedWeight` ; pendant l'inférence on n'empaquette que le A qui change avec la requête.

### Faut-il stocker les poids directement au format par blocs dans le fichier

L'idée naturelle est de stocker les poids au format par blocs déjà dans le fichier, épargnant le gather de stride à blocs. L'implémentation actuelle le fait déjà en mémoire : au chargement on empaquette les poids de projection, et chaque projection linéaire lit directement les panneaux par la suite, sans regather le poids depuis la vue stride. Si le chargement du fichier donne un `PackedWeight` identique, l'accès mémoire de la projection en régime établi ne change pas ; on ne doit plus espérer le même gain. L'empaquetage préalable du fichier peut réduire la conversion de démarrage à froid et les copies intermédiaires ; le chargement BLAS d'il y a a fait environ 48 ms, reference environ 33 ms, mais cette différence d'environ 16 ms inclut aussi les coûts de construction des poids fusionnés et d'allocation, et n'est pas une durée d'empaquetage mesurée séparément ni une borne stricte.

Le format de fichier exige des conventions supplémentaires : largeur de panneau 16, dimensions et complétion par zéros, ordre des poids fusionnés, version, ordre des octets et valeur de contrôle des poids source. SSE ou NEON peut choisir des panneaux compatibles 16 colonnes, ou peut exiger une disposition différente ; il n'existe pas encore de contrat de format inter-ISA. Remplacer directement les safetensors amont casserait la compatibilité existante ; on ne change donc pas la disposition des poids publiés à ce stade. Si le démarrage à froid devient réellement une goulot, on peut conserver le modèle d'origine et générer séparément un cache de backend jetable avec version et hachage source, ré-empaqueté en cas de non-correspondance ; c'est plus sûr que d'imposer à tous les backends la même disposition interne ; pas encore implémenté.

Le decode quantifié mérite surtout d'étudier le stockage de panneaux hors ligne : là les poids sont stockés en INT8 ou FP16 ; les stocker dans l'ordre des micro-panneaux permet à la déquantification de produire directement les 16 colonnes exigées par le micro-noyau, sans déployer d'abord en matrice FP32 complète. L'interface de cette partie est déjà réservée, voir plus bas.

### Blocage de cache : boucles à cinq niveaux

![Boucles à cinq niveaux](<../../assets/blas-loops.svg>)

Le blocage de registres résout le ratio chargement/FMA ; le blocage de cache doit garantir que ces chargements touchent le cache proche. On suit les boucles à cinq niveaux de BLIS : la plus extérieure jc découpe N par NC, pc découpe K par KC, ic découpe M par MC, jr et ir parcourent les micro-panneaux par 16 colonnes et 6 lignes.

À KC=256, un micro-panneau B fait 256×16×4=16 Ko, plus 6 Ko de micro-panneau A, ce qui rentre dans le L1 de 32 Ko ; à MC=120, le bloc empaqueté A fait environ 120 Ko, placé dans le L2 de 512 Ko, réutilisé par tous les micro-panneaux B de la boucle jr ; le panneau KC×NC de B reste en L2 ou L3, réutilisé par tous les blocs A de la boucle ic. À l'empaquetage préalable des poids, NC prend directement tout N, car les principales projections d'attention et MLP de 5M n'ont pas N supérieur à 1024 ; cela évite de ré-empaquer A par NC ; si B reste en cache dépend toujours de la capacité, de la concurrence et de l'ordre d'exécution. Les candidats initiaux sont KC 128 ou 256, MC 120 ou 240, NC 128 ou 256 ; l'état actuel est KC=256, MC=120, NC générique=256 ; la liste des candidats est un périmètre de planification et ne signifie pas une énumération systématique exhaustive de toutes les combinaisons. La note du jeu de travail du cache ne donne qu'une analyse de plausibilité et ne prétend pas à des preuves de taux de toucher par compteurs matériels.

### Petit M : noyaux étroits et decode

Le modèle contient beaucoup de projections dont M n'est pas un multiple de 6 ; au decode, M est même égal à 1. M=4 avec un micro-noyau 6 lignes gaspillerait un tiers du calcul ; on a donc écrit un noyau étroit pour 1 à 5 lignes ; les enregistrements historiques indiquent que les formes M=4 passent d'environ 98 à 113 GF/s, soit environ 73 % à 84 % en dénominateur nominal 4.2 GHz.

Le noyau étroit n'utilise qu'une chaîne d'accumulation K. On a essayé l'entrelacement multi-chaînes pour masquer la latence FMA, mais cela change l'ordre de sommation ; l'écart entre shared decode et full recompute du modèle réel dépasse le seuil ; on a donc reviré à l'ordre mono-chaîne ; l'audit final a aussi supprimé le paramètre de chaîne du gabarit constamment égal à 1, pour éviter de garder une abstraction inutile.

À M=1, chaque poids participe à une seule FMA ; 2 FLOPs pour lire 4 octets ; l'intensité de calcul et la bande passante mémoire deviennent la goulot ; le MFU nominal de cette bouée est environ 21 % à 24 %. C'est exactement la forme que le decode quantifié doit résoudre : plus les poids sont petits, plus ils peuvent alimenter les FMA vite.

### Précision et ordre d'addition

Le backend BLAS ne desserre aucun seuil existant : état caché max(1e-5, 1e-4×RMS) et L2 relatif ne dépassant pas 1e-5, prédiction finale 1e-4+1e-3×|ref|, decode en cache contre recalcul intégral max(1e-6, 1e-5×RMS). Le blocage change l'ordre de sommation ; les seuils sont donc sensibles à l'échelle ; mais à l'intérieur du même backend, full recompute et cached decode doivent suivre le même ordre dans la direction K, sinon l'écart entre les deux peut être amplifié ; on doit le reverifier et ne pas présupposer qu'il reste dans 1e-5×RMS ; la contrainte de mono-chaîne des noyaux étroits vient de là.

## Attention : softmax en ligne et saut structuré

### Le problème de la manière naïve

L'attention standard calcule d'abord une matrice de scores L×L, fait un softmax par ligne, puis multiplie par V. À L=1028, la matrice de scores fait environ 4 Mo ; chaque tête doit l'écrire puis la relire, et sous masque causal environ la moitié des scores serait de toute façon masquée.

### Softmax en ligne

![Softmax en ligne](<../../assets/blas-online-softmax.svg>)

Le softmax en ligne (la pensée par blocs utilisée par la série FlashAttention) découpe les clés en blocs ; chaque ligne ne maintient que la valeur maximale courante m, le dénominateur l et la sortie non normalisée O. Après avoir lu les scores S d'un nouveau bloc de clés, on pose m'=max(m, max S), α=exp(m−m'), puis l'=α·l+Σexp(S−m'), O'=α·O+exp(S−m')·V, et on sort O/l à la fin. Quand toute la ligne est masquée, l=0 ; la sortie est fixée à 0, sans inf−inf.

La taille de bloc est 6 requêtes × 64 clés, réutilisant exactement le micro-noyau GEMM : Q est empaqueté en panneau A par [d][6], K par [d][16], V par [64 clés][16 dimensions] en panneau B ; QK est un appel de micro-noyau de profondeur=headDim, PV est un appel de micro-noyau de profondeur=64. Les probabilités écrites par le softmax sont directement rangées par [clé][6], devenant le panneau A de PV, sans transposition supplémentaire intermédiaire.

### Classification des blocs de masque

![Classification des blocs de masque](<../../assets/blas-mask.svg>)

Chaque bloc 6×64 est d'abord classé en trois types : le bloc Empty ne contient aucune paire autorisée (q,k), QK et PV sont sautés ; le bloc Full est entièrement autorisé, on y passe le masque de bytes tous 1 partagé, sans appeler de prédicat élément par élément ni déployer le masque ; seul le bloc Partial masque élément par élément. L'accélérement vient de calculer moins de manière structurée, sans compter sur la prédiction de branche.

Le masque de Timestral n'est pas un simple triangle : les patches manquants, les rôles known-future et l'isolement des groupes de variables perforrent en dessous de la diagonale ; les clés 20 à 22 manquantes de la figure transforment un bloc qui serait Full en Partial. La classification doit donc se baser sur le prédicat réel. La première implémentation appelait le prédicat élément par élément par bloc pour classer, ce qui prenait lui-même beaucoup de temps. L'amélioration est en deux étapes : sans covariable, on prouve géométriquement si un bloc est Full selon les trois branches de `timeAllowed` ; la validité d'une clé est jugée par préfixe somme en O(1). Le bloc Partial déploie le prédicat en un masque de 64 bytes d'un coup avec `fill`, évitant d'appeler `std::function` dans l'intérieur du softmax.

### Chemin à petit nombre de clés

La longueur de séquence de l'axe variate n'est que le nombre de variables ; si le nombre de clés n'excède pas 16, le coût d'empaquetage et de blocage dépasse le calcul ; on passe donc par le chemin court du softmax direct à deux passes.

## Éléments par éléments : exp vectoriel

### Pourquoi exp mérite une optimisation à part

Le softmax en ligne calcule une exp par score ; la porte et SwiGLU en calculent aussi une par élément. Le premier profilage montre que sur 411 ms du forward de 32768 échantillons, l'attention time occupe environ 238 ms, la porte sigmoid environ 22 ms, SwiGLU environ 18 ms ; la cause principale est `std::exp` élément par élément, pas le micro-noyau.

### Décomposition 2^x

![Décomposition exp](<../../assets/blas-exp.svg>)

L'identité de changement de base est `exp(x)=exp2(x·log2(e))`, avec `log(x)=log2(x)/log2(e)` en miroir. Mais remplacer directement par des appels de bibliothèque standard `std::exp2` ne garantit pas plus vite, et AVX2 n'a pas d'instruction exp générique. Ici on accomplit soi-même la réduction vectorielle de portée et la construction par bits, évitant les appels par voie de fonctions de bibliothèque scalaires ; la pensée de mise à l'échelle en puissance exprimée par `ldexp` est dépliée en addition entière, décalage et multiplication.

En exploitant la constitution des flottants, e^x peut être réécrit 2^(x·log2e)=2^n·2^f, où n est entier, |f|≤0.5. Quand n est dans la plage d'exposant normal, on décale n+127 à gauche de 23 bits dans le domaine d'exposant ; l'interprétation par bits est 2^n. L'implémentation réelle décompose n en deux moitiés, construit chacune un nombre normal puis multiplie, couvrant les nombres subnormaux et la borne de débordement. Le 2^f restant est sur un intervalle très étroit ; un polynôme d'ordre 6 y fait environ 1 ulp.

L'implémentation a plusieurs détails concrets :

- L'arrondi utilise une constante magique : les bits bas de la somme flottante de t+1.5·2^23 sont round(t) ; la même somme donne à la fois n flottant et, par soustraction entière, n entier, épargnant un round et un cvt séparés. Elle dépend de l'arrondi par défaut au plus proche pair ; les trois entrées Native vérifient MXCSR, et en cas d'arrondi non par défaut appellent directement l'implémentation scalaire, sans modifier de son propre chef l'environnement de l'appelant
- f=x·log2e−n est calculé par une FMA, avec un seul arrondi ; log2e est décomposé en deux segments haut et bas, compensé par une deuxième FMA l'erreur de la constante elle-même ; sinon l'erreur de la constante multipliée par un |x| grand peut dépasser le budget de 1 ulp de précision simple
- La portée de n est [−150, 128] ; décomposé en deux moitiés, chacune est assemblée en puissance de 2 puis multipliée en chaîne ; les nombres subnormaux et les sorties proches de FLT_MAX sont représentés correctement
- L'entrée est d'abord tronquée à [−103.97, 88.72]. Ces deux extrémités donnent par opération 0 et +inf elles-mêmes ; il n'y a plus besoin de corriger par blend ; max et min renvoient le second opérande en cas de NaN ; en plaçant x en deuxième position, NaN se propage naturellement à travers l'arithmétique

Les coefficients proviennent de exp2f.c de Cephes Release 2.2 ; le droit d'auteur et la note de publication sont dans [les sources tierces](<../../THIRD_PARTY.md>). L'implémentation d'origine est la réduction de Cody-Waite plus le polynôme e^r de Cephes, avec un blend trois fois pour traiter les deux extrémités et NaN. Dans un environnement d'arrondi par défaut au plus proche pair, sans FTZ/DAZ activé, on parcoure tous les floats finis dans [−110,110], puis on ajoute ±inf et des charges NaN partielles (2243428616 éléments de test au total, avec complétion par zéros alignée), comparés à l'exp de précision double. Le ULP moyen ici cumule l'erreur réelle pour chaque résultat de référence fini ; un arrondi correct n'est pas compté zéro non plus ; les résultats de débordement et non finis sont vérifiés séparément par catégorie. Trois bouées de débit de pure fonction de 4096 éléments donnent le tableau suivant :

~~~text
Variante                        Erreur max    Erreur moyenne    Durée par élément
Cody-Waite+polynôme e^r+3 blends  1.26 ulp    0.0739 ulp   0.72 à 0.74 ns
Décomposition 2^x, blend conservé  1.18 ulp    0.0634 ulp   0.70 à 0.75 ns
Décomposition 2^x, blend des deux extrémités supprimé  1.18 ulp    0.0634 ulp   0.59 à 0.63 ns
Décomposition 2^x, NaN aussi par ordre des opérandes  1.18 ulp    0.0634 ulp   0.53 à 0.62 ns
~~~

Remplacer e^r par 2^f tout seul ne garantit pas plus vite ; les deux réductions ont un nombre d'instructions proche, l'erreur s'améliore légèrement ; le changement complet qui supprime encore le blend superflu abaisse la durée par élément de pure fonction d'environ 16 % à 26 % cette bouée. Le ULP moyen baisse d'environ 14 %, mais un parcours uniforme par motif de bits inclut beaucoup d'entrées proches de zéro et ne représente pas l'erreur moyenne sur une distribution d'activation réelle. On ne doit pas présenter cette approximation comme un substitut à l'arrondi correct de la bibliothèque standard ; elle doit encore passer les régressions du softmax, de la porte, de SwiGLU et du modèle complet.

## Garde-fous de correctitude

### Dispatch à l'exécution et isolation d'unités de compilation

Le code AVX2 est placé seulement dans des fichiers source compilés séparément, avec /arch:AVX2 ou -mavx2 -mfma par fichier, pas activé globalement. L'unité de référence confirme AVX2, FMA et la sauvegarde d'état YMM par le système d'exploitation via CPUID, OSXSAVE et XGETBV avant de choisir le noyau AVX2, sinon retour scalaire.

Dans l'unité AVX2, on n'utilise que des intrinsèques et des boucles écrites à la main ; on n'instancie pas de gabarits inline std, on n'appelle pas de `std::exp` scalaire dans les points chauds vectoriels, et on ne lève pas d'exception. La raison : certaines fonctions ou gabarits inline de bibliothèque standard peuvent former des COMDAT de même nom dans plusieurs unités de compilation ; le lien pourrait choisir la copie de l'unité AVX2 pour le chemin de retour scalaire, faisant exécuter des instructions AVX2 sur des machines non compatibles. La validation des paramètres et les exceptions restent dans l'unité de référence.

### NaN et ligne entière masquée

Quand NaN apparaît dans les clés autorisées, NaN doit être transmis à la sortie et non avalé comme une ligne entière masquée ; seule quand les scores de toutes les clés autorisées sont −inf, on considère la ligne entièrement masquée. Cette sémantique est vérifiée par un jeu de tests injectant ±inf et NaN sur les deux chemins scalaire et AVX2.

### Couverture de tests

Le GEMM aléatoire couvre les bords de KC, MC, NC, la transposition et le stride, les blocs de queue, beta, les dimensions vides, les débordements, l'aliasing et la concurrence ; les noyaux étroits 1 à 6 lignes, l'empaquetage RMS, les panneaux INT8 locaux, l'exp vectoriel et les lignes softmax sont comparés au scalaire ; 800 jeux d'attention en ligne couvrent le causal, le sink plus SWA et le masque rare ainsi que les deux chemins fill et prédicat ; le trace, le forward, la pagination et le decode partagé sont vérifiés sur 5M réel et mini-modèle.

## Couche jeu d'instructions : retour SSE4.2 et arbitrage ARM

AVX2+FMA est le premier objectif, mais pas tous les x86 ne le supportent. Le retour SSE4.2 suit la frontière séparant pack et microkernel, réutilise la disposition de panneau 16 colonnes, sans ajouter de nouveau format d'empaquetage. x64 n'a que 16 XMM ; les 24 accumulateurs YMM du 6×16 d'origine ne tiennent pas ; on a donc décomposé le même bloc 6×16 en deux fois 6×8, soit 12 accumulateurs 128 bits ; SSE4.2 n'a pas de FMA ; on garde la mono-chaîne K par sortie en mul puis add, sans réordonner l'ordre de sommation ; l'oracle scalaire et les seuils sont inchangés. Le symbole Native ne désigne toujours que AVX2 ; SSE a son unité de calcul indépendante et `cpuSupportsSse42` (CPUID feuille 1 ECX bit 20, sans vérifier OSXSAVE) ; Auto dispatche par AVX2+FMA→SSE4.2→Scalar ; le mode SSE forcé vérifie la disponibilité avant dispatch, et lève une exception si non disponible. L'unité de compilation SSE utilise des drapeaux par fichier de type `-msse4.2 -mno-avx -mno-avx2 -mno-fma -ffp-contract=off` ; la désassemblage confirme l'absence de mélange VEX ou FMA ; le chemin élément par élément continue de passer par le scalaire sous SSE.

Le GEMM SSE forcé fait environ 40-41.7 GF/s, environ 30 % en dénominateur nominal 134.4 ; l'attention causale simple-tête de géométrie 1028 fait 7.02 ms contre 2.35 ms d'AVX2. C'est un retour de correctitude exécutable, pas un chemin de performance.

ARM n'a pas d'environnement d'exécution et de validation croisée sur cette machine ; pas d'implémentation de NEON non vérifiée. L'utilisateur autorise explicitement ARM à passer par Eigen ; la construction identifie donc l'architecture cible : TIMESTRAL_TARGET_PROCESSOR prend en priorité CMAKE_GENERATOR_PLATFORM (la génération croisée -A ARM64 de VS ne change pas CMAKE_SYSTEM_PROCESSOR) ; si la cible est ARM, on choisit Eigen et explique le backend réel par un message STATUS. Cela ne valide que le routage de la configuration et n'est pas une validation numérique sur du matériel ARM.

## Decode quantifié : seuls les poids sont compressés, le prefill reste FP32

La goulot du decode est différente du prefill. Le prefill calcule un segment entier d'historique d'un coup, c'est à forte densité de calcul ; la CPU ne supporte de toute façon que le calcul FP32 physique, la quantification n'aide pas ; le decode ne fait avancer qu'une poignée de patches à chaque pas, M est petit, et l'accès mémoire des poids commence à dominer. La quantification vise donc seulement le decode : les poids invariants restent compressés, convertis en FP32 à l'arrivée dans les registres pour l'accumulation ; les entrées d'activation, les sorties, les cumulés et le KV sont toujours FP32, sans multiplication de matrice de niveau 3 quantifiée.

Les poids sont toujours rangés en panneaux [K][16 colonnes]. FP16 est converti de retour par `_mm256_cvtph_ps` de F16C dans l'unité AVX2 ; BF16 décale ses 16 bits hauts à gauche en FP32 (c'est une extension exacte, pas une approximation) ; INT8 met à l'échelle symétrique indépendamment par 32 valeurs K et par colonne de sortie : q=clamp(RNE(w/s), −127, 127), le s d'un groupe tout zéro est noté 1. INT8 n'utilise pas VNNI ; après conversion, il passe par des FP32 FMA comme les autres ; ces trois chemins ne sont donc pas du « calcul basse précision », seulement du « stockage basse précision + accumulation FP32 ».

### Porte de phase et retour

La quantification n'est active que sur le vrai decode. `ProjectionPhase` se divise en Float32 et Decode, transmis par valeur ; toute requête de prefill, de trace enregistrée, avec covariables, de point de départ défilant ou d'historique neuf passe par FP32 ; `runLayers` refuse directement la phase Decode dans ces scénarios, au lieu de dégrader silencieusement. Seuls quand la page de cache n'est pas vide, qu'il n'y a pas de nouveau patch d'historique à ce pas (freshPatches==0) et que le nombre de lignes de la tête de sortie n'excède pas 4, la projection est remplacée par `decodeProject` passant par les poids compressés ; M plus grand retombe sur `gemmPacked` FP32. Cela signifie que le decode unique et le decode par lots peuvent passer par des chemins de précision différents ; l'écart est validé selon le budget préfigé, sans être masqué à l'extérieur.

### Budget d'acceptation et mesures réelles

Le budget est figé avant d'exécuter le modèle réel ; non conforme, c'est noté échec, sans seuil rétroactivement rempli : FP16 L2 relatif ≤0.005 et maxAbs≤0.02×max(RMS de référence, 0.1), BF16 et INT8 L2 relatif ≤0.05 et maxAbs≤0.2×max(RMS de référence, 0.1). La conversion de bits énumère les 65536 motifs 16 bits, les parités d'exactement au milieu et les quatre modes fenv ; `decodeTest` utilise le point double comme oracle et couvre les bords M/N/K, le stride, beta, l'aliasing et le débordement ; `decodeModelTest` confirme sur tiny et 5M réel que le forward et les pages de prefill des quatre formats sont deux à deux bit à bit identiques (le format decode n'affecte que la phase Decode), le decode quantifié passe la référence selon le budget, le retour M>4 est bit à bit identique à la base Float32, et le hachage des pages ne change pas avant et après le decode.

La performance est enregistrée honnêtement. Sur le micro-projection GEMV (M=1/4, formes réelles 5M), la quantification est uniformément plus lente que FP32 : à M=4, fp32 environ 98-112 GF/s, fp16 environ 46-80, bf16 environ 44-74, int8 environ 38-69 ; ces formes restent à forte densité de calcul ; le coût de conversion+mise à l'échelle+FMA domine ; c'est noté honnêtement comme sans gain pour le micro-noyau. Mais au niveau modèle, le decode paginé direct quantifié est au contraire plus rapide : H=1 médiane fp32 0.715 ms contre fp16 0.457/bf16 0.454/int8 0.469 ms, H=4 1.118 contre 0.853/0.898/0.908, H=16 2.723 contre 2.586/2.653/2.590. Le prix est l'augmentation de la mémoire résidente (coexistence de packed FP32 et de copie compressée) et le coût de quantification à la construction (construct_ms fp32 36.8 contre fp16 110.2/bf16 87.1/int8 150.9) ; on ne proclame pas une diminution de la mémoire du modèle. Si la requête publique active la moyenne miroir, le pass miroir recalcule intégralement en FP32 ; le gain decode ne couvre que le chemin de pagination directe.

## Expériences sans gain

- Déploiement K 4 pas du micro-noyau plus préchargement B : débordement d'accumulateurs, 105 à 66 GF/s au cube 1024, annulé
- FTZ/DAZ : 461 ms contre 465 ms sur 32768 échantillons, sans différence, non adopté
- Accumulation résiduelle beta=1 : 186 ms à 184 ms, dans le bruit, conservé seulement pour économiser un tenseur temporaire
- Entrelacement multi-chaînes du noyau étroit : changement d'ordre de sommation, dépassement numérique, annulé
- exp ne remplaçant e^r que par 2^f : meilleure précision mais plus lente, le gain de vitesse vient de la suppression du blend

## Résultats d'acceptation et variation des mesures

![Comparaison des backends et processus d'optimisation de l'attention](<../../assets/blas-results.svg>)

La figure suit les données de validation côte à côte de la même machine de la première bouée, sans mélanger les différences de modèle entier de l'expérience exp de cette bouée et de l'ancienne version en gain. Tous les MFU sont unifiés ici en 4.2 GHz×32=134.4 GFLOP/s ; l'ancien dénominateur des journaux bruts est conservé comme preuve historique.

| Objet | Enregistrement première bouée | MFU nominal 4.2 GHz | Re-mesure de cette bouée |
| --- | --- | --- | --- |
| GEMM pré-empaqueté des formes principales de modèle M≥12 | 122 à 130 GF/s | 91 % à 97 % | La plupart du balayage complet environ 105 à 123 GF/s, avec des points aberrants plus bas |
| GEMM pré-empaqueté cube 1024 | 125.2 GF/s | 93.2 % | 110 à 112 GF/s, environ 82 % à 83 % |
| Projection principale M=1 | environ 31 GF/s | environ 23 % | environ 28 à 33 GF/s |
| Attention causale 1028 simple-tête | 2.35 ms, 57.6 GF/s effectifs | 42.8 % | 2.38 à 2.53 ms |
| forward T=32768 (scale inclus) | 169.98 ms, 72.5 GF/s effectifs | 54.0 % | profile 188 à 199 ms, environ 46 % à 49 % |

Le MFU effectif au niveau modèle de la première bouée T=256/1024/4096/16384/32768 converti est respectivement 45.3 %, 56.2 %, 60.0 %, 56.4 %, 54.0 %. Cette bouée sur la même séquence environ 43 %, 53 % à 54 %, 56 %, 51 %, 46 % à 49 % ; le débit absolu est plus bas que la première bouée, même le GEMM sans changement d'algorithme s'est ralenti ; il n'y a actuellement pas de preuve de fréquence, de température ou de comptage de planification pour affirmer la cause.

La durée du micro-benchmark exp prend le minimum des 15 de chaque bouée, 4096 entrées, portée [−20,0], répété 4000 fois ; c'est un expérimentation de débit de pure fonction, pas un protocole de médiane de modèle entier. Les médianes du modèle 32768 en entrelaçant les bouées old/new sont respectivement 213.26/216.41 ms et 198.47/196.73 ms ; la direction n'est pas cohérente ; on ne confirme donc qu'une amélioration du débit local d'exp, sans proclamer une accélération stable du modèle entier. Le stageBench de cette bouée est 191.2 ms, batch environ 312 req/s, stream environ 7.0 ms, decode environ 1.19 ms ; les 171.7 ms, environ 320 req/s, 6.3 ms et 1.02 ms de la première bouée dans une session différente sont tous conservés, sans couvrir la variation par la meilleure fois.

Builds par défaut final, Eigen, ytensor 25/25 chacun, build BLAS AVX2 29/29, aucun avertissement de compilation ; le build scalaire BLAS avec AVX2 explicitement désactivé et exemples désactivés passe 26/26, validant l'unité Fallback renommée. Le profile 32768 final avec protection d'arrondi ajoutée est 196.13 ms, MFU effectif nominal 46.76 % ; la comparaison 176 lignes du reference par défaut donne une différence de 0. Les preuves de build et numériques détaillées sont dans les notes d'acceptation locales. Cette machine n'exécute que Windows/MSVC ; GCC et Clang sous Unix-like n'ont pas été exécutés ; la classification Full de géométrie sans covariable est couverte pour l'instant indirectement par la régression du modèle entier, sans test de comparaison de prédicat par bloc indépendant.

## Entrées de code et sources de recherche

| Couche à lire | Entrée de code |
| --- | --- |
| Fusion du modèle et empaquetage préalable | [Modèle BLAS](<../../timestral/blas/model.cpp>) et [Transformer commun](<../../timestral/utils/transformer.cpp>) |
| Boucles à cinq niveaux, vues et panneaux | [gemm.cpp](<../../timestral/blas/gemm.cpp>) et [gemm.hpp](<../../timestral/blas/gemm.hpp>) |
| 6×16 et noyaux étroits | [Micro-noyau complet](<../../timestral/blas/microkernelAvx2.cpp>) et [noyaux étroits](<../../timestral/blas/microkernelNarrowAvx2.cpp>) |
| Micro-noyau SSE4.2 | [microkernelSse42.cpp](<../../timestral/blas/microkernelSse42.cpp>) |
| Poids et projection du decode quantifié | [decode.cpp](<../../timestral/blas/decode.cpp>) et [decodeAvx2.cpp](<../../timestral/blas/decodeAvx2.cpp>) |
| Softmax en ligne et classification des blocs | [attention.cpp](<../../timestral/blas/attention.cpp>) |
| Exp vectoriel, sigmoid et SwiGLU | [elementwiseAvx2.cpp](<../../timestral/blas/elementwiseAvx2.cpp>) |
| Détection CPU/OS | [cpuX86.cpp](<../../timestral/blas/cpuX86.cpp>) |
| Reproduction du débit et des formules | Outils locaux blasBench et blasProfile (non versionnés) |

BLIS référence le commit figé [e8566eb3](<https://github.com/flame/blis/tree/e8566eb3e773fb54d11b33e371d13f22d2941e50>) ; les conceptions et sources concrètes lues sont dans les notes de recherche locales. Le softmax en ligne peut être comparé à [FlashAttention](<https://arxiv.org/abs/2205.14135>) et [FlashAttention-2](<https://arxiv.org/abs/2307.08691>) ; cette implémentation emprunte la pensée par blocs et la réduction en ligne, sans prétendre reproduire la planification warp du GPU, le pipeline asynchrone ou des caractéristiques de performance identiques. L'écriture du tutoriel « problème de base → structure mathématique → amélioration d'ingénierie → arbitrage » s'inspire de [tdt-nav-kit version figée](<https://github.com/T-DT-Algorithm-2026/tdt-nav-kit/tree/814d5ce0237acc7faa8c070e5c5e61e2ddc81996>) ; les figures sont redessinées par ce projet.

## Route de suivi

- GPU : actuellement seul le backend modèle complet Vulkan et la publication GLSL/SPIR-V sont conservés ; les anciennes expériences OpenGL/OpenCL sont supprimées et ne seront plus des backends de suivi, voir [l'enregistrement d'arbitrage](<gpuScope.md>)
- ARM NEON : si plus tard un environnement ARM vérifiable existe, on implémeta alors en se référant aux noyaux ARMv8 de BLIS, en conservant son droit d'auteur et sa déclaration de licence BSD-3-Clause lors de la copie de la source
