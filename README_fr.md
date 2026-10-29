# Timestral C++

[中文](<README_zh.md>) · [English](<README.md>) · Français · [![linux](<https://github.com/SnifferCaptain/timestral.cpp/actions/workflows/linux.yml/badge.svg?branch=main>)](<https://github.com/SnifferCaptain/timestral.cpp/actions/workflows/linux.yml>)

Bibliothèque d'inférence C++ locale pour le modèle de séries temporelles [Timestral](<https://github.com/mistralai/timestral>) : fournissez-lui des séries numériques horodatées, elle renvoie la médiane, les quantiles et les intervalles de prédiction des pas futurs.

- Le build par défaut n'utilise que la bibliothèque standard C++17, sans Python ni PyTorch
- Prévision multivariée conjointe avec valeurs manquantes et complétion conditionnelle
- Décodage autorégressif par blocs avec un cache KV à adresse de contenu
- Backends optionnels : BLAS intégré, Eigen, ytensor, Vulkan, un seul par build
- Les modèles se chargent depuis des fichiers locaux et ne sont jamais téléchargés

## Compilation

CMake 3.16 ou plus récent, avec un compilateur C++17.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
~~~

Le build par défaut ne produit que la bibliothèque `timestral`, sans tests ni exemples. Les tests sont des outils de régression locaux non livrés avec le dépôt ; ajoutez `-DTIMESTRAL_BUILD_TESTS=ON` à la configuration pour les compiler, puis exécutez `ctest --test-dir build -C Release --output-on-failure`.

## Utilisation

Placez config.json et model.safetensors de la même release dans un même répertoire de modèle ; les sommes de contrôle sont dans la liste locale des checkpoints (non livrée avec le dépôt). Liez la cible `timestral` et incluez l'[en-tête d'entrée](<timestral/timestral.hpp>).

~~~cpp
#include "timestral/timestral.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    timestral::Timestral model{"model/timestral-5m"};

    std::vector<timestral::Sample> history;
    for (std::size_t t = 0; t < 128; ++t) {
        history.push_back({static_cast<double>(t) * 0.01, std::sin(static_cast<double>(t) * 0.05)});
    }

    timestral::Timestral::Request request;
    request.variables = {history};
    request.sampleInterval = 0.01;
    request.horizonSeconds = 0.32;

    const auto result = model.predict(request);
    for (std::size_t t = 0; t < result.steps(); ++t) {
        std::cout << result.time(t) << ',' << result.median(t) << '\n';
    }
}
~~~

Chaque canal est un `vector<Sample>` avec un temps en secondes et une valeur ; NaN marque une lecture manquante. Regroupez les canaux dans une seule Request pour les prévoir conjointement. Un sampleInterval de 0 est déduit de l'entrée, et horizonSeconds compte depuis la dernière observation valide.

### Multivarié

Quand un canal court plus longtemps, la partie excédentaire complète les canaux plus courts avant la prévision conjointe, et les positions connues conservent leurs valeurs d'origine.

![Régions d'historique, de complétion et de prévision pour deux canaux](<assets/multivariate-regions.png>)

- Result.quantiles stocke les valeurs dans l'ordre canal, pas, niveau. Le modèle détermine les niveaux ; 5M en a 21.
- median, quantile, lower et upper visent le canal 0 par défaut.
- isObserved distingue les lectures réelles des valeurs générées ; la sortie est les quantiles marginaux par canal.

### Planification

predict renvoie de façon synchrone, submit renvoie un future, et les poids ne se chargent qu'une fois.

- Fifo exécute les requêtes dans l'ordre de soumission.
- Mailbox ne garde que la dernière requête en attente par streamId.
- Continuous regroupe par lots les requêtes compatibles : maxBatch 4 et maxVariables 8.
- abort annule les requêtes en attente sans préempter une prévision en cours.

### Cache KV

Quand les canaux s'arrêtent à des observations différentes, la complétion conditionnelle s'exécute par étapes sans réutiliser l'historique commun en KV paginée. Les requêtes streaming éligibles mettent en cache un patch par page, et la passe miroir de la moyenne symétrique garde ses propres pages. Pour 5M, une page tient 32 points ; cachePages plafonne les pages par nombre de canaux à 2048 par défaut et les alloue à la demande, ou 0 désactive le cache. Chaque Result nomme une session : placez seulement les nouveaux échantillons dans une Request dont session est renseigné, et la bibliothèque les ajoute à l'historique conservé. Les sessions synchronisées utilisent tous les nouveaux points tant que l'historique retenu ne dépasse pas 512 points, quitte à refaire le préremplissage ; fullPrefillThreshold règle cette limite, et 0 fixe toujours les frontières. Au-delà, les frontières restent ancrées sur le dernier historique entièrement accepté, et un patch final incomplet reste en attente jusqu'à être complet. Result.bufferedSamples indique le nombre de points en attente ; la fenêtre renvoyée commence toujours juste après le dernier échantillon. Au-delà de la limite de contexte, une fenêtre glissante alignée sur les patchs impose toujours un recalcul.

## Performances

### Mesure par backend

Timestral-5M, médiane de 3 exécutions, moyennage symétrique et troncature non négative désactivés ; ar est 1 patch par tour et par est un avant de 32 patches.

![Préremplissage 32k et décodage ar/par de 16k de contexte vers 1k par backend](<assets/backend-compare-1k-20261026.png>)

Les données brutes et les recettes sont dans les [notes de mesure](<doc/zh/benchmark.md>).

### Durée de préremplissage

Variable unique, en millisecondes.

| Longueur | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan normal | Vulkan accéléré | Vulkan accéléré 512 | Vulkan accéléré 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 51.50 | 15.12 | 10.46 | 6.18 | 6.39 | 2.36 | 2.19 | 1.47 | 4.27 | 2.38 | 2.50 | 2.46 | 4.20 | 3.60 | 14.40 | 4.21 |
| 1k | 90.21 | 30.12 | 19.42 | 10.98 | 10.88 | 4.32 | 2.78 | 2.63 | 8.07 | 4.32 | 4.76 | 4.61 | 4.71 | 4.50 | 26.21 | 6.93 |
| 2k | 183.01 | 63.29 | 44.87 | 24.60 | 23.70 | 8.72 | 5.72 | 4.52 | 16.19 | 8.93 | 9.34 | 9.22 | 6.38 | 6.41 | 50.53 | 10.60 |
| 4k | 391.11 | 148.48 | 104.13 | 63.57 | 61.81 | 19.85 | 15.62 | 13.06 | 37.43 | 20.79 | 20.76 | 20.69 | 11.64 | 8.89 | 57.68 | 13.16 |
| 8k | 835.35 | 371.16 | 292.20 | 200.56 | 191.16 | 51.30 | 36.75 | 27.85 | 87.78 | 50.47 | 52.79 | 52.77 | 24.65 | 18.57 | 147.99 | 28.98 |
| 16k | 1954.58 | 1086.64 | 889.81 | 784.45 | 819.54 | 152.42 | 112.14 | 73.33 | 229.22 | 146.71 | 154.96 | 153.84 | 60.03 | 45.15 | 435.50 | 76.36 |
| 32k | 5035.31 | 3284.99 | 3066.65 | 2906.21 | 2580.37 | 488.94 | 354.55 | 224.23 | 706.29 | 499.09 | 529.33 | 501.28 | 168.30 | 138.33 | 1672.28 | 229.07 |

[Résultats 2/4/8 variables](<doc/zh/benchmark.md#多变量prefill>).

### Fenêtre glissante

Un signal à deux canaux avance jusqu'à 128k en prévoyant 512 points à chaque tour. La FIFO figée est une approximation explicite et est désactivée par défaut.

![Courbes de RMSE de fenêtre glissante et latence par image, recalcul intégral natif contre KV paginée en cache](<assets/swa-rmse.png>)

### Prévisions courtes et taille de bloc de décodage

Prévisions 4k regroupées sur six fonctions ; 54 points dépassent le budget de comparaison numérique. [Données et détails de mesure](<doc/zh/benchmark.md#短预测精度与块大小>)

![RMSE regroupé et durée de décodage 4k](<assets/block-ar-4k-summary-20261026.png>)

## Backends et configuration

Sans backend choisi, x86 construit le BLAS intégré, qui ne lie aucune bibliothèque externe et choisit au runtime les noyaux AVX2, SSE4.2 ou scalaires. Les autres cibles utilisent reference par défaut, qui ne demande que la bibliothèque standard, et ARM et RISC-V basculent sur Eigen quand BLAS est sélectionné ; TIMESTRAL_USE_REFERENCE force reference. Eigen et ytensor utilisent les dépendances optionnelles du dépôt, et ytensor exige C++20 et AVX2/FMA. Vulkan demande des en-têtes, un chargeur et un pilote de calcul ; les shaders peuvent être compilés ou fournis en jeu précompilé complet. Voir le [guide Vulkan](<doc/zh/vulkan.md>). Les backends optionnels sont mutuellement exclusifs.

~~~bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_USE_BLAS=ON
~~~

Les autres interrupteurs sont TIMESTRAL_USE_EIGEN, TIMESTRAL_USE_YTENSOR et TIMESTRAL_USE_VULKAN, et TIMESTRAL_BLAS_DECODE sélectionne FP16, BF16 ou INT8 pour le backend BLAS. Les [paramètres](<timestral/timestral.hpp#L40-L57>) sont figés à la construction : threads 1, patchBatch 16 et cachePages 2048 par défaut. attentionMask ajoute un prédicat de visibilité sur les tokens patch de l'axe temporel ; reference et BLAS l'exécutent nativement, tandis qu'Eigen, ytensor et Vulkan passent à reference et deviennent nettement plus lents.

## Support

✅ mesuré · ✅✅ mesuré avec optimisation dédiée · ⚠ non entièrement testé sur matériel · 🛠️ échec non résolu · 🗓️ prévu · ❌ non fourni

### Plates-formes et backends

| Architecture CPU | OS / toolchain | reference | BLAS intégré | Eigen | ytensor | Modèle Vulkan |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| x86-64 | Windows / MSVC | ✅ | ✅✅ | ✅ | 🛠️ | ✅✅¹ |
| x86-64 | GNU/Linux / GCC, Clang | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| x86-64 | macOS / Apple Clang | ⚠ | ⚠ | ⚠ | ⚠ | ❌² |
| x86 (32 bits) | Windows, GNU/Linux | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| ARM64 | Windows | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |
| ARM64 | GNU/Linux | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |
| ARM64 | macOS / Apple Silicon | ⚠ | ❌³ | ⚠ | ❌ | ❌² |
| RISC-V64 | GNU/Linux | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |

1. Vulkan ne couvre que les familles de GPU ci-dessous et le chemin 5M habituel ; le x86 32 bits n'est pas déduit des résultats 64 bits.
2. L'énumération de portabilité et le support de sous-ensembles requis par MoltenVK ne sont pas implémentés.
3. Le BLAS intégré n'a que des noyaux x86, et ytensor exige x86 AVX2+FMA.

Les problèmes de build Linux ont été traités d'après un rapport d'audit, mais la CI GCC/Clang n'a pas encore été exécutée ; ARM et RISC-V n'ont pas d'acceptation matérielle. ytensor est mesuré sous Windows, mais parallelDecodeReal dépasse encore le budget numérique figé ; voir les [notes de compatibilité](<doc/zh/linuxCompatibility.md>). Le 5M réel a été vérifié couche par couche et sur les prévisions, tandis que 28M, 123M et 209M ont seulement été vérifiés contre leur contrat de configuration et de poids.

### Périphériques Vulkan

| Plate-forme | Famille GPU | Statut | Portée |
| --- | --- | :---: | --- |
| Windows x86-64 | AMD RDNA 2 | ✅✅ | Modèle 5M habituel, préfixes paginés et partagés, opérateurs de base, optimisation GLSL dédiée. Le test de charge à 8 variables échoue encore |
| Windows x86-64 | NVIDIA Ada (SM 8.9) | ✅✅ | Modèle habituel, opérateurs et optimisation dédiée. Un échec de budget longue requête est connu |
| Windows, GNU/Linux | Intel Xe graphique intégré | ⚠ | Capacité générique et traitement de sous-groupes en place, aucun résultat matériel. 🗓️ Acceptation sur matériel cible |
| GNU/Linux | Vulkan logiciel, AMD, NVIDIA | ⚠ | Seul un smoke test initial sur ICD logicielle. La CI vérifie les builds et l'édition de liens, pas l'inférence |

RDNA 2 et Ada n'ont chacun été testés que sur un seul périphérique (Radeon 680M, RTX 4060 Laptop). Vulkan est le seul backend GPU ; voir le [périmètre du backend GPU](<doc/zh/gpuScope.md>).

## Documentation et licence

L'interface est documentée dans la [référence API](<doc/zh/api.md>), les détails d'inférence dans les [notes d'inférence](<doc/zh/inference.md>), la sélection du backend dans les [backends de calcul](<doc/zh/backends.md>), les autres guides dans l'[index de documentation](<doc/README.md>), les changements dans l'[historique des mises à jour](<doc/fr/changelog.md>), et les options de ligne de commande dans les [notes des exemples](<example/README.md>).

Les parties Timestral portées dérivent du projet amont Apache-2.0, sources dans les [notes de tiers](<THIRD_PARTY.md>). Ce dépôt est sous licence [MIT](<LICENSE>) ; les poids publiés utilisent une licence MIT modifiée distincte et ne sont pas distribués ici.
