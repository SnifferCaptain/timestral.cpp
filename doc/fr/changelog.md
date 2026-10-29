# Historique des mises à jour

[中文](<../zh/changelog.md>) | [English](<../en/changelog.md>) | [Index de documentation](<README.md>)

Chaque version utilise un titre de niveau 2 et un tableau de changements brefs ; les remarques sont facultatives. Ajouter les nouvelles versions à la fin, dans l'ordre des publications. Conserver les mêmes versions et le même ordre des lignes dans chaque langue.

## 1.1.0

Modifications de travail, pas encore publiées.

| Type | Mise à jour | Remarques |
| --- | --- | --- |
| Modification | BLAS monothread par défaut sur x86 | TIMESTRAL_USE_REFERENCE force reference. |
| Optimisation | Cache KV : 2048 pages, allocation à la demande | cachePages=0 désactive le cache. |
| Ajout | Horizon omis : un tour de bloc AR par défaut | 512 points par défaut pour 5M. |
| Ajout | Identifiants, ajouts et fermeture de sessions | maxSessions=16 ; 0 désactive les sessions. |
| Ajout | Prédicats ajoutés aux masques causaux natifs | Eigen/ytensor/Vulkan : reference, plus lent. |
| Correction | Alignement à gauche des historiques incomplets | Résultats alignés sur le recalcul complet. |
| Correction | Frontières stables, queues de session en attente | bufferedSamples indique les points en attente. |
| Ajout | Seuil de préremplissage fixé à 512 points | fullPrefillThreshold=0 : frontières fixes. |
| Correction | Prévision juste après le dernier point reçu | Les points en attente ne sont pas encore utilisés. |
| Optimisation | Cache de l'historique opposé pour le moyennage | Les compteurs ne couvrent que le préfixe positif. |
| Correction | Réutilisation du KV miroir en décodage partagé | Les queues divergentes restent calculées à part. |
| Correction | Horizon conditionnel par défaut respecté | horizonSeconds=0. |
| Correction | Conservation du passé requis et du futur connu | Couvre les grands horodatages absolus. |
| Correction | Libérer les anciennes vues pour éviter le blocage | Les autres sources gardent leurs références. |
| Correction | Annulation du prefill et des pages partielles | Les préfixes déjà utilisés sont préservés. |
| Correction | Fermeture des sessions avec streamId explicite | Les calculs en cours ne sont pas préemptés. |
| Correction | Ordre cohérent de l'historique et des requêtes | Évite l'inversion des instantanés concurrents. |
| Correction | Historique complet avant remplacement de session | Aucun historique partiel si l'allocation échoue. |
| Correction | Protection du contexte et des identifiants saturés | Vérifie les dépassements lors de la troncature. |
| Correction | Contrôle des tailles et des indices des pages KV | Un indice invalide lève une exception. |
| Correction | Page occupée seulement après allocation du KV |  |
| Correction | Contrôle des intervalles et conversions de pas | Évite aussi le débordement du calcul médian. |
| Correction | Projections ytensor alignées pour fiabiliser le KV | Au plus 5 lignes de travail, pas de points ajoutés. |
| Expérimental | KV figé avec une page d'attention sink | reference/BLAS uniquement, pas par défaut. |
| Documentation | Documents de conception classés par langue | Les documents chinois se trouvent dans doc/zh. |
| Documentation | [README français](<../../README_fr.md>) et sélecteurs de trois langues |  |
| Documentation | Historiques et index propres à chaque langue | Ajouter les versions suivantes à la fin. |
| Documentation | Corriger le statut obsolète de la CI du README | La CI vérifie seulement builds et liens. |
| Tests | Sessions, conditions, erreurs et KV miroir testés | Chemins concernés vérifiés sur cinq backends. |
| Limite connue | RoPE ytensor à 8 canaux encore hors tolérance | Tolérance inchangée ; sessions corrigées. |
