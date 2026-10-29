# Mission 1, étape 2 — Résultats OpenRouter, comparaison étendue à GPT-6.1 Terra

L'expérience est terminée : **240 essais de prompt et 4000 classifications de test**, avec les mêmes 1 000 tickets pour les 4 modèles. Les contrôles confirment l'unicité et l'égalité des identifiants de test. Les données, catégories et partitions de l'étape 1 sont conservées.

Les trois configurations initiales datent du 19 octobre 2026 ; **GPT-6.1 Terra a été ajouté le 28 octobre**, avec les mêmes trois prompts et les mêmes cas de développement et de test. Les anciennes réponses sont réutilisées avec leurs coûts et latences d'origine, sans nouvel appel. Les conditions de fournisseur et de réseau diffèrent entre ces sessions : les latences ne sont pas une mesure simultanée.

| Modèle | Premier test, UTC | Dernier test, UTC |
|---|---|---|
| deepseek/deepseek-v4-flash-0731 | 2026-10-19 | 2026-10-19 |
| openai/gpt-6-luna | 2026-10-19 | 2026-10-19 |
| openai/gpt-6.1-terra | 2026-10-28 | 2026-10-28 |
| qwen/qwen3.8-flash | 2026-10-19 | 2026-10-19 |

## Évaluation finale

Le F1 macro porte sur les **17 catégories présentes** ; `Virtual currency` est absente et n'est pas évaluée. Les échecs sont comptés comme absences de prédiction, sans retirer de ticket.

| Modèle | Prompt choisi sur le train | F1 macro | Accuracy | Médiane, s | p95, s | Test, USD |
|---|---|---:|---:|---:|---:|---:|
| deepseek/deepseek-v4-flash-0731 | labels_only | 0.380 | 58.3 % | 1.51 | 2.55 | 0.08709 |
| openai/gpt-6-luna | definitions | 0.390 | 55.1 % | 1.38 | 3.02 | 0.08592 |
| qwen/qwen3.8-flash | few_shot | 0.407 | 54.5 % | 1.27 | 6.91 | 0.07279 |
| openai/gpt-6.1-terra | definitions | 0.475 | 70.6 % | 1.34 | 1.94 | 0.87078 |

La latence correspond à une demande de classification complète, dans les conditions documentées (trois demandes au maximum en vol, une par modèle). Elle ne comprend pas encore le délai de l'interface utilisateur.

Chaque ligne évalue une configuration **modèle + prompt choisi**. Les prompts et la disponibilité des paramètres diffèrent ; les écarts ne mesurent donc pas uniquement la capacité intrinsèque des modèles.

## Coût réellement connu

**1.19307 USD**, sur 4240 demandes facturées enregistrées, dont les essais de prompts. Nombre de demandes à coût inconnu : **0**. La prévision sans réduction de cache pour les 4 modèles est de 1.550 à 4.678 USD ; le plafond cumulé de réservation reste 2.00 USD. Les frais de recharge éventuels ne sont pas compris.

Coût de l'ajout Terra, essais compris : **0.931184 USD**. Le tarif catalogue vérifié le 28 octobre est de 1 USD/M tokens d'entrée et 5 USD/M tokens de sortie ; un cache peut modifier le montant réellement facturé.

| Modèle | Tokens d'entrée moyens, test | Tokens de sortie moyens, test |
|---|---:|---:|
| deepseek/deepseek-v4-flash-0731 | 573.2 | 10.2 |
| openai/gpt-6-luna | 749.8 | 15.7 |
| qwen/qwen3.8-flash | 1139.4 | 10.2 |
| openai/gpt-6.1-terra | 749.8 | 18.0 |

Le chiffrage initial supposait prudemment 64 à 256 tokens de sortie et couvrait tous les prompts possibles. La consommation mesurée utilise le prompt sélectionné et la réponse JSON effectivement produite ; les prix réels et les éventuelles réductions de cache du fournisseur sont reflétés dans `usage.cost`.

Un arrêt d'export initial a été corrigé ; les réponses ont été réutilisées depuis le cache. Le journal de facturation inclut toutes les demandes enregistrées, pas seulement celles du dernier passage du script. Les durées d'origine sont conservées.

### Projection d'exploitation indicative

| Modèle | USD par ticket | Pilote : 30 000 tickets/mois | Global : 1 200 000 tickets/mois |
|---|---:|---:|---:|
| deepseek/deepseek-v4-flash-0731 | 0.0000871 | 2.61 | 104.51 |
| openai/gpt-6-luna | 0.0000859 | 2.58 | 103.10 |
| qwen/qwen3.8-flash | 0.0000728 | 2.18 | 87.35 |
| openai/gpt-6.1-terra | 0.0008708 | 26.12 | 1044.94 |

Hypothèses : 30 jours, respectivement 1 000 et 40 000 tickets/jour, une seule demande par ticket, mêmes longueurs de texte, prompts, prix et éventuelles réductions de cache que dans cette mesure. La moyenne provient des coûts de tokens retournés pour les 1 000 demandes de test de chaque modèle ; elle n'inclut pas les essais de prompts, l'hébergement, le développement, les reprises ou les frais de recharge. Les effets de cache et les prix peuvent évoluer : cette extrapolation n'est pas un devis de production.

## Comparaison des prompts sur 20 tickets d'entraînement

| Modèle | Prompt | F1 macro, 18 catégories | Accuracy |
|---|---|---:|---:|
| deepseek/deepseek-v4-flash-0731 | definitions | 0.333 | 40.0 % |
| deepseek/deepseek-v4-flash-0731 | few_shot | 0.315 | 35.0 % |
| deepseek/deepseek-v4-flash-0731 | labels_only | 0.389 | 45.0 % |
| openai/gpt-6-luna | definitions | 0.500 | 55.0 % |
| openai/gpt-6-luna | few_shot | 0.435 | 50.0 % |
| openai/gpt-6-luna | labels_only | 0.426 | 50.0 % |
| openai/gpt-6.1-terra | definitions | 0.537 | 60.0 % |
| openai/gpt-6.1-terra | few_shot | 0.481 | 55.0 % |
| openai/gpt-6.1-terra | labels_only | 0.481 | 55.0 % |
| qwen/qwen3.8-flash | definitions | 0.435 | 50.0 % |
| qwen/qwen3.8-flash | few_shot | 0.519 | 60.0 % |
| qwen/qwen3.8-flash | labels_only | 0.424 | 55.0 % |

La sélection ne repose pas sur le test. Avec une vingtaine de cas, ces différences sont peu précises ; elles justifient une évaluation indépendante, sans promettre que le prompt choisi est optimal.

## Confusions fréquentes

- deepseek/deepseek-v4-flash-0731 : `Credit reporting, credit repair services, or other personal consumer reports` → `Credit reporting` (59 cas).
- deepseek/deepseek-v4-flash-0731 : `Credit reporting` → `Credit reporting, credit repair services, or other personal consumer reports` (58 cas).
- deepseek/deepseek-v4-flash-0731 : `Credit card or prepaid card` → `Credit card` (39 cas).
- openai/gpt-6-luna : `Credit reporting, credit repair services, or other personal consumer reports` → `Credit reporting` (155 cas).
- openai/gpt-6-luna : `Credit card or prepaid card` → `Credit card` (48 cas).
- openai/gpt-6-luna : `Bank account or service` → `Checking or savings account` (31 cas).
- qwen/qwen3.8-flash : `Credit reporting, credit repair services, or other personal consumer reports` → `Credit reporting` (172 cas).
- qwen/qwen3.8-flash : `Credit card or prepaid card` → `Credit card` (46 cas).
- qwen/qwen3.8-flash : `Debt collection` → `Credit reporting` (27 cas).
- openai/gpt-6.1-terra : `Credit reporting` → `Credit reporting, credit repair services, or other personal consumer reports` (58 cas).
- openai/gpt-6.1-terra : `Bank account or service` → `Checking or savings account` (26 cas).
- openai/gpt-6.1-terra : `Credit card` → `Credit card or prepaid card` (22 cas).

Les libellés historiques se recouvrent parfois et ne peuvent pas toujours être départagés à partir du seul texte. L'absence de dates dans les entrées respecte le protocole, mais limite l'identification de ces conventions d'annotation.

## Lecture des résultats

**openai/gpt-6.1-terra** a la meilleure accuracy observée (**70.6 %**). **openai/gpt-6.1-terra** a le meilleur F1 macro observé (**0.475**), avec une médiane de **1.34 s** et un p95 de **1.94 s**. Le p95 le plus court revient à **openai/gpt-6.1-terra** (**1.94 s**). La recommandation doit considérer la qualité sur le flux, les classes minoritaires et les délais.

Écarts d'accuracy appariés :

- deepseek/deepseek-v4-flash-0731 face à openai/gpt-6-luna : +3.2 points (intervalle descriptif à 95 % : -0.2 à +6.4 points). L’intervalle inclut zéro : cette mesure ne les départage pas clairement.
- deepseek/deepseek-v4-flash-0731 face à qwen/qwen3.8-flash : +3.8 points (intervalle descriptif à 95 % : +0.6 à +7.0 points).
- deepseek/deepseek-v4-flash-0731 face à openai/gpt-6.1-terra : -12.3 points (intervalle descriptif à 95 % : -15.4 à -9.1 points).
- openai/gpt-6-luna face à qwen/qwen3.8-flash : +0.6 points (intervalle descriptif à 95 % : -1.5 à +2.7 points). L’intervalle inclut zéro : cette mesure ne les départage pas clairement.
- openai/gpt-6-luna face à openai/gpt-6.1-terra : -15.5 points (intervalle descriptif à 95 % : -18.9 à -11.8 points).
- qwen/qwen3.8-flash face à openai/gpt-6.1-terra : -16.1 points (intervalle descriptif à 95 % : -19.6 à -12.5 points).

Les scores globaux et les confusions constatées appellent une validation métier des catégories et une comparaison au ML avant de retenir une configuration pour l'orientation automatique. Les classes à un ou quelques tickets ne permettent pas de départager solidement les modèles sur ces produits.

## Incertitude et limites

`paired_accuracy_bootstrap.csv` contient les écarts d'accuracy entre modèles et un intervalle descriptif à 95 %, calculé par 2 000 rééchantillonnages appariés (seed 42). Il ne s'agit pas d'un intervalle pour le F1 macro ni d'une garantie sur de futurs tickets.

Les petits effectifs par classe, le choix du prompt sur seulement 20 cas, la possibilité d'une exposition antérieure des LLM à ce dataset public ancien et les variations de réseau/fournisseur limitent l'interprétation. Les données sans texte ne font pas partie du périmètre évalué. La comparaison avec les modèles classiques est présentée dans le [bilan de l'étape 3](etape_3_resultats.md) ; la recommandation métier viendra à l'étape 4.

Le [notebook](../notebooks/01_analyse_et_comparaison.ipynb) présente les prompts, tableaux, graphiques et exemples d'erreurs. Les exports sont conservés dans `artifacts/step2/` et le [protocole](etape_2_protocole_et_couts.md) documente les coûts et limites d'appel.
