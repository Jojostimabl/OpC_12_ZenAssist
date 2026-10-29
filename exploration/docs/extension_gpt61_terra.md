# Ajout de GPT-6.1 Terra à la comparaison — 28 octobre 2026

Le modèle `openai/gpt-6.1-terra` est ajouté aux trois LLM déjà évalués et à la comparaison avec le ML, dans le [notebook unique](../notebooks/01_analyse_et_comparaison.ipynb). Le [support de recommandation](../livrables/ZenAssist_Mission1_Recommandation.pptx) est synchronisé.

## Résultats sur les mêmes 1 000 tickets

| Modèle | Accuracy | F1 macro | Médiane, s | p95, s | API du test, USD |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 72.6 % | 0.557 | 0.002 | 0.003 | 0.000000 |
| linear_svc | 76.5 % | 0.556 | 0.002 | 0.003 | 0.000000 |
| deepseek/deepseek-v4-flash-0731 | 58.3 % | 0.380 | 1.515 | 2.550 | 0.087092 |
| openai/gpt-6-luna | 55.1 % | 0.390 | 1.377 | 3.016 | 0.085918 |
| openai/gpt-6.1-terra | 70.6 % | 0.475 | 1.339 | 1.936 | 0.870779 |
| qwen/qwen3.8-flash | 54.5 % | 0.407 | 1.271 | 6.909 | 0.072794 |

Le F1 macro porte sur les 17 catégories présentes. `Virtual currency` reste absente du lot commun. Les sorties invalides sont comptées comme erreurs, sans retirer de ticket.

La régression logistique choisie sur validation conserve un F1 macro supérieur à celui de Terra sur ce lot.

L’écart d’accuracy **ML retenu moins Terra** vaut +2.0 points, avec un intervalle descriptif apparié à 95 % de -1.1 à +5.2 points. Cet intervalle concerne l’accuracy, pas le F1 macro.

## Protocole conservé

- Trois prompts inchangés, évalués sur les mêmes 20 cas du train ; quatre démonstrations distinctes pour le few-shot.
- Prompt Terra retenu : **`definitions`**, par F1 macro de développement puis accuracy. Aucun réglage sur le test.
- Même texte complet, mêmes 18 étiquettes possibles, sortie JSON structurée limitée à 256 tokens, seed 42 et raisonnement désactivé.
- Température omise pour Terra, car le catalogue ne la déclare pas, comme pour Luna.
- Une seule demande Terra en vol, aucun retry automatique et même plafond cumulé de réservation de 2 USD.
- Les 3 180 réponses historiques sont conservées ; aucune nouvelle requête pour les trois anciens modèles et aucun réentraînement ML.
- Terra est mesuré le 28 octobre ; les mesures historiques datent du 19 octobre. Réseau, charge et infrastructure fournisseur n’étant pas contrôlés entre ces sessions, les écarts de latence ne sont pas une comparaison simultanée.

## Coût et traçabilité

**1 060 nouveaux appels** (60 essais + 1 000 tests), pour **0.931184 USD**. Le total connu de l’étude LLM est **1.193072 USD**, sur 4240 demandes enregistrées ; coûts inconnus : 0. Les éventuels frais de recharge sont exclus.

Tarif catalogue vérifié le 28 octobre : **1 USD/M tokens d’entrée, 5 USD/M tokens de sortie**, avec tarification de cache distincte. La facture est fondée sur les valeurs `usage.cost`, pas sur la prévision. [Source OpenRouter](https://openrouter.ai/openai/gpt-6.1-terra).

La sauvegarde de l’étude initiale est référencée dans `artifacts/step2/terra_extension.json`. Les contrôles d’intégrité sont dans `artifacts/step2/terra_validation.json`, avec les empreintes des prompts, partitions et modèles figés. Les coûts et périodes d’exécution sont dans `completion_summary.json` et `billing_history.csv`.

Le notebook ne déclenche aucun appel payant lorsqu’il est réexécuté. Les comparaisons de qualité, latence et coût, les erreurs par classe, les intervalles appariés et les projections d’exploitation incluent désormais Terra. Les intégrations applicatives des missions 2 et 3 restent inchangées.
