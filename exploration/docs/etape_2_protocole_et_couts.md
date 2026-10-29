# Mission 1, étape 2 — Protocole et coûts

Préparé le 19 octobre 2026. Les réponses finales et leur analyse sont présentées dans la section LLM du [notebook unique](../notebooks/01_analyse_et_comparaison.ipynb).

## Expérience retenue

- DeepSeek V4 Flash **0731** (`deepseek/deepseek-v4-flash-0731`) : version GA datée, pour éviter un alias « latest » susceptible de changer pendant l'expérience.
- GPT-6 Luna (`openai/gpt-6-luna`), demandé pour la comparaison.
- Qwen3.8 Flash (`qwen/qwen3.8-flash`), d'une autre famille avec un tarif d'entrée/sortie du même ordre que Luna.

Trois prompts (catégories seules, définitions, puis quatre exemples) sont comparés sur **20 tickets d'entraînement** couvrant les 18 étiquettes. Les exemples inclus dans le prompt viennent d'autres lignes du train. Le prompt de chaque modèle est sélectionné uniquement sur ce développement, avant d'évaluer le lot commun de **1 000 tickets de test**. Total : **3 180 demandes**, toutes variantes comprises.

Les textes du test restent complets et les catégories ne sont pas fusionnées. Les étiquettes réelles ne sont jamais envoyées pour les exemples à prédire. Le raisonnement est désactivé lorsque supporté ; la sortie est limitée à 256 tokens et validée par schéma JSON. La température est fixée à 0 pour DeepSeek et Qwen ; elle est omise pour Luna car le catalogue ne la déclare pas supportée.

## Prévision avant exécution

Tarifs du [catalogue OpenRouter](https://openrouter.ai/api/v1/models) consulté le 19 octobre 2026 et conservé dans `config/llm_models.json`. Ces prix servent aussi de plafonds fournisseur par million de tokens : un endpoint plus cher est exclu de l'appel.

| Modèle | Entrée, USD / million | Sortie, USD / million | Prévision totale, USD |
|---|---:|---:|---:|
| DeepSeek V4 Flash 0731 | 0,15 | 0,15 | 0,132 à 0,364 |
| GPT-6 Luna | 0,10 | 0,50 | 0,115 à 0,351 |
| Qwen3.8 Flash | 0,15 | 0,47 | 0,153 à 0,451 |
| **Trois modèles, 3 180 demandes** | | | **0,400 à 1,166** |

L'estimation inclut prompts, schéma, récits et une marge de protocole. Elle utilise les octets UTF-8 divisés par 4 à 3 comme approximation des tokens, avec 64 à 256 tokens de sortie. Le prompt gagnant étant encore inconnu, la fourchette couvre les variantes proposées. Il ne s'agit pas d'une tokenisation native ni d'une garantie de facture. Aucune réduction liée au cache fournisseur n'est supposée. Les frais d'achat de crédits ou taxes éventuelles sont exclus.

OpenRouter peut publier des prix différents selon le fournisseur : le prix d'une page de modèle n'est pas nécessairement celui du catalogue ou de l'endpoint finalement utilisé. La [comptabilisation d'usage](https://openrouter.ai/support/) permet de relever le coût facturé de chaque réponse. Ce montant sera conservé dans les exports ; un coût absent restera inconnu.

## Maîtrise des appels et des coûts

- La clé est lue depuis `.env` ou l'environnement ; elle n'est enregistrée ni dans les notebooks, ni dans le cache ou les rapports.
- Limite locale de réservation : **2 USD** par défaut, configurable avec `STEP2_BUDGET_USD`.
- Réservation prudente avant chaque envoi, puis remplacement par `usage.cost` lorsqu'il est connu. Une demande sans coût connu conserve sa réservation.
- Prix fournisseurs plafonnés au tarif retenu ; pas de bascule automatique vers un modèle plus cher.
- Aucune reprise du SDK, aucun appel d'échauffement caché et aucun appel de correction de format.
- Réponse enregistrée immédiatement en SQLite. Relancer le script récupère les demandes déjà faites sans les repayer.
- Une demande dont l'issue est incertaine après interruption n'est pas rejouée automatiquement : vérifier d'abord l'activité OpenRouter.
- Trois échecs consécutifs déclenchent un arrêt pour diagnostic.

Le 19 octobre, un premier arrêt était dû à la sérialisation JSON de scalaires NumPy lors de l'export des statistiques, après 146 réponses valides (environ 0,0133 USD). L'export a été corrigé et les 146 réponses reprises du cache, sans répétition des appels.

## Évaluation et latence

Les erreurs et formats invalides restent dans les scores. Le F1 macro principal porte sur les 17 catégories réellement présentes dans le lot commun ; `Virtual currency` est signalée comme non évaluée. Les rapports par classe contiennent les effectifs et la confusion conserve une colonne pour les absences de prédiction.

La latence couvre la préparation du prompt, l'appel, la génération complète, le décodage et la validation. Les timings du cache ne remplacent pas les durées d'origine. Trois demandes au maximum sont en vol, une par modèle ; cette condition est consignée. La médiane, le p95, la moyenne et la couverture sont rapportés, avec un complément sur les seules réponses valides.

Un test de 1 000 textes anciens ne prouve pas la qualité sur les tickets futurs. Les catégories historiques se recouvrent, certaines classes sont très rares et une exposition antérieure des LLM à ce dataset public n'est pas exclue.

## Reproduire

```powershell
# Préparer les références et le chiffrage, sans appel payant :
.venv\Scripts\python.exe scripts/run_step2.py

# Exécuter ou reprendre les appels demandés, à partir du cache :
.venv\Scripts\python.exe scripts/run_step2.py --run

# Recalculer le rapport et les projections depuis les réponses complètes, sans appel API :
.venv\Scripts\python.exe scripts/finalize_step2.py

# Rafraîchir les sorties du notebook à partir des résultats, sans appel API :
.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/01_analyse_et_comparaison.ipynb
```

Les données expérimentales sont dans `artifacts/step2/` : `requests.sqlite`, `predictions.csv`, `metrics.csv`, `selection_*.json`, rapports par classe, confusions, prompts et protocole.

À la fin d'une exécution complète, `run_step2.py --run` lance automatiquement la vérification finale et la production du rapport. Le script de finalisation peut aussi être relancé seul : il vérifie les 3 180 réponses, rapproche la facture du journal SQLite et calcule les projections d'exploitation. Le notebook lit ensuite ces exports.
