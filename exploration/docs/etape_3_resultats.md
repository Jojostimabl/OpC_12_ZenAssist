# Mission 1, étape 3 — Évaluation Machine Learning

## Protocole

Les **254 731 textes du train** servent à ajuster un TF-IDF partagé et les classifieurs ; les **54 585 textes de validation** servent à sélectionner les paramètres. Les 18 étiquettes d'origine restent distinctes. Les textes sont complets. Le test n'intervient pas dans la sélection et aucun nouvel appel LLM n'est effectué.

TF-IDF : unigrammes et bigrammes de mots, 100 000 variables au maximum, `min_df=3`, `max_df=0.995`, fréquence sous-linéaire, norme L2, calcul float32. Les négations et stop words sont conservés ; casse et accents sont normalisés pour les variables. Le même vocabulaire appris uniquement sur le train est réutilisé dans toutes les configurations.

Sept configurations sont comparées : Dummy majoritaire, ComplementNB (`alpha=0.1` ou `1`), régression logistique et SVM linéaire (`C=1`, avec ou sans pondération `balanced`). Les solveurs sont `lbfgs` pour la régression et `dual=auto` pour le SVM. Le F1 macro sur validation, puis l'accuracy et l'ordre défini des candidats, déterminent les choix. Les alertes de convergence sont exportées ; un candidat non convergé est exclu du choix. Aucun réentraînement train+validation n'est effectué avant le test.

**Modèle ML retenu sur validation : `logistic_regression`.** Les paramètres par famille sont dans `selected_models.csv` et les sept essais dans `validation_metrics.csv`. Les pipelines contiennent le vectoriseur et le classifieur ajustés ; sklearn accepte directement les noms des étiquettes, sans encodage numérique supplémentaire.

## Comparaison sur les mêmes 1 000 tickets

| Approche | Modèle | Accuracy | F1 macro, 17 classes | Médiane, ms | p95, ms | API du test, USD |
|---|---|---:|---:|---:|---:|---:|
| ML | logistic_regression | 72.6 % | 0.557 | 1.761 | 3.387 | 0.00000 |
| ML | linear_svc | 76.5 % | 0.556 | 1.589 | 2.855 | 0.00000 |
| ML | complement_nb | 64.4 % | 0.369 | 17.615 | 25.819 | 0.00000 |
| ML | dummy | 23.1 % | 0.022 | 2.643 | 4.671 | 0.00000 |
| LLM | deepseek/deepseek-v4-flash-0731 | 58.3 % | 0.380 | 1514.545 | 2550.172 | 0.08709 |
| LLM | openai/gpt-6-luna | 55.1 % | 0.390 | 1376.843 | 3015.501 | 0.08592 |
| LLM | openai/gpt-6.1-terra | 70.6 % | 0.475 | 1339.393 | 1936.071 | 0.87078 |
| LLM | qwen/qwen3.8-flash | 54.5 % | 0.407 | 1270.507 | 6909.370 | 0.07279 |

`Virtual currency` est absente du lot commun et n'est pas évaluée dans ce tableau. Les taux de sorties valides sont exportés. Le coût API ML de zéro signifie qu'aucun fournisseur n'a été appelé ; il ne représente pas un coût total d'exploitation nul.

Le modèle choisi sur validation obtient **0.557 de F1 macro** et **72.6 % d'accuracy** sur le lot commun, avec **3.387 ms au p95**. La recommandation de production sera traitée à l'étape 4.

Les latences ML mesurent des prédictions unitaires avec texte brut : vectorisation, classification et restitution de l'étiquette, modèle déjà chargé. Chaque pipeline est mesuré sans entraînement concurrent, avec deux threads natifs au maximum. Les appels LLM ont été mesurés via le réseau dans les conditions de l'étape 2 ; ces chiffres comparent deux modes d'exploitation, pas deux calculs sur le même matériel. Le délai de l'API Python et le rendu de l'interface restent à ajouter. Le premier appel est conservé et identifié dans les exports.

GPT-6.1 Terra a été ajouté le 28 octobre sur les mêmes identifiants et avec le même protocole de choix des prompts. Les trois premiers LLM et les modèles ML conservent leurs mesures du 19 octobre. Le matériel, le réseau et le fournisseur ne sont pas contrôlés entre ces sessions ; cette réserve s'applique aux comparaisons de latence.

`paired_accuracy_ml_llm.csv` donne les écarts d'accuracy entre le ML choisi sur validation et chaque LLM, avec intervalles descriptifs à 95 % sur 2 000 rééchantillonnages appariés (seed 42). Ce ne sont pas des intervalles pour le F1 macro et ils ne garantissent pas la généralisation future.

## Complément : ensemble du test

| Modèle | Tickets | Accuracy | F1 macro, 18 classes |
|---|---:|---:|---:|
| logistic_regression | 54586 | 72.9 % | 0.565 |
| linear_svc | 54586 | 76.1 % | 0.567 |
| complement_nb | 54586 | 67.3 % | 0.373 |
| dummy | 54586 | 23.1 % | 0.021 |

Ce complément n'est pas une comparaison directe avec les LLM. `Virtual currency` n'a que deux tickets dans le test complet : son score reste très incertain. Le F1 macro du test complet porte sur 18 classes ; il ne se compare pas directement au F1 macro sur les 17 classes du lot commun.

## Entraînement et ressources

Ajustement TF-IDF sur le train : **183.3 s**, 100,000 variables, matrice creuse train d'environ 450.5 Mo. Le vectoriseur est ajusté une seule fois pour cette recherche. Son temps doit être ajouté à celui du classifieur pour obtenir le temps d'apprentissage d'un pipeline isolé.

| Modèle | Ajustement classifieur, s | Pipeline compressé, Mo | Rechargement local, s |
|---|---:|---:|---:|
| logistic_regression | 102.5 | 7.99 | 1.540 |
| linear_svc | 354.4 | 15.13 | 0.816 |
| complement_nb | 3.0 | 21.70 | 1.021 |
| dummy | 0.4 | 1.25 | 1.628 |

Les tailles compressées ne mesurent pas la RAM du service. Les durées de prédiction en batch du classifieur sont exportées séparément et ne sont pas présentées comme une latence utilisateur. L'entraînement consomme du calcul et de l'électricité locaux ; aucun montant d'hébergement ni temps de développement n'est inventé. Ces postes seront estimés explicitement à l'étape 4.

Environnement : Python 3.13.5, sklearn 1.9.1, numpy 2.5.3, 22 processeurs logiques, `Intel64 Family 6 Model 170 Stepping 4, GenuineIntel`. Les versions, empreintes de données, paramètres et bibliothèques de calcul sont conservés dans les manifestes.

## Analyse des erreurs

- logistic_regression : `Credit reporting, credit repair services, or other personal consumer reports` → `Credit reporting` (37 cas).
- logistic_regression : `Credit reporting, credit repair services, or other personal consumer reports` → `Debt collection` (12 cas).
- logistic_regression : `Bank account or service` → `Checking or savings account` (11 cas).
- linear_svc : `Credit reporting, credit repair services, or other personal consumer reports` → `Credit reporting` (18 cas).
- linear_svc : `Credit reporting` → `Credit reporting, credit repair services, or other personal consumer reports` (15 cas).
- linear_svc : `Debt collection` → `Student loan` (10 cas).
- complement_nb : `Credit reporting` → `Credit reporting, credit repair services, or other personal consumer reports` (67 cas).
- complement_nb : `Credit card` → `Credit card or prepaid card` (21 cas).
- complement_nb : `Credit reporting, credit repair services, or other personal consumer reports` → `Debt collection` (20 cas).

Le notebook présente les scores par classe avec les effectifs, une matrice de confusion et des exemples d'erreurs du modèle choisi. Les paramètres restent figés après inspection du test. Les libellés historiques qui se recouvrent, les très petites classes, les variantes de lettres et la différence entre corpus historique et flux futur limitent la portée des résultats. Une exposition antérieure des LLM au dataset public est également possible. Les méthodes n'utilisent pas les mêmes informations d'apprentissage : les modèles ML apprennent sur tout le train, les LLM sont employés sans fine-tuning avec les prompts de l'étape 2.

## Artefacts et reproduction

Le [notebook unique](../notebooks/01_analyse_et_comparaison.ipynb) contient les trois premières étapes. Les sorties sont dans `artifacts/step3/` : modèles locaux, vocabulaire, essais sur validation, choix figés, prédictions unitaires et batch, métriques, rapports par classe, confusions et comparaison ML/LLM.

```powershell
.venv\Scripts\python.exe scripts/run_step3.py
.venv\Scripts\python.exe scripts/finalize_step3.py
.venv\Scripts\python.exe scripts/execute_notebook.py
```

Une exécution complète est reprise depuis les résultats existants. Les artefacts de l'étude ne sont pas modifiés par l'ajout de Terra ; l'export et l'API sont décrits séparément dans le [bilan de mission 3](mission_3_integration_ml.md).

Sources des estimateurs : [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
