# Mission 1, étape 3 — Protocole ML

L'étape 3 complète le [notebook unique](../notebooks/01_analyse_et_comparaison.ipynb). Elle utilise les artefacts de l'étape 1 et les résultats LLM de l'étape 2, sans nouvel appel API.

## Apprentissage et sélection

- Train : 254 731 textes complets, 18 étiquettes d'origine. Le vocabulaire TF-IDF et tous les classifieurs sont ajustés sur ce train seul.
- Validation : 54 585 textes pour choisir une configuration par famille, puis le modèle ML de référence, sur le F1 macro, départagé par l'accuracy puis l'ordre défini des candidats.
- Test : 54 586 textes pour l'évaluation finale complémentaire et le lot fixe de 1 000 tickets identique à celui des LLM. Aucun réglage ne dépend de leurs scores.
- Le modèle sélectionné reste ajusté sur le train seul, sans réentraînement train+validation. L'absence de doublons normalisés entre partitions provient de la préparation globale de l'étape 1.

Le TF-IDF utilise unigrammes et bigrammes de mots, `min_df=3`, `max_df=0.995`, au maximum 100 000 variables, fréquence sous-linéaire, norme L2 et float32. Il normalise casse et accents sans supprimer les stop words ou négations. Ce réglage est fixé avant l'entraînement. Le même vectoriseur est réutilisé pour comparer les classifieurs, puis intégré aux pipelines sauvegardés.

| Famille | Configurations |
|---|---|
| DummyClassifier | Classe majoritaire, sans lecture utile du texte |
| ComplementNB | `alpha=0.1` et `alpha=1` |
| Régression logistique | `C=1`, `class_weight=None` ou `balanced`, solveur `lbfgs`, 300 itérations maximum, tolérance `1e-3` |
| SVM linéaire | `C=1`, `class_weight=None` ou `balanced`, `dual=auto`, 3 000 itérations maximum |

Cette petite recherche compare trois algorithmes utiles à une référence ; elle n'est pas exhaustive. Les alertes de convergence sont conservées et les candidats non convergés sont exclus du choix. Les chaînes d'étiquettes sont utilisées directement par sklearn. Les modèles et métriques sont sauvegardés après chaque candidat pour permettre une reprise du même protocole.

## Qualité et latence

Le F1 macro commun porte sur les 17 classes présentes. `Virtual currency` est non évaluée sur ce lot. Le F1 macro du test complet porte sur 18 classes ; ses deux exemples de `Virtual currency` ne permettent pas une estimation fiable.

Les prédictions unitaires utilisent le texte brut et tout le pipeline chargé en mémoire. Le chronomètre comprend TF-IDF, classification, restitution et validation de l'étiquette. Chaque modèle est mesuré sans entraînement concurrent, avec deux threads natifs au maximum. Aucun appel d'échauffement ni reprise cachée n'est ajouté ; le premier appel est conservé. Médiane, p95, moyenne, maximum et taux de sorties valides sont exportés.

Les prédictions sur le test complet utilisent des matrices creuses en batch ; leur temps de classification est identifié séparément. Il ne remplace pas le temps d'attente d'une demande unitaire. Le temps d'ajustement du vectoriseur est distinct de celui des classifieurs, de même que le rechargement d'un pipeline et sa taille compressée. Aucune taille mémoire de service n'est déduite de cette compression.

Le rapprochement final vérifie les mêmes identifiants, les mêmes cibles et la cohérence entre prédictions unitaires et batch. Les comparaisons appariées d'accuracy ML/LLM utilisent 2 000 rééchantillonnages des mêmes tickets (seed 42), avec intervalles descriptifs à 95 %. La sélection ML précède ces comparaisons.

## Reproduction

```powershell
.venv\Scripts\python.exe scripts/run_step3.py
.venv\Scripts\python.exe scripts/finalize_step3.py
.venv\Scripts\python.exe scripts/execute_notebook.py
```

Les sorties sont dans `artifacts/step3/`. Un protocole différent ne remplace pas silencieusement les modèles déjà ajustés. L'exécution du notebook lit les résultats existants ; elle ne relance ni les apprentissages ni les appels LLM. L'étape 4 traitera les coûts d'hébergement, le temps d'implémentation et la recommandation.

Références : documentation officielle [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html), [régression logistique](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) et [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
