"""Rapprocher les évaluations ML/LLM et rédiger le bilan, sans entraînement ni API."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ml_evaluation import check_sources, quality_scores, save_json
from llm_evaluation import MODEL_IDS


def main():
    out = ROOT/'artifacts/step3'
    check_sources(ROOT)
    completion = json.loads((out/'completion.json').read_text(encoding='utf-8'))
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    common_ids = pd.read_csv(ROOT/'artifacts/step1/test_common_ids.csv',dtype={'complaint_id':str})
    ml = pd.read_csv(out/'predictions_common.csv',dtype={'complaint_id':str})
    llm_all = pd.read_csv(ROOT/'artifacts/step2/predictions.csv',dtype={'complaint_id':str})
    llm = llm_all[llm_all.phase.eq('test')].copy()
    ml_metrics = pd.read_csv(out/'test_metrics.csv')
    llm_metrics = pd.read_csv(ROOT/'artifacts/step2/metrics.csv')
    llm_metrics = llm_metrics[llm_metrics.phase.eq('test')]
    assert len(ml)==4000 and len(llm)==len(MODEL_IDS)*1000
    together = pd.concat([ml,llm],ignore_index=True)
    assert together.groupby('complaint_id').expected.nunique().eq(1).all()
    for model, group in together.groupby('model'):
        assert len(group)==1000 and group.complaint_id.is_unique,model
        assert set(group.complaint_id)==set(common_ids.complaint_id),model
    comparisons = ml_metrics[ml_metrics.scope.eq('common_1000')].copy()
    comparisons['approach']='ML'
    comparisons['api_test_cost_usd']=0.0
    llm_metrics = llm_metrics.copy()
    llm_metrics['approach']='LLM'
    llm_metrics['api_test_cost_usd']=llm_metrics.cost_known_usd
    columns=['approach','model','n','accuracy','f1_macro_present','f1_weighted','median_ms','p95_ms','coverage','api_test_cost_usd']
    comparisons=pd.concat([comparisons[columns],llm_metrics[columns]],ignore_index=True)
    comparisons.to_csv(out/'comparison_llm_ml.csv',index=False)
    best_model=selection['best_model']
    together['correct']=together.expected.eq(together.prediction).astype(int)
    paired=together.pivot(index='complaint_id',columns='model',values='correct')
    indices=np.random.default_rng(42).integers(0,len(paired),size=(2000,len(paired)))
    pair_rows=[]
    for model in llm.model.unique():
        diff=(paired[best_model]-paired[model]).to_numpy()
        means=diff[indices].mean(axis=1)
        lo,hi=np.quantile(means,[.025,.975])
        pair_rows.append({'ml_selected_on_validation':best_model,'llm':model,
                          'accuracy_difference':diff.mean(),'ic95_low':lo,'ic95_high':hi})
    pd.DataFrame(pair_rows).to_csv(out/'paired_accuracy_ml_llm.csv',index=False)
    # Vérifier aussi que l'évaluation unitaire conserve les mêmes sorties que le batch.
    for model in ml.model.unique():
        full=pd.read_csv(out/f'predictions_full_{model}.csv',dtype={'complaint_id':str})
        assert len(full)==completion['full_test_size'] and full.complaint_id.is_unique
        same=ml[ml.model.eq(model)].merge(full,on='complaint_id',suffixes=('_unit','_batch'),validate='one_to_one')
        assert same.expected_unit.eq(same.expected_batch).all()
        assert same.prediction_unit.eq(same.prediction_batch).all(),model
    rows=[]
    for r in comparisons.itertuples(index=False):
        rows.append(f'| {r.approach} | {r.model} | {100*r.accuracy:.1f} % | {r.f1_macro_present:.3f} | '
                    f'{r.median_ms:.3f} | {r.p95_ms:.3f} | {r.api_test_cost_usd:.5f} |')
    full_rows=[]
    for r in ml_metrics[ml_metrics.scope.eq('full_test')].itertuples(index=False):
        full_rows.append(f'| {r.model} | {r.n} | {100*r.accuracy:.1f} % | {r.f1_macro_present:.3f} |')
    selected=pd.read_csv(out/'selected_models.csv')
    resource_rows=[]
    for r in ml_metrics[ml_metrics.scope.eq('common_1000')].itertuples(index=False):
        resource_rows.append(f'| {r.model} | {r.fit_classifier_s:.1f} | {r.pipeline_bytes/1e6:.2f} | {r.load_pipeline_s:.3f} |')
    errors=ml[ml.expected.ne(ml.prediction)]
    lines=[]
    for model in ml.model.unique():
        if model=='dummy':continue
        for (expected,prediction),n in errors[errors.model.eq(model)].groupby(['expected','prediction']).size().nlargest(3).items():
            lines.append(f'- {model} : `{expected}` → `{prediction}` ({n} cas).')
    winner=comparisons[comparisons.model.eq(best_model)].iloc[0]
    vectorization=json.loads((out/'vectorization.json').read_text(encoding='utf-8'))
    report=f'''# Mission 1, étape 3 — Évaluation Machine Learning

## Protocole

Les **254 731 textes du train** servent à ajuster un TF-IDF partagé et les classifieurs ; les **54 585 textes de validation** servent à sélectionner les paramètres. Les 18 étiquettes d'origine restent distinctes. Les textes sont complets. Le test n'intervient pas dans la sélection et aucun nouvel appel LLM n'est effectué.

TF-IDF : unigrammes et bigrammes de mots, 100 000 variables au maximum, `min_df=3`, `max_df=0.995`, fréquence sous-linéaire, norme L2, calcul float32. Les négations et stop words sont conservés ; casse et accents sont normalisés pour les variables. Le même vocabulaire appris uniquement sur le train est réutilisé dans toutes les configurations.

Sept configurations sont comparées : Dummy majoritaire, ComplementNB (`alpha=0.1` ou `1`), régression logistique et SVM linéaire (`C=1`, avec ou sans pondération `balanced`). Les solveurs sont `lbfgs` pour la régression et `dual=auto` pour le SVM. Le F1 macro sur validation, puis l'accuracy et l'ordre défini des candidats, déterminent les choix. Les alertes de convergence sont exportées ; un candidat non convergé est exclu du choix. Aucun réentraînement train+validation n'est effectué avant le test.

**Modèle ML retenu sur validation : `{best_model}`.** Les paramètres par famille sont dans `selected_models.csv` et les sept essais dans `validation_metrics.csv`. Les pipelines contiennent le vectoriseur et le classifieur ajustés ; sklearn accepte directement les noms des étiquettes, sans encodage numérique supplémentaire.

## Comparaison sur les mêmes 1 000 tickets

| Approche | Modèle | Accuracy | F1 macro, 17 classes | Médiane, ms | p95, ms | API du test, USD |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`Virtual currency` est absente du lot commun et n'est pas évaluée dans ce tableau. Les taux de sorties valides sont exportés. Le coût API ML de zéro signifie qu'aucun fournisseur n'a été appelé ; il ne représente pas un coût total d'exploitation nul.

Le modèle choisi sur validation obtient **{winner.f1_macro_present:.3f} de F1 macro** et **{winner.accuracy*100:.1f} % d'accuracy** sur le lot commun, avec **{winner.p95_ms:.3f} ms au p95**. La recommandation de production sera traitée à l'étape 4.

Les latences ML mesurent des prédictions unitaires avec texte brut : vectorisation, classification et restitution de l'étiquette, modèle déjà chargé. Chaque pipeline est mesuré sans entraînement concurrent, avec deux threads natifs au maximum. Les appels LLM ont été mesurés via le réseau dans les conditions de l'étape 2 ; ces chiffres comparent deux modes d'exploitation, pas deux calculs sur le même matériel. Le délai de l'API Python et le rendu de l'interface restent à ajouter. Le premier appel est conservé et identifié dans les exports.

GPT-6.1 Terra a été ajouté le 28 octobre sur les mêmes identifiants et avec le même protocole de choix des prompts. Les trois premiers LLM et les modèles ML conservent leurs mesures du 19 octobre. Le matériel, le réseau et le fournisseur ne sont pas contrôlés entre ces sessions ; cette réserve s'applique aux comparaisons de latence.

`paired_accuracy_ml_llm.csv` donne les écarts d'accuracy entre le ML choisi sur validation et chaque LLM, avec intervalles descriptifs à 95 % sur 2 000 rééchantillonnages appariés (seed 42). Ce ne sont pas des intervalles pour le F1 macro et ils ne garantissent pas la généralisation future.

## Complément : ensemble du test

| Modèle | Tickets | Accuracy | F1 macro, 18 classes |
|---|---:|---:|---:|
{chr(10).join(full_rows)}

Ce complément n'est pas une comparaison directe avec les LLM. `Virtual currency` n'a que deux tickets dans le test complet : son score reste très incertain. Le F1 macro du test complet porte sur 18 classes ; il ne se compare pas directement au F1 macro sur les 17 classes du lot commun.

## Entraînement et ressources

Ajustement TF-IDF sur le train : **{vectorization['fit_transform_train_s']:.1f} s**, {vectorization['vocabulary_size']:,} variables, matrice creuse train d'environ {vectorization['csr_bytes']/1e6:.1f} Mo. Le vectoriseur est ajusté une seule fois pour cette recherche. Son temps doit être ajouté à celui du classifieur pour obtenir le temps d'apprentissage d'un pipeline isolé.

| Modèle | Ajustement classifieur, s | Pipeline compressé, Mo | Rechargement local, s |
|---|---:|---:|---:|
{chr(10).join(resource_rows)}

Les tailles compressées ne mesurent pas la RAM du service. Les durées de prédiction en batch du classifieur sont exportées séparément et ne sont pas présentées comme une latence utilisateur. L'entraînement consomme du calcul et de l'électricité locaux ; aucun montant d'hébergement ni temps de développement n'est inventé. Ces postes seront estimés explicitement à l'étape 4.

Environnement : Python {protocol['python']}, sklearn {protocol['sklearn']}, numpy {protocol['numpy']}, {protocol['logical_cpus']} processeurs logiques, `{protocol['processor']}`. Les versions, empreintes de données, paramètres et bibliothèques de calcul sont conservés dans les manifestes.

## Analyse des erreurs

{chr(10).join(lines)}

Le notebook présente les scores par classe avec les effectifs, une matrice de confusion et des exemples d'erreurs du modèle choisi. Les paramètres restent figés après inspection du test. Les libellés historiques qui se recouvrent, les très petites classes, les variantes de lettres et la différence entre corpus historique et flux futur limitent la portée des résultats. Une exposition antérieure des LLM au dataset public est également possible. Les méthodes n'utilisent pas les mêmes informations d'apprentissage : les modèles ML apprennent sur tout le train, les LLM sont employés sans fine-tuning avec les prompts de l'étape 2.

## Artefacts et reproduction

Le [notebook unique](../notebooks/01_analyse_et_comparaison.ipynb) contient les trois premières étapes. Les sorties sont dans `artifacts/step3/` : modèles locaux, vocabulaire, essais sur validation, choix figés, prédictions unitaires et batch, métriques, rapports par classe, confusions et comparaison ML/LLM.

```powershell
.venv\\Scripts\\python.exe scripts/run_step3.py
.venv\\Scripts\\python.exe scripts/finalize_step3.py
.venv\\Scripts\\python.exe scripts/execute_notebook.py
```

Une exécution complète est reprise depuis les résultats existants. Les artefacts de l'étude ne sont pas modifiés par l'ajout de Terra ; l'export et l'API sont décrits séparément dans le [bilan de mission 3](mission_3_integration_ml.md).

Sources des estimateurs : [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
'''
    (ROOT/'docs/etape_3_resultats.md').write_text(report,encoding='utf-8')
    print(comparisons[['approach','model','accuracy','f1_macro_present','median_ms','p95_ms']].to_string(index=False),flush=True)
    print('Rapport ML terminé ; identifiants, cibles et sorties unitaires/batch vérifiés.',flush=True)


if __name__=='__main__':
    main()
