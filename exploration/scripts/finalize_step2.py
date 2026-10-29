"""Vérifier les lots complets et produire le bilan, sans nouvel appel LLM."""
from contextlib import closing
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from llm_evaluation import MODEL_IDS, export_results


def main():
    out = ROOT/'artifacts/step2'
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    for field,relative in {
        'source_manifest_sha256':'artifacts/step1/dataset_manifest.json',
        'partitions_sha256':'artifacts/step1/partitions.csv.gz',
        'common_ids_sha256':'artifacts/step1/test_common_ids.csv',
    }.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==protocol[field], relative
    predictions = pd.read_csv(out/'predictions.csv', dtype={'complaint_id':str})
    test = predictions[predictions.phase.eq('test')]
    expected_ids = set(pd.read_csv(out/'test_common_ids.csv',dtype={'complaint_id':str}).complaint_id)
    for model in MODEL_IDS:
        selected = test[test.model.eq(model)]
        assert len(selected)==1000 and selected.complaint_id.is_unique
        assert set(selected.complaint_id)==expected_ids
    assert len(predictions)==len(MODEL_IDS)*(1000+3*20)
    assert predictions.groupby('complaint_id')['expected'].nunique().eq(1).all()
    labels = sorted(json.loads((out/'prompts.json').read_text(encoding='utf-8'))['labels_only'].split('Allowed labels:\n')[1].splitlines())
    # Régénérer les rapports avec le code final, notamment les classes sans support.
    summary = export_results(out,predictions.to_dict('records'),labels)
    with closing(sqlite3.connect(out/'requests.sqlite')) as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls WHERE state='pending'").fetchone()[0]==0
        billing = [json.loads(x[0]) for x in conn.execute("SELECT detail FROM calls WHERE state='done'")]
    billing = pd.DataFrame(billing)
    billing.to_csv(out/'billing_history.csv',index=False)
    assert set(predictions['key']) <= set(billing['key'])
    actual = billing.cost_usd.sum(min_count=1)
    unknown = int(billing.cost_usd.isna().sum())
    # Comparaison appariée : mêmes réclamations, rééchantillonnées ensemble pour les modèles.
    test = test.copy()
    test['correct'] = test.expected.eq(test.prediction).astype(int)
    paired = test.pivot(index='complaint_id',columns='model',values='correct')
    assert paired.notna().all().all()
    indices = np.random.default_rng(42).integers(0, len(paired), size=(2000,len(paired)))
    comparisons = []
    for a,b in combinations(MODEL_IDS,2):
        differences = (paired[a]-paired[b]).to_numpy()
        boot = differences[indices].mean(axis=1)
        lo,hi = np.quantile(boot,[.025,.975])
        comparisons.append({'modele_A':a,'modele_B':b,'ecart_accuracy_A_moins_B':differences.mean(),
                            'ic95_bas':lo,'ic95_haut':hi})
    pd.DataFrame(comparisons).to_csv(out/'paired_accuracy_bootstrap.csv',index=False)
    final = summary[summary.phase.eq('test')].set_index('model').loc[MODEL_IDS]
    best_accuracy = final.accuracy.idxmax()
    best_f1 = final.f1_macro_present.idxmax()
    best_p95 = final.p95_ms.idxmin()
    comparison_lines = []
    for pair in comparisons:
        lo,hi = pair['ic95_bas']*100,pair['ic95_haut']*100
        caution = ' L’intervalle inclut zéro : cette mesure ne les départage pas clairement.' if lo<=0<=hi else ''
        comparison_lines.append(f'- {pair["modele_A"]} face à {pair["modele_B"]} : '
                                f'{pair["ecart_accuracy_A_moins_B"]*100:+.1f} points '
                                f'(intervalle descriptif à 95 % : {lo:+.1f} à {hi:+.1f} points).{caution}')
    projections = []
    for model,row in final.iterrows():
        unit_cost = row.cost_known_usd / row.n
        projections.append({'model':model, 'mean_cost_usd_per_ticket':unit_cost,
                            'pilot_30000_tickets_usd':unit_cost*30000,
                            'global_1200000_tickets_usd':unit_cost*1200000})
    pd.DataFrame(projections).to_csv(out/'operating_cost_projection.csv',index=False)
    projection_rows = [f'| {r["model"]} | {r["mean_cost_usd_per_ticket"]:.7f} | '
                       f'{r["pilot_30000_tickets_usd"]:.2f} | {r["global_1200000_tickets_usd"]:.2f} |'
                       for r in projections]
    token_rows = [f'| {model} | {row.prompt_tokens/row.n:.1f} | {row.completion_tokens/row.n:.1f} |'
                  for model,row in final.iterrows()]
    rows = []
    for model,row in final.iterrows():
        rows.append(f'| {model} | {row["prompt"]} | {row.f1_macro_present:.3f} | {100*row.accuracy:.1f} % | '
                    f'{row.median_ms/1000:.2f} | {row.p95_ms/1000:.2f} | {row.cost_known_usd:.5f} |')
    dev = summary[summary.phase.eq('development')]
    dev_rows = [f'| {row.model} | {row.prompt} | {row.f1_macro_present:.3f} | {100*row.accuracy:.1f} % |'
                for row in dev.itertuples()]
    confusion_lines=[]
    errors=test[test.correct.eq(0)]
    for model in MODEL_IDS:
        top=errors[errors.model.eq(model)].groupby(['expected','prediction'],dropna=False).size().nlargest(3)
        for (truth,pred),count in top.items():
            confusion_lines.append(f'- {model} : `{truth}` → `{pred}` ({count} cas).')
    sources = {'development_calls':int(predictions.phase.eq('development').sum()),
               'test_calls':int(predictions.phase.eq('test').sum()), 'ledger_calls':len(billing),
               'known_cost_usd':actual,'unknown_cost_calls':unknown,
               'not_evaluated_labels':sorted(set(labels)-set(test.expected)),
               'llm_evaluation_code_sha256':hashlib.sha256((ROOT/'src/llm_evaluation.py').read_bytes()).hexdigest()}
    periods = test.groupby('model').finished_at_utc.agg(['min','max']).to_dict('index')
    sources['measurement_periods_utc'] = periods
    extension = protocol.get('extension')
    if extension:
        added = billing[billing.model.eq(extension['added_model'])]
        sources['extension'] = {'model':extension['added_model'], 'calls':len(added),
                                'known_cost_usd':float(added.cost_usd.sum()),
                                'historical_calls_reused':int(extension['baseline_summary']['ledger_calls'])}
    (out/'completion_summary.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    period_rows = [f'| {model} | {values["min"][:10]} | {values["max"][:10]} |' for model, values in periods.items()]
    estimate = pd.read_csv(out/'estimated_costs.csv')
    report = f'''# Mission 1, étape 2 — Résultats OpenRouter, comparaison étendue à GPT-6.1 Terra

L'expérience est terminée : **{sources['development_calls']} essais de prompt et {sources['test_calls']} classifications de test**, avec les mêmes 1 000 tickets pour les {len(MODEL_IDS)} modèles. Les contrôles confirment l'unicité et l'égalité des identifiants de test. Les données, catégories et partitions de l'étape 1 sont conservées.

Les trois configurations initiales datent du 19 octobre 2026 ; **GPT-6.1 Terra a été ajouté le 28 octobre**, avec les mêmes trois prompts et les mêmes cas de développement et de test. Les anciennes réponses sont réutilisées avec leurs coûts et latences d'origine, sans nouvel appel. Les conditions de fournisseur et de réseau diffèrent entre ces sessions : les latences ne sont pas une mesure simultanée.

| Modèle | Premier test, UTC | Dernier test, UTC |
|---|---|---|
{chr(10).join(period_rows)}

## Évaluation finale

Le F1 macro porte sur les **17 catégories présentes** ; `Virtual currency` est absente et n'est pas évaluée. Les échecs sont comptés comme absences de prédiction, sans retirer de ticket.

| Modèle | Prompt choisi sur le train | F1 macro | Accuracy | Médiane, s | p95, s | Test, USD |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

La latence correspond à une demande de classification complète, dans les conditions documentées (trois demandes au maximum en vol, une par modèle). Elle ne comprend pas encore le délai de l'interface utilisateur.

Chaque ligne évalue une configuration **modèle + prompt choisi**. Les prompts et la disponibilité des paramètres diffèrent ; les écarts ne mesurent donc pas uniquement la capacité intrinsèque des modèles.

## Coût réellement connu

**{actual:.5f} USD**, sur {len(billing)} demandes facturées enregistrées, dont les essais de prompts. Nombre de demandes à coût inconnu : **{unknown}**. La prévision sans réduction de cache pour les {len(MODEL_IDS)} modèles est de {estimate.estimate_low_usd.sum():.3f} à {estimate.estimate_high_usd.sum():.3f} USD ; le plafond cumulé de réservation reste {protocol['budget_usd']:.2f} USD. Les frais de recharge éventuels ne sont pas compris.

Coût de l'ajout Terra, essais compris : **{sources.get('extension',{}).get('known_cost_usd',0):.6f} USD**. Le tarif catalogue vérifié le 28 octobre est de 1 USD/M tokens d'entrée et 5 USD/M tokens de sortie ; un cache peut modifier le montant réellement facturé.

| Modèle | Tokens d'entrée moyens, test | Tokens de sortie moyens, test |
|---|---:|---:|
{chr(10).join(token_rows)}

Le chiffrage initial supposait prudemment 64 à 256 tokens de sortie et couvrait tous les prompts possibles. La consommation mesurée utilise le prompt sélectionné et la réponse JSON effectivement produite ; les prix réels et les éventuelles réductions de cache du fournisseur sont reflétés dans `usage.cost`.

Un arrêt d'export initial a été corrigé ; les réponses ont été réutilisées depuis le cache. Le journal de facturation inclut toutes les demandes enregistrées, pas seulement celles du dernier passage du script. Les durées d'origine sont conservées.

### Projection d'exploitation indicative

| Modèle | USD par ticket | Pilote : 30 000 tickets/mois | Global : 1 200 000 tickets/mois |
|---|---:|---:|---:|
{chr(10).join(projection_rows)}

Hypothèses : 30 jours, respectivement 1 000 et 40 000 tickets/jour, une seule demande par ticket, mêmes longueurs de texte, prompts, prix et éventuelles réductions de cache que dans cette mesure. La moyenne provient des coûts de tokens retournés pour les 1 000 demandes de test de chaque modèle ; elle n'inclut pas les essais de prompts, l'hébergement, le développement, les reprises ou les frais de recharge. Les effets de cache et les prix peuvent évoluer : cette extrapolation n'est pas un devis de production.

## Comparaison des prompts sur 20 tickets d'entraînement

| Modèle | Prompt | F1 macro, 18 catégories | Accuracy |
|---|---|---:|---:|
{chr(10).join(dev_rows)}

La sélection ne repose pas sur le test. Avec une vingtaine de cas, ces différences sont peu précises ; elles justifient une évaluation indépendante, sans promettre que le prompt choisi est optimal.

## Confusions fréquentes

{chr(10).join(confusion_lines)}

Les libellés historiques se recouvrent parfois et ne peuvent pas toujours être départagés à partir du seul texte. L'absence de dates dans les entrées respecte le protocole, mais limite l'identification de ces conventions d'annotation.

## Lecture des résultats

**{best_accuracy}** a la meilleure accuracy observée (**{final.loc[best_accuracy,'accuracy']*100:.1f} %**). **{best_f1}** a le meilleur F1 macro observé (**{final.loc[best_f1,'f1_macro_present']:.3f}**), avec une médiane de **{final.loc[best_f1,'median_ms']/1000:.2f} s** et un p95 de **{final.loc[best_f1,'p95_ms']/1000:.2f} s**. Le p95 le plus court revient à **{best_p95}** (**{final.loc[best_p95,'p95_ms']/1000:.2f} s**). La recommandation doit considérer la qualité sur le flux, les classes minoritaires et les délais.

Écarts d'accuracy appariés :

{chr(10).join(comparison_lines)}

Les scores globaux et les confusions constatées appellent une validation métier des catégories et une comparaison au ML avant de retenir une configuration pour l'orientation automatique. Les classes à un ou quelques tickets ne permettent pas de départager solidement les modèles sur ces produits.

## Incertitude et limites

`paired_accuracy_bootstrap.csv` contient les écarts d'accuracy entre modèles et un intervalle descriptif à 95 %, calculé par 2 000 rééchantillonnages appariés (seed 42). Il ne s'agit pas d'un intervalle pour le F1 macro ni d'une garantie sur de futurs tickets.

Les petits effectifs par classe, le choix du prompt sur seulement 20 cas, la possibilité d'une exposition antérieure des LLM à ce dataset public ancien et les variations de réseau/fournisseur limitent l'interprétation. Les données sans texte ne font pas partie du périmètre évalué. La comparaison avec les modèles classiques est présentée dans le [bilan de l'étape 3](etape_3_resultats.md) ; la recommandation métier viendra à l'étape 4.

Le [notebook](../notebooks/01_analyse_et_comparaison.ipynb) présente les prompts, tableaux, graphiques et exemples d'erreurs. Les exports sont conservés dans `artifacts/step2/` et le [protocole](etape_2_protocole_et_couts.md) documente les coûts et limites d'appel.
'''
    (ROOT/'docs/etape_2_resultats.md').write_text(report,encoding='utf-8')
    print('Vérification finale et rapport terminés ; aucun appel API.',flush=True)
    print(final[['prompt','accuracy','f1_macro_present','median_ms','p95_ms','cost_known_usd']].to_string(),flush=True)
    print(f'Coût total connu : {actual:.6f} USD ; inconnus : {unknown}',flush=True)


if __name__=='__main__':
    main()
