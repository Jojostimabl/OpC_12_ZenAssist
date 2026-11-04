"""Vérifier les lots et calculer les métriques, sans nouvel appel LLM."""
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
    projections = []
    for model,row in final.iterrows():
        unit_cost = row.cost_known_usd / row.n
        projections.append({'model':model, 'mean_cost_usd_per_ticket':unit_cost,
                            'pilot_30000_tickets_usd':unit_cost*30000,
                            'global_1200000_tickets_usd':unit_cost*1200000})
    pd.DataFrame(projections).to_csv(out/'operating_cost_projection.csv',index=False)
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
    print('Métriques LLM calculées.',flush=True)
    print(final[['prompt','accuracy','f1_macro_present','median_ms','p95_ms','cost_known_usd']].to_string(),flush=True)
    print(f'Coût total connu : {actual:.6f} USD ; inconnus : {unknown}',flush=True)


if __name__=='__main__':
    main()
