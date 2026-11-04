"""Rapprocher les évaluations ML/LLM, sans entraînement ni API."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ml_evaluation import check_sources
from llm_evaluation import MODEL_IDS


def main():
    out = ROOT/'artifacts/step3'
    check_sources(ROOT)
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
    print(comparisons[['approach','model','accuracy','f1_macro_present','median_ms','p95_ms']].to_string(index=False),flush=True)
    print('Comparaison ML/LLM calculée.',flush=True)


if __name__=='__main__':
    main()
