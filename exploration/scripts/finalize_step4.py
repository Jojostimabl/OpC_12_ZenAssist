"""Synthèse de mission 1 à partir des mesures existantes, sans API ni entraînement."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ml_evaluation import check_sources, save_json




def main():
    out=ROOT/'artifacts/step4'
    out.mkdir(parents=True,exist_ok=True)
    check_sources(ROOT)
    a=json.loads((ROOT/'config/step4_assumptions.json').read_text(encoding='utf-8'))
    comparison=pd.read_csv(ROOT/'artifacts/step3/comparison_llm_ml.csv')
    selection=json.loads((ROOT/'artifacts/step3/selection.json').read_text(encoding='utf-8'))
    ml=comparison[comparison.model.eq(selection['best_model'])].iloc[0]
    tasks=pd.DataFrame(a['implementation_tasks'])
    tasks.to_csv(out/'implementation_tasks.csv',index=False)
    setup=[]
    for approach in ['llm','ml']:
        lo=float(tasks[f'{approach}_low_h'].sum())
        hi=float(tasks[f'{approach}_high_h'].sum())
        setup.append({'approach':approach.upper(),'hours_low':lo,'hours_high':hi,
                      'cost_low_usd':lo*a['hourly_rate_usd'],'cost_high_usd':hi*a['hourly_rate_usd']})
    pd.DataFrame(setup).to_csv(out/'implementation_budget.csv',index=False)
    costs=[]
    llm=comparison[comparison.approach.eq('LLM')]
    best_llm_f1=llm.sort_values('f1_macro_present',ascending=False).iloc[0]
    best_llm_accuracy=llm.sort_values('accuracy',ascending=False).iloc[0]
    cheapest_llm=llm.assign(unit_cost=llm.api_test_cost_usd/llm.n).sort_values('unit_cost').iloc[0]
    for scenario,daily in [('pilot',a['pilot_tickets_per_day']),('global',a['global_tickets_per_day'])]:
        volume=daily*a['days_per_month']
        for r in llm.itertuples(index=False):
            api=r.api_test_cost_usd/r.n*volume*a['calls_per_ticket']
            maint=a['maintenance_llm_h_per_month']
            costs.append({'scenario':scenario,'approach':'LLM','model':r.model,'tickets_per_month':volume,
                          'api_usd':api,'hosting_low_usd':0,'hosting_high_usd':0,
                          'maintenance_low_h':maint[0],'maintenance_high_h':maint[1],
                          'budget_low_usd':api+maint[0]*a['hourly_rate_usd'],
                          'budget_high_usd':api+maint[1]*a['hourly_rate_usd']})
        maint=a['maintenance_ml_h_per_month']
        host=a['ml_incremental_hosting_usd_per_month']
        costs.append({'scenario':scenario,'approach':'ML','model':selection['best_model'],
                      'tickets_per_month':volume,'api_usd':0,'hosting_low_usd':host[0],'hosting_high_usd':host[1],
                      'maintenance_low_h':maint[0],'maintenance_high_h':maint[1],
                      'budget_low_usd':host[0]+maint[0]*a['hourly_rate_usd'],
                      'budget_high_usd':host[1]+maint[1]*a['hourly_rate_usd']})
    costs=pd.DataFrame(costs)
    costs.to_csv(out/'monthly_budget.csv',index=False)
    pipeline=ROOT/'artifacts/step3/models'/f'{selection["best_model"]}_pipeline.joblib'
    synthesis={'analysis_date':a['analysis_date'],'measurement_date':a['measurement_date'],
               'recommended_model':selection['best_model'],'candidate':selection['best_candidate'],
               'selection_source':'validation','accuracy_common':ml.accuracy,
               'f1_macro_common_17':ml.f1_macro_present,'p95_ms':ml.p95_ms,
               'wrong_common':int(round(ml.n*(1-ml.accuracy))),
               'automatic_routing_validated':False,'human_confirmation':True,
               'pipeline_sha256':hashlib.sha256(pipeline.read_bytes()).hexdigest(),
               'assumptions_sha256':hashlib.sha256((ROOT/'config/step4_assumptions.json').read_bytes()).hexdigest(),
               'external_api_calls':0,'new_training_runs':0}
    synthesis['best_llm_f1']={'model':best_llm_f1.model,'f1_macro_present':best_llm_f1.f1_macro_present}
    synthesis['best_llm_accuracy']={'model':best_llm_accuracy.model,'accuracy':best_llm_accuracy.accuracy}
    synthesis['measurement_dates']=a.get('measurement_dates',{'initial':a['measurement_date']})
    save_json(out/'synthesis.json',synthesis)
    save_json(out/'assumptions.json',a)
    print('Synthèse et budgets calculés.',flush=True)
    print(pd.DataFrame(setup).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
