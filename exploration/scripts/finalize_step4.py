"""Synthèse de mission 1 à partir des mesures existantes, sans API ni entraînement."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ml_evaluation import check_sources, save_json


def records(frame):
    return json.loads(frame.to_json(orient='records',double_precision=15))


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
    save_json(out/'presentation_data.json',{
        'synthesis':synthesis,'assumptions':a,'comparison':records(comparison),
        'validation':records(pd.read_csv(ROOT/'artifacts/step3/validation_metrics.csv')),
        'selected_ml':records(pd.read_csv(ROOT/'artifacts/step3/selected_models.csv')),
        'llm_trials':records(pd.read_csv(ROOT/'artifacts/step2/metrics.csv').query("phase=='development'")),
        'monthly_costs':records(costs),'implementation':setup,
        'dataset':json.loads((ROOT/'artifacts/step1/dataset_manifest.json').read_text(encoding='utf-8')),
        'llm_selections':[
            json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'artifacts/step2').glob('selection_*.json'))]
        ,'llm_billing':json.loads((ROOT/'artifacts/step2/completion_summary.json').read_text(encoding='utf-8'))
    })
    cost_rows=[f'| {r.scenario} | {r.model} | {r.tickets_per_month:,} | {r.api_usd:.2f} | '
               f'{r.hosting_low_usd:.0f}–{r.hosting_high_usd:.0f} | {r.budget_low_usd:.2f}–{r.budget_high_usd:.2f} |'
               for r in costs.itertuples(index=False)]
    task_rows=[f'| {r.task} | {r.llm_low_h}–{r.llm_high_h} | {r.ml_low_h}–{r.ml_high_h} |'
               for r in tasks.itertuples(index=False)]
    report=f'''# Mission 1, étape 4 — Synthèse et recommandation

Analyse mise à jour le **28 octobre 2026**, avec GPT-6.1 Terra mesuré le 28 octobre et les mesures initiales ML/LLM du **19 octobre 2026**. Ce script de synthèse ne lance aucun entraînement ni appel payant : il relit les résultats enregistrés.

## Décision proposée

**Privilégier la régression logistique pondérée comme aide à l'étiquetage, avec confirmation humaine.**

- Le choix ML a été figé sur le F1 macro de validation : `{selection['best_candidate']}`.
- Sur les mêmes 1 000 tickets que les LLM, elle obtient un F1 macro de **{ml.f1_macro_present:.3f}**, une accuracy de **{ml.accuracy*100:.1f} %**, une médiane de **{ml.median_ms:.3f} ms** et un p95 de **{ml.p95_ms:.3f} ms**. Le meilleur F1 macro LLM est **{best_llm_f1.f1_macro_present:.3f}** (`{best_llm_f1.model}`) ; la meilleure accuracy LLM est **{best_llm_accuracy.accuracy*100:.1f} %** (`{best_llm_accuracy.model}`). Les p95 LLM vont de {llm.p95_ms.min()/1000:.2f} à {llm.p95_ms.max()/1000:.2f} secondes.
- Le SVM est proche en F1 macro (0,556) et a une meilleure accuracy (76,5 %). Il reste une alternative si le besoin métier privilégie les résultats sur le volume global. Ce constat ne remplace pas le critère de sélection fixé ; valider tout changement sur de nouvelles données indépendantes.
- **{synthesis['wrong_common']} tickets sur 1 000 sont mal classés** par le ML retenu. Ce résultat ne justifie pas une orientation totalement automatique. L'utilisateur doit pouvoir confirmer ou corriger l'étiquette.
- Les mesures ML portent sur le pipeline chargé en mémoire, les mesures LLM incluent le réseau. Le délai clic-vers-affichage et le comportement sous charge restent à tester.

Le test commun couvre 17 catégories. `Virtual currency` n'y est pas évaluée ; deux exemples dans le test complet ne suffisent pas à fiabiliser son score. Les données sans texte et les tickets futurs restent hors du périmètre de preuve. Ne pas fusionner les libellés historiques sans décision métier ni table de correspondance.

## Produit et scénario pédagogique

La recommandation produit découle des scores et latences observés. Le scénario OpenClassrooms impose néanmoins une **mission 2 avec intégration LLM**, puis une **mission 3 avec remplacement par le ML**. Le pilote LLM sera donc une étape pédagogique de transition ; il ne prouve pas que le LLM est le meilleur choix final.

Le pilote de mission 2 a été intégré avec Qwen à partir de la comparaison initiale. Après ajout de Terra, le meilleur F1 macro observé revient à `{best_llm_f1.model}`, la meilleure accuracy à `{best_llm_accuracy.model}` et le coût API unitaire le plus bas à `{cheapest_llm.model}`. Cette extension de comparaison ne modifie pas le code des intégrations existantes. Aucun seuil d'attente acceptable n'est imposé dans la fiche ; l'établir avec le client avant de figer un choix produit. Les coûts observés ne sont pas remplacés par un dépassement fictif pour reproduire le scénario pédagogique.

Les conditions de réseau/fournisseur des sessions du 19 et du 28 octobre diffèrent ; les latences ne sont pas des mesures simultanées. Les prompts, partitions et identifiants restent identiques, et le nouveau prompt est choisi exclusivement sur le développement.

## Coût d'exploitation mensuel

Hypothèses : 30 jours, pilote de 1 000 tickets/jour (30 000/mois), scénario global de 40 000 tickets/jour (1 200 000/mois, base de 200 clients), un appel par ticket, mêmes prompts, textes, cache et coûts d'inférence que lors de la mesure du modèle (19 ou 28 octobre). Les prix ne sont pas garantis à la date de cette synthèse.

| Scénario | Modèle | Tickets/mois | API, USD | Serveur ML ajouté, USD | Budget avec maintenance, USD |
|---|---|---:|---:|---:|---:|
{chr(10).join(cost_rows)}

La colonne de budget inclut une **hypothèse de maintenance** de 2–4 h/mois pour le LLM et 3–6 h/mois pour le ML, valorisées à **50 USD/h**, convention modifiable et non tarif convenu. Elle ne constitue pas une dépense mesurée. Les plages se recouvrent : l'avantage économique global n'est pas établi par la seule absence de facture API ML.

Référence serveur vérifiée le 25 octobre : [DigitalOcean Basic Regular](https://www.digitalocean.com/pricing/droplets), 1 vCPU/1 GiB à 6 USD/mois ou 1 vCPU/2 GiB à 12 USD/mois. **Scénario budgétaire à valider en charge**, sans promesse que ces tailles suffisent. L'entraînement reste séparé du serveur d'inférence. Une ressource existante peut éviter ce serveur supplémentaire si sa capacité est vérifiée.

Coûts communs exclus : hébergement de l'application, base de données et authentification existantes. Taxes, recharge OpenRouter, options de stockage/sauvegarde/observabilité, incidents majeurs et nouvelle collecte d'annotations ne sont pas compris. Les coûts API suivent la [facturation d'usage OpenRouter](https://openrouter.ai/docs/faq), avec frais d'achat de crédits distincts.

Au pilote, les API seules coûtent environ {costs[(costs.scenario=='pilot') & (costs.approach=='LLM')].api_usd.min():.2f}–{costs[(costs.scenario=='pilot') & (costs.approach=='LLM')].api_usd.max():.2f} USD/mois, à comparer au serveur ML supplémentaire de 6–12 USD. Au volume global, les API seules représentent environ {costs[(costs.scenario=='global') & (costs.approach=='LLM')].api_usd.min():.2f}–{costs[(costs.scenario=='global') & (costs.approach=='LLM')].api_usd.max():.2f} USD/mois ; le budget serveur ML pourrait être inférieur si sa capacité suffit. Les coûts humains et la qualité restent déterminants.

## Temps et coût de mise en place

**Estimations de travail à affiner après lecture du dépôt d'intégration.** Un développeur connaît l'application, les accès sont disponibles, l'authentification existe et les données/modèles de mission 1 sont prêts. Les heures excluent refonte métier, achats, attente d'accès et audit réglementaire ; elles ne décrivent pas le temps de calcul de l'entraînement.

| Travail | LLM, h | ML, h |
|---|---:|---:|
{chr(10).join(task_rows)}
| **Total** | **14–24** | **20–34** |

À la convention de 50 USD/h, cela représente **700–1 200 USD pour le LLM** et **1 000–1 700 USD pour le ML**, hors étude déjà réalisée. Les hypothèses sont modifiables dans `config/step4_assumptions.json` ; les tableaux sont recalculés par `scripts/finalize_step4.py`.

## Mise en production LLM

1. Dans React, bouton d'auto-étiquetage de la réclamation, état de chargement, prévention des doubles envois, résultat proposé puis confirmation/correction humaine.
2. Le navigateur appelle le serveur de l'application. Celui-ci vérifie l'utilisateur, l'accès au ticket et la taille/type du texte, puis appelle OpenRouter avec la clé stockée côté serveur, le modèle et le prompt versionnés.
3. Valider l'objet JSON et l'étiquette parmi les catégories autorisées. Traiter timeout, erreur HTTP, sortie vide ou incorrecte. Prévoir un nombre borné de reprises et comptabiliser coût et attente cumulés ; ne pas changer de modèle ou de prix silencieusement.
4. Journaliser identifiant du ticket, version du prompt/modèle, résultat, latence et coût, sans recopier le texte complet dans les logs d'exploitation. Évaluer les conditions de traitement des récits avant l'envoi au fournisseur.
5. Déployer le pilote après tests fonctionnels et cas d'erreurs ; mesurer clic-vers-affichage, taux de correction humaine, taux d'échec et dépenses. Désactivation du bouton ou retour à l'étiquetage manuel si le service échoue.

## Mise en production ML

1. Emballer TF-IDF et classifieur dans un seul pipeline, avec versions des dépendances, catégories, empreinte et métriques. Le pipeline local d'étude n'est pas encore un artefact de release.
2. Créer l'export `pickle` et le workflow GitHub Actions à la release conformément à la mission 3 ; joindre l'artefact et les métriques à la release. Charger uniquement une version provenant de ce circuit de publication.
3. Le service FastAPI charge le pipeline au démarrage et expose la route `POST /tags` avec le texte `user_claim`. Valider entrée et catégorie retournée, vérifier accès et disponibilité, limiter les ressources et gérer les erreurs.
4. Remplacer l'appel LLM de React par l'API ML dans la version finale, en conservant confirmation, correction et chargement. Tester la chaîne complète ; l'API/réseau ajoute un délai absent des mesures locales.
5. Versionner les déploiements et conserver la version précédente pour revenir en arrière. Suivre RAM, charge, p50/p95, erreurs, fréquence des classes et corrections humaines. Reconstituer un jeu annoté récent, réentraîner hors du serveur d'inférence et valider avant toute nouvelle release.

Le dataset prédit une catégorie de produit, **pas une équipe destinataire**. L'orientation requiert une correspondance catégorie→équipe validée par le métier, avec traitement manuel des catégories non couvertes.

## Conditions avant généralisation

- Valider la taxonomie attendue et les équipes destinataires ; conserver les 18 catégories tant que cette décision n'est pas prise.
- Constituer un jeu récent annoté indépendamment des essais ; compléter les classes rares et mesurer les erreurs coûteuses par produit.
- Fixer avec le client les critères d'acceptation par classe, le taux de correction acceptable et le délai produit ; aucune valeur numérique n'est présentée comme une exigence déjà fournie.
- Tester les erreurs, la charge et les pointes de trafic sur l'infrastructure cible ; mesurer mémoire réelle, clic-vers-affichage et coût opérationnel.
- Garder le contrôle humain et un retour manuel tant que ces validations ne sont pas concluantes. Un score de probabilité de la régression n'est pas une garantie de confiance ; tout mécanisme d'abstention doit être évalué et calibré séparément.

## Livrables de mission 1

Le [notebook unique](../notebooks/01_analyse_et_comparaison.ipynb) regroupe les quatre étapes. Le support de recommandation est conservé dans `livrables/` (13 diapositives). Le suivi des critères de la fiche décrit les preuves disponibles sans cocher les cases à la place de l'étudiant. Les intégrations sont documentées séparément dans les bilans de missions 2 et 3 ; l'ajout de Terra ne les modifie pas.
'''
    (ROOT/'docs/etape_4_recommandation.md').write_text(report,encoding='utf-8')
    print('Synthèse calculée, budgets et rapport produits ; aucun appel API ni entraînement.',flush=True)
    print(pd.DataFrame(setup).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
