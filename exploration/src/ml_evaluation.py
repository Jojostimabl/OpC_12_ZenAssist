"""Mission 1, étape 3 : sélection sur validation et évaluation ML locale."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_info, threadpool_limits

from data_preparation import read_partition
from response_time import benchmark_response_time


SEED = 42
VECTORIZER_PARAMS = dict(ngram_range=(1, 2), min_df=3, max_df=.995,
                         max_features=100_000, sublinear_tf=True, strip_accents='unicode',
                         lowercase=True, stop_words=None, dtype=np.float32)
CANDIDATES = [
    ('dummy', 'majority', {}),
    ('complement_nb', 'alpha_0.1', {'alpha': .1}),
    ('complement_nb', 'alpha_1', {'alpha': 1.0}),
    ('logistic_regression', 'unweighted', {'C': 1.0, 'class_weight': None}),
    ('logistic_regression', 'balanced', {'C': 1.0, 'class_weight': 'balanced'}),
    ('linear_svc', 'unweighted', {'C': 1.0, 'class_weight': None}),
    ('linear_svc', 'balanced', {'C': 1.0, 'class_weight': 'balanced'}),
]


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_estimator(family, params):
    if family == 'dummy':
        return DummyClassifier(strategy='most_frequent', random_state=SEED)
    if family == 'complement_nb':
        return ComplementNB(**params)
    if family == 'logistic_regression':
        return LogisticRegression(solver='lbfgs', max_iter=300, tol=1e-3, random_state=SEED, **params)
    if family == 'linear_svc':
        return LinearSVC(dual='auto', max_iter=3000, random_state=SEED, **params)
    raise ValueError(family)


def quality_scores(expected, predicted, labels):
    present = sorted(set(expected))
    return {'n': len(expected), 'accuracy': accuracy_score(expected, predicted),
            'f1_macro_present': f1_score(expected, predicted, labels=present, average='macro', zero_division=0),
            'f1_weighted': f1_score(expected, predicted, labels=labels, average='weighted', zero_division=0),
            'absent_labels': '|'.join(sorted(set(labels)-set(present)))}


def select_candidates(rows):
    """Aucune prédiction de test n'est acceptée par cette fonction de sélection."""
    table = pd.DataFrame(rows)
    if not table.phase.eq('validation').all():
        raise ValueError('La sélection accepte seulement les métriques de validation.')
    if not table.converged.all():
        table = table[table.converged]
    return table.sort_values(['f1_macro_present', 'accuracy'], ascending=False, kind='stable').drop_duplicates('model')


def load_common(root, test):
    common = pd.read_csv(Path(root)/'artifacts/step1/test_common_ids.csv', dtype={'complaint_id': str})
    result = common[['complaint_id', 'label']].merge(test, on='complaint_id', suffixes=('_reference', ''), validate='one_to_one')
    assert len(result) == 1000 and result.label.eq(result.label_reference).all()
    return result.drop(columns='label_reference').sort_values('complaint_id').reset_index(drop=True)


def check_sources(root):
    root = Path(root)
    previous = json.loads((root/'artifacts/step2/protocol.json').read_text(encoding='utf-8'))
    files = {'source_manifest_sha256': 'dataset_manifest.json', 'partitions_sha256': 'partitions.csv.gz',
             'common_ids_sha256': 'test_common_ids.csv'}
    for field, name in files.items():
        assert digest(root/'artifacts/step1'/name) == previous[field], f'Les données ont changé : {name}'
    return {field: previous[field] for field in files}


def export_class_reports(out, model, scope, expected, predicted, labels):
    present = sorted(set(expected))
    report = pd.DataFrame(classification_report(expected, predicted, labels=present,
                          output_dict=True, zero_division=0)).T
    for absent in set(labels)-set(present):
        report.loc[absent] = [0 if c == 'support' else np.nan for c in report.columns]
    report.to_csv(out/f'classes_{scope}_{model}.csv')
    matrix_labels = labels+['__NO_PREDICTION__']
    matrix = confusion_matrix(expected, predicted, labels=matrix_labels)
    pd.DataFrame(matrix, index=matrix_labels, columns=matrix_labels).to_csv(out/f'confusion_{scope}_{model}.csv')


def run_experiment(root):
    root = Path(root)
    out = root/'artifacts/step3'
    out.mkdir(parents=True, exist_ok=True)
    sources = check_sources(root)
    experiment = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'sources': sources,
                  'seed': SEED, 'fit_partition': 'train', 'selection_partition': 'validation',
                  'selection_metric': 'f1_macro_present', 'tie_breaker': 'accuracy_then_candidate_order',
                  'refit_train_plus_validation': False, 'full_text': True,
                  'vectorizer': {k: ('float32' if k == 'dtype' else v) for k,v in VECTORIZER_PARAMS.items()},
                  'candidates': CANDIDATES, 'threads': 2, 'code_sha256': digest(Path(__file__)),
                  'python': platform.python_version(), 'sklearn': sklearn.__version__,
                  'numpy': np.__version__, 'scipy': scipy.__version__,
                  'machine': platform.machine(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
                  'external_api_calls': 0}
    signature = hashlib.sha256(json.dumps({k:v for k,v in experiment.items() if k!='created_at_utc'},
                                         sort_keys=True, default=json_default).encode()).hexdigest()
    old_path = out/'protocol.json'
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding='utf-8'))
        if old.get('signature') != signature:
            raise RuntimeError('Un autre protocole existe dans artifacts/step3 ; conserver ses résultats avant de le remplacer.')
        if (out/'completion.json').exists():
            print('Expérience déjà complète : résultats conservés, aucun entraînement relancé.', flush=True)
            return
    experiment['signature'] = signature
    save_json(old_path, experiment)
    started = perf_counter()
    print('Chargement train et validation (textes complets)...', flush=True)
    train = read_partition(root/'artifacts/step1', 'train')
    validation = read_partition(root/'artifacts/step1', 'validation')
    assert train.complaint_id.is_unique and validation.complaint_id.is_unique
    assert set(train.complaint_id).isdisjoint(validation.complaint_id)
    labels = sorted(train.label.unique())
    vectorizer_path = out/'tfidf.joblib'
    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
        print('TF-IDF repris du cache du même protocole.', flush=True)
        train_matrix = vectorizer.transform(train.text)
        fit_vectorizer_seconds = json.loads((out/'vectorization.json').read_text(encoding='utf-8'))['fit_transform_train_s']
    else:
        print('Ajustement TF-IDF sur les 254 731 textes du train uniquement...', flush=True)
        vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
        begin = perf_counter()
        train_matrix = vectorizer.fit_transform(train.text)
        fit_vectorizer_seconds = perf_counter()-begin
        joblib.dump(vectorizer, vectorizer_path, compress=3)
        save_json(out/'vectorization.json', {'fit_transform_train_s': fit_vectorizer_seconds,
                  'shape': train_matrix.shape, 'nonzero': train_matrix.nnz,
                  'csr_bytes': train_matrix.data.nbytes+train_matrix.indices.nbytes+train_matrix.indptr.nbytes,
                  'vocabulary_size': len(vectorizer.vocabulary_)})
    begin = perf_counter()
    validation_matrix = vectorizer.transform(validation.text)
    print(f'TF-IDF : {train_matrix.shape}, {train_matrix.nnz:,} valeurs non nulles ; ajustement {fit_vectorizer_seconds:.1f} s.', flush=True)
    (out/'models').mkdir(exist_ok=True)
    rows = []
    existing = out/'validation_metrics.csv'
    saved = pd.read_csv(existing).to_dict('records') if existing.exists() else []
    estimators = {}
    with threadpool_limits(limits=2):
        save_json(out/'threadpools.json', threadpool_info())
        for family, name, params in CANDIDATES:
            candidate = f'{family}__{name}'
            model_path = out/'models'/f'{candidate}.joblib'
            old = [r for r in saved if r['candidate']==candidate]
            if old and model_path.exists():
                rows.append(old[0])
                print(f'{candidate} : métriques et modèle repris du cache.', flush=True)
                continue
            print(f'Entraînement : {candidate}', flush=True)
            clf = make_estimator(family, params)
            begin = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                clf.fit(train_matrix, train.label)
            fit_seconds = perf_counter()-begin
            warning_texts = [str(w.message) for w in caught]
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            prediction = clf.predict(validation_matrix)
            row = {'phase':'validation', 'model':family, 'candidate':candidate,
                   'parameters':json.dumps(params), 'fit_classifier_s':fit_seconds,
                   'fit_vectorizer_s':fit_vectorizer_seconds, 'converged':converged,
                   'n_iter_max':int(np.max(clf.n_iter_)) if hasattr(clf,'n_iter_') else 0,
                   'warnings':json.dumps(warning_texts), **quality_scores(validation.label,prediction,labels)}
            joblib.dump(clf, model_path, compress=3)
            rows.append(row)
            pd.DataFrame(rows).to_csv(existing,index=False)
            print(f'  {fit_seconds:.1f} s | F1 macro validation {row["f1_macro_present"]:.4f} | accuracy {row["accuracy"]:.4f} | convergence {converged}', flush=True)
        chosen = select_candidates(rows)
        assert len(chosen)==4, 'Une famille n’a pas de candidat convergé.'
        chosen.to_csv(out/'selected_models.csv',index=False)
        best = chosen[chosen.model.ne('dummy')].iloc[0]
        save_json(out/'selection.json', {'selection_partition':'validation', 'metric':'f1_macro_present',
                  'best_model':best.model, 'best_candidate':best.candidate,
                  'test_inspected_during_selection':False})
        print(f'Choix figé sur validation : {best.model}. Chargement du test.', flush=True)
        test = read_partition(root/'artifacts/step1','test')
        assert set(train.complaint_id).isdisjoint(test.complaint_id)
        assert set(validation.complaint_id).isdisjoint(test.complaint_id)
        common = load_common(root, test)
        test_matrix = vectorizer.transform(test.text)
        summaries, predictions = [], []
        for row in chosen.itertuples(index=False):
            clf = joblib.load(out/'models'/f'{row.candidate}.joblib')
            pipeline = Pipeline([('tfidf',vectorizer), ('classifier',clf)])
            begin = perf_counter()
            path = out/'models'/f'{row.model}_pipeline.joblib'
            joblib.dump(pipeline,path,compress=3)
            begin = perf_counter()
            loaded = joblib.load(path)
            load_seconds = perf_counter()-begin
            # L'appel unitaire utilise tout le pipeline et non une ligne prévectorisée.
            measurements, latency = benchmark_response_time(lambda text: loaded.predict([text])[0], common,
                                                            is_valid=lambda value: value in labels)
            measurements = measurements.merge(common[['complaint_id','label']],on='complaint_id',validate='one_to_one')
            measurements = measurements.rename(columns={'label':'expected','statut':'status'})
            measurements['model'] = row.model
            measurements.to_csv(out/f'predictions_common_{row.model}.csv',index=False)
            predictions.append(measurements)
            all_latency = latency[latency.perimetre.eq('toutes_les_demandes')].iloc[0]
            common_pred = measurements.prediction.fillna('__NO_PREDICTION__')
            summaries.append({'model':row.model,'candidate':row.candidate,'scope':'common_1000',
                              **quality_scores(measurements.expected,common_pred,labels),
                              'median_ms':all_latency.mediane_ms,'p95_ms':all_latency.p95_ms,
                              'mean_ms':all_latency.moyenne_ms,'max_ms':all_latency.maximum_ms,
                              'first_call_ms':measurements.response_ms.iloc[0],
                              'coverage':measurements.status.eq('valide').mean(),
                              'load_pipeline_s':load_seconds,'pipeline_bytes':path.stat().st_size,
                              'fit_classifier_s':row.fit_classifier_s,'fit_vectorizer_s':fit_vectorizer_seconds})
            export_class_reports(out,row.model,'common',measurements.expected,common_pred,labels)
            begin = perf_counter()
            full_prediction = clf.predict(test_matrix)
            batch_seconds = perf_counter()-begin
            full = test[['complaint_id','label']].rename(columns={'label':'expected'}).copy()
            full['prediction'] = full_prediction
            full.to_csv(out/f'predictions_full_{row.model}.csv',index=False)
            summaries.append({'model':row.model,'candidate':row.candidate,'scope':'full_test',
                              **quality_scores(test.label,full_prediction,labels),
                              'batch_classifier_s':batch_seconds})
            export_class_reports(out,row.model,'full',test.label,full_prediction,labels)
            print(f'Test {row.model} : F1 macro commun {summaries[-2]["f1_macro_present"]:.4f}, '
                  f'accuracy {summaries[-2]["accuracy"]:.4f}, p95 {all_latency.p95_ms:.2f} ms.',flush=True)
        pd.concat(predictions,ignore_index=True).to_csv(out/'predictions_common.csv',index=False)
        pd.DataFrame(summaries).to_csv(out/'test_metrics.csv',index=False)
    check_sources(root)
    save_json(out/'completion.json', {'completed_at_utc':datetime.now(timezone.utc).isoformat(),
              'wall_seconds_this_run':perf_counter()-started, 'train_size':len(train),
              'validation_size':len(validation),'full_test_size':len(test),'common_test_size':len(common),
              'chosen_ml':best.model, 'labels':labels, 'test_labels':sorted(common.label.unique()),
              'external_api_calls':0})
    print('Étape 3 : entraînements et évaluations terminés.',flush=True)

