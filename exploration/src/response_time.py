"""Chronométrer une demande de classification complète, une réclamation à la fois."""
from time import perf_counter

import numpy as np
import pandas as pd


def benchmark_response_time(predict_one, samples, *, is_valid):
    """Retourner les mesures et un résumé en ms, en conservant les erreurs.

    samples contient complaint_id et text. predict_one englobe la préparation du
    texte/prompt, la prédiction ou requête réseau, les reprises éventuelles et le
    décodage. Le chargement initial et l'entraînement restent hors du chronométrage.
    Aucun appel d'échauffement ni nouvelle tentative n'est ajouté par cet utilitaire.
    """
    if samples.empty:
        raise ValueError("Le lot de mesure est vide")
    if samples['complaint_id'].duplicated().any():
        raise ValueError("Le lot doit contenir des identifiants uniques")
    rows = []
    for complaint_id, text in samples[['complaint_id', 'text']].itertuples(index=False, name=None):
        prediction = None
        started = perf_counter()
        error_type = ''
        try:
            prediction = predict_one(text)
            status = 'valide' if is_valid(prediction) else 'format_invalide'
        except Exception as exc:
            status, error_type = 'erreur', type(exc).__name__
        elapsed_ms = (perf_counter() - started) * 1000
        rows.append({'complaint_id': complaint_id, 'n_chars': len(text),
                     'prediction': prediction, 'response_ms': elapsed_ms,
                     'statut': status, 'erreur_type': error_type})
    measurements = pd.DataFrame(rows)
    valid = measurements['statut'].eq('valide')
    summaries = []
    for scope, subset in [('toutes_les_demandes', measurements),
                          ('reponses_valides', measurements.loc[valid])]:
        durations = subset['response_ms']
        summaries.append({
            'perimetre': scope, 'n': len(subset),
            'moyenne_ms': durations.mean() if len(subset) else np.nan,
            'mediane_ms': durations.median() if len(subset) else np.nan,
            'p95_ms': durations.quantile(.95) if len(subset) else np.nan,
            'maximum_ms': durations.max() if len(subset) else np.nan,
        })
    summary = pd.DataFrame(summaries)
    summary['taux_sortie_valide'] = valid.mean()
    summary['taux_erreur'] = measurements['statut'].eq('erreur').mean()
    summary['taux_format_invalide'] = measurements['statut'].eq('format_invalide').mean()
    return measurements, summary
