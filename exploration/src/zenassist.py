"""Préparation et évaluation reproductibles du jeu de réclamations."""

from collections import Counter, defaultdict
import csv
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


LABEL_MAP = {
    "Bank account or service": "Compte bancaire",
    "Checking or savings account": "Compte bancaire",
    "Credit card": "Carte et prépayé",
    "Credit card or prepaid card": "Carte et prépayé",
    "Prepaid card": "Carte et prépayé",
    "Credit reporting": "Rapport de crédit",
    "Credit reporting, credit repair services, or other personal consumer reports": "Rapport de crédit",
    "Debt collection": "Recouvrement",
    "Mortgage": "Crédit immobilier",
    "Student loan": "Prêt étudiant",
    "Vehicle loan or lease": "Prêt automobile",
    "Consumer Loan": "Prêt personnel",
    "Payday loan": "Prêt personnel",
    "Payday loan, title loan, or personal loan": "Prêt personnel",
    "Money transfers": "Transfert d'argent",
    "Money transfer, virtual currency, or money service": "Transfert d'argent",
    "Virtual currency": "Transfert d'argent",
    "Other financial service": "Autre service financier",
}


def normalize_claim(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_sample(path: str | Path, per_class: int = 5000, seed: int = 42):
    """Scan complet, échantillonnage réservoir équilibré et dédoublonnage."""
    rng = random.Random(seed)
    profile = Counter()
    source_counts = Counter()
    eligible_counts = Counter()
    reservoirs = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            profile["total"] += 1
            raw_label = (row.get("Tag") or "").strip()
            source_counts[raw_label] += 1
            label = LABEL_MAP.get(raw_label)
            claim = normalize_claim(row.get("Consumer Claim") or "")
            if not claim:
                profile["sans_texte"] += 1
                continue
            if label is None:
                profile["libelle_inconnu"] += 1
                continue
            if len(claim) < 40:
                profile["texte_trop_court"] += 1
                continue
            eligible_counts[label] += 1
            item = {"id": row["Complaint ID"], "texte": claim, "categorie": label}
            bucket = reservoirs[label]
            if len(bucket) < per_class:
                bucket.append(item)
            else:
                index = rng.randrange(eligible_counts[label])
                if index < per_class:
                    bucket[index] = item

    sample = pd.DataFrame(item for bucket in reservoirs.values() for item in bucket)
    sample["cle_texte"] = sample["texte"].str.casefold()
    conflicting = sample.groupby("cle_texte")["categorie"].nunique()
    conflicts = set(conflicting[conflicting > 1].index)
    profile["doublons_ambigus_ecartes"] = int(sample["cle_texte"].isin(conflicts).sum())
    sample = sample.loc[~sample["cle_texte"].isin(conflicts)]
    profile["doublons_identiques_ecartes"] = int(sample.duplicated("cle_texte").sum())
    sample = sample.drop_duplicates("cle_texte").drop(columns="cle_texte").reset_index(drop=True)
    profile["echantillon_final"] = len(sample)
    return sample, profile, source_counts, eligible_counts


def score_predictions(y_true, y_pred, labels):
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }
