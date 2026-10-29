"""Audit complet en flux et partitions reproductibles, sans chargement du corpus en RAM."""

from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split


TEXT = "Consumer Claim"
LABEL = "Tag"
ID = "Complaint ID"
DATE = "Date received"
EMPTY_MARKERS = {"nan", "none", "null", "n/a", "na", "not provided", "no narrative"}


def normalize_text(value):
    # Garder ponctuation, négations, nombres, casse et marqueurs d'anonymisation.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def has_content(text):
    if text.casefold() in EMPTY_MARKERS:
        return False
    without_masks = re.sub(r"\b[xX]{2,}\b", "", text)
    return bool(re.search(r"[^\W\d_]", without_masks, flags=re.UNICODE))


def text_fingerprint(text):
    return hashlib.sha256(normalize_text(text).casefold().encode("utf-8")).hexdigest()


def _save_counter(counter, columns):
    return pd.DataFrame([(*key, value) for key, value in counter.items()], columns=columns)


def prepare_corpus(source, output, chunksize=25_000):
    """Recalcule les artefacts. Le CSV d'origine reste la référence inchangée."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    db = output / "complaints.sqlite"
    missing, labels_all, labels_text = Counter(), Counter(), Counter()
    temporal, categories, audit, issues = Counter(), Counter(), Counter(), Counter()
    first_date, last_date = None, None
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("PRAGMA temp_store=MEMORY")
        # Ces tables sont exclusivement des artefacts régénérables de cette fonction.
        for table in ("claims", "source_ids", "text_groups", "partitions"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("CREATE TABLE source_ids (id TEXT PRIMARY KEY)")
        conn.execute("""CREATE TABLE claims (
            source_record INTEGER PRIMARY KEY, complaint_id TEXT, label TEXT,
            received TEXT, text TEXT, text_key TEXT)""")
        for chunk in pd.read_csv(source, dtype=str, keep_default_na=False, chunksize=chunksize):
            assert {TEXT, LABEL, ID, DATE}.issubset(chunk.columns), "Colonnes requises absentes"
            audit["lignes_source"] += len(chunk)
            stripped = {col: chunk[col].str.strip() for col in chunk.columns}
            missing.update({col: int(values.eq("").sum()) for col, values in stripped.items()})
            for col in (LABEL, "Submitted via", "Timely response?", "Consumer disputed?", "State"):
                if col in chunk:
                    categories.update({(col, str(k)): int(v) for k, v in stripped[col].value_counts().items()})
            label = stripped[LABEL]
            text = chunk[TEXT].map(normalize_text)
            with_text = text.ne("")
            labels_all.update(label.value_counts().to_dict())
            labels_text.update(label[with_text].value_counts().to_dict())

            received = pd.to_datetime(stripped[DATE], format="%m/%d/%Y", errors="coerce")
            issues["dates_reception_invalides_non_vides"] += int((received.isna() & stripped[DATE].ne("")).sum())
            if received.notna().any():
                lo, hi = received.min(), received.max()
                first_date = lo if first_date is None else min(first_date, lo)
                last_date = hi if last_date is None else max(last_date, hi)
            if "Date sent to company" in chunk:
                sent = pd.to_datetime(stripped["Date sent to company"], format="%m/%d/%Y", errors="coerce")
                issues["dates_envoi_invalides_non_vides"] += int((sent.isna() & stripped["Date sent to company"].ne("")).sum())
                issues["envoi_avant_reception"] += int((sent < received).sum())
            for col in ("Timely response?", "Consumer disputed?"):
                if col in chunk:
                    issues[f"valeurs_inattendues_{col}"] += int((~stripped[col].isin(["", "Yes", "No"])).sum())
            year = received.dt.year.astype("Int64").astype(str).replace("<NA>", "Date invalide/absente")
            counts = pd.DataFrame({"annee": year, "label": label, "avec_texte": with_text}).value_counts()
            temporal.update({key: int(value) for key, value in counts.items()})

            valid_id = stripped[ID].str.fullmatch(r"\d+")
            ids = stripped[ID][valid_id].tolist()
            before = conn.total_changes
            conn.executemany("INSERT OR IGNORE INTO source_ids VALUES (?)", ((x,) for x in ids))
            issues["identifiants_dupliques"] += len(ids) - (conn.total_changes - before)
            # Motifs exclusifs : leur somme permet de réconcilier toutes les lignes.
            masks = [
                ("sans_texte", ~with_text),
                ("identifiant_invalide", ~valid_id),
                ("sans_etiquette", label.eq("")),
                ("sans_contenu_lexical", ~text.map(has_content)),
            ]
            keep = pd.Series(True, index=chunk.index)
            for reason, mask in masks:
                audit[reason] += int((keep & mask).sum())
                keep &= ~mask
            records = []
            for idx in chunk.index[keep]:
                value = text.at[idx]
                date = received.at[idx]
                records.append((int(idx) + 1, stripped[ID].at[idx], label.at[idx],
                                None if pd.isna(date) else date.date().isoformat(),
                                value, text_fingerprint(value)))
            conn.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?)", records)
            audit["candidats_avec_texte"] += len(records)
            conn.commit()

        if issues["identifiants_dupliques"]:
            raise ValueError("Identifiants dupliqués : résoudre leur ambiguïté avant de créer les partitions.")
        conn.execute("CREATE INDEX claims_key ON claims(text_key)")
        conn.execute("""CREATE TABLE text_groups AS SELECT text_key,
            MIN(source_record) AS first_record, COUNT(*) AS n_rows,
            COUNT(DISTINCT label) AS n_labels FROM claims GROUP BY text_key""")
        conn.execute("CREATE UNIQUE INDEX group_key ON text_groups(text_key)")
        audit["textes_contradictoires"] = conn.execute(
            "SELECT COALESCE(SUM(n_rows),0) FROM text_groups WHERE n_labels>1").fetchone()[0]
        audit["doublons_coherents"] = conn.execute(
            "SELECT COALESCE(SUM(n_rows-1),0) FROM text_groups WHERE n_labels=1").fetchone()[0]
        metadata = pd.read_sql_query("""SELECT c.source_record, c.complaint_id, c.label,
            c.received, c.text_key, LENGTH(c.text) AS n_chars FROM claims c
            JOIN text_groups g ON c.source_record=g.first_record WHERE g.n_labels=1
            ORDER BY c.source_record""", conn)
    audit["lignes_nettoyees"] = len(metadata)
    excluded = ["sans_texte", "identifiant_invalide", "sans_etiquette", "sans_contenu_lexical",
                "textes_contradictoires", "doublons_coherents"]
    assert audit["lignes_source"] == len(metadata) + sum(audit[k] for k in excluded)
    assert metadata["text_key"].is_unique and metadata["complaint_id"].is_unique
    columns = pd.DataFrame({"colonne": list(missing), "manquants": list(missing.values())})
    columns["taux_manquants_pct"] = 100 * columns["manquants"] / audit["lignes_source"]
    distribution = pd.DataFrame({"source": pd.Series(labels_all),
                                  "avec_texte": pd.Series(labels_text),
                                  "nettoye": metadata["label"].value_counts()}).fillna(0).astype(int)
    distribution.index.name = "label"
    distribution["couverture_texte_pct"] = 100 * distribution["avec_texte"] / distribution["source"]
    dates = _save_counter(temporal, ["annee", "label", "avec_texte", "nombre"])
    cats = _save_counter(categories, ["colonne", "valeur", "nombre"])
    columns.to_csv(output / "quality_by_column.csv", index=False)
    distribution.to_csv(output / "label_distribution.csv")
    dates.to_csv(output / "temporal_label_counts.csv", index=False)
    cats.to_csv(output / "categorical_values.csv", index=False)
    pd.Series(audit, name="nombre").to_csv(output / "cleaning_audit.csv")
    source_hash = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            source_hash.update(block)
    manifest = {"source": source.name, "bytes": source.stat().st_size,
                "sha256": source_hash.hexdigest(), "audit": dict(audit), "issues": dict(issues),
                "received_min": None if first_date is None else first_date.date().isoformat(),
                "received_max": None if last_date is None else last_date.date().isoformat(),
                "target": LABEL, "input": TEXT, "label_mapping": "original_labels_no_merging",
                "normalization": "Unicode NFC, whitespace; casefold for duplicate fingerprint only",
                "placeholder_markers": sorted(EMPTY_MARKERS)}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata, columns, distribution, dates, cats, manifest


def create_partitions(metadata, output, seed=42, common_size=1000):
    """70 % train, 15 % validation, 15 % test ; aucun rééquilibrage des classes."""
    assert metadata["text_key"].is_unique and metadata["complaint_id"].is_unique
    if metadata["label"].value_counts().min() < 7:
        raise ValueError("Classe trop rare pour trois partitions : définir un protocole explicite.")
    train, holdout = train_test_split(metadata.index, test_size=.30,
                                    stratify=metadata["label"], random_state=seed)
    validation, test = train_test_split(holdout, test_size=.50,
                                      stratify=metadata.loc[holdout, "label"], random_state=seed)
    result = metadata.copy()
    result["partition"] = ""
    for name, ids in (("train", train), ("validation", validation), ("test", test)):
        result.loc[ids, "partition"] = name
    assert result["partition"].ne("").all()
    assert result.groupby("label")["partition"].nunique().eq(3).all()
    # Le sous-ensemble commun est représentatif, il ne garantit pas une ligne par classe rare.
    test_rows = result.loc[test]
    if common_size < len(test_rows):
        _, selected = train_test_split(test_rows.index, test_size=common_size,
                                      stratify=test_rows["label"], random_state=seed)
        common = result.loc[selected].sort_values("source_record")
    else:
        common = test_rows.sort_values("source_record")
    output = Path(output)
    result.to_csv(output / "partitions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    common.to_csv(output / "test_common_ids.csv", index=False)
    with closing(sqlite3.connect(output / "complaints.sqlite")) as conn:
        result[["source_record", "partition"]].to_sql("partitions", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX partition_record ON partitions(source_record)")
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"] = {"seed": seed, "ratio": [0.70, 0.15, 0.15],
                         "counts": result["partition"].value_counts().to_dict(),
                         "common_test_size": len(common),
                         "common_test_absent_labels": sorted(set(result["label"]) - set(common["label"]))}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, common


def read_partition(output, partition="train", limit=None):
    """Charger les textes d'une partition quand l'étape de modélisation le nécessite."""
    if partition not in {"train", "validation", "test"}:
        raise ValueError("Partition inconnue")
    query = """SELECT c.complaint_id, c.label, c.text FROM claims c
        JOIN partitions p ON c.source_record=p.source_record WHERE p.partition=?
        ORDER BY c.source_record"""
    params = [partition]
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit doit être un entier positif")
        query += " LIMIT ?"
        params.append(limit)
    with closing(sqlite3.connect(Path(output) / "complaints.sqlite")) as conn:
        return pd.read_sql_query(query, conn, params=params)
