# Service ML — FastAPI

Le service expose le pipeline **TF-IDF + régression logistique pondérée** sélectionné sur validation en mission 1. Les 18 étiquettes originales sont conservées. Le modèle est chargé et vérifié une seule fois au démarrage ; aucun apprentissage, téléchargement de modèle ou appel fournisseur n’a lieu pendant une prédiction.

## Installer et lancer

Depuis `application/ml-service` :

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv sync --locked
uv run --locked python -m scripts.export_model
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Ces commandes suffisent depuis un clone du dépôt : le pipeline source de 8 Mo est inclus dans `trained/`, sans dataset, accès au notebook ou dépendance à un chemin du poste d’origine. Le modèle exporté et ses JSON sont placés dans `models/` (ignoré par Git).

- [GET /health](http://127.0.0.1:8000/health) : `status=ready`, modèle, version et nombre de catégories.
- [Interface de test /docs](http://127.0.0.1:8000/docs) : ouvrir `POST /tags`, **Try it out**, saisir un corps JSON puis **Execute**.

```json
{"user_claim": "There is an incorrect account on my credit report."}
```

Réponse : `tag`, `model`, `modelVersion`, `elapsedMs`. `elapsedMs` mesure l’inférence et la préparation du résultat dans Python ; l’application affiche séparément la durée côté navigateur jusqu’à réception de la proposition.

Test direct en PowerShell :

```powershell
$body = @{ user_claim = 'There is an incorrect account on my credit report.' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tags -ContentType 'application/json' -Body $body
```

## Contrat et erreurs

- `user_claim` doit être une chaîne non vide, de 100 000 caractères maximum, avec du texte exploitable ; aucun champ supplémentaire. JSON invalide ou données invalides : HTTP 422, sans recopier le texte dans la réponse d’erreur.
- La normalisation reprend Unicode NFC et le regroupement des espaces de mission 1, en conservant ponctuation, casse, nombres et négations.
- Jusqu’à deux prédictions simultanées et deux threads de calcul. Quand ces deux places sont occupées : HTTP 503, sans ajouter de file d’attente dans le modèle.
- Une exception d’inférence donne HTTP 500 et un message neutre. Les textes ne sont pas journalisés.
- Si l’artefact manque, si son empreinte est incorrecte ou si les dépendances sont incompatibles, le démarrage échoue ; le service n’accepte pas de trafic avec un modèle non chargé.

L’API est locale, sans authentification et sans CORS navigateur : Next.js l’appelle côté serveur. Une exposition publique nécessiterait authentification, limites persistantes et configuration réseau adaptées.

## Modèle et export

`trained/selected_pipeline.joblib` est l’artefact de mission 1 ; `trained/manifest.json` fixe son SHA-256, sa provenance, les paramètres, les catégories et les versions d’entraînement. `trained/metrics.json` reprend les résultats existants, sans les réestimer à chaque export.

Le script `python -m scripts.export_model` vérifie la provenance et l’environnement, exporte un véritable pickle standard, puis compare les prédictions source/export sur six textes synthétiques. Il produit :

- `models/model.pkl` : pipeline complet, pas le classifieur seul.
- `models/metadata.json` : empreintes, version, catégories, environnement et provenance.
- `models/metrics.json` : métriques de mission 1 et périmètre d’évaluation.

Python 3.13 et les versions exactes de scikit-learn, NumPy, SciPy et joblib sont conservés. `uv.lock` fixe également les dépendances API et de test. Les formats pickle/joblib pouvant exécuter du code au chargement, utiliser exclusivement les fichiers de notre dépôt ou de nos releases approuvées ; une empreinte vérifie l’intégrité, pas l’identité d’un fournisseur inconnu.

Pour promouvoir un nouveau modèle depuis le projet d’étude complet : `uv run python scripts/prepare_mission3_training.py` à la racine du projet d’étude, puis contrôler les nouveaux artefacts et tests avant de les versionner. Ce script de promotion n’est pas requis pour servir ou republier le modèle déjà inclus.

## Tests et publication

```powershell
uv run --locked pytest -q
```

Les tests couvrent une vraie prédiction, la normalisation, l’export, les entrées invalides, le refus d’un modèle altéré/incompatible, le chargement unique, la saturation et les erreurs internes. Leurs fichiers temporaires restent dans `.test-temp/`.

Le workflow `../.github/workflows/release-model.yml` s’exécute sur `release: published` et `workflow_dispatch`. L’environnement est restauré sous Linux, l’export est exécuté, les tests passent avant tout téléversement, puis `model.pkl`, `metadata.json` et `metrics.json` sont joints à la release. Sans événement de release, seul l’artefact Actions est créé. Aucun écrasement silencieux d’un asset existant. Le workflow n’a pas été exécuté sur GitHub depuis ce poste.

Pour changer de version en exploitation, arrêter le service, installer les trois fichiers d’une même release approuvée avec la version de code correspondante et son `uv.lock`, puis redémarrer. Garder la release précédente pour revenir en arrière. Le service ne se met pas à jour automatiquement.
