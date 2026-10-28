# ZenAssist — mission 3, classification ML

Cette branche `mission-3-ml` réunit l’application [JavaScript/Next.js](javascript/README.md) et le [service Python/FastAPI](ml-service/README.md) dans le même dépôt. La variante `typescript/` est le starter officiel inchangé et ne contient pas cette intégration. La version LLM de mission 2 est conservée sur `main`, commit `f41c052`.

## Démarrage local

À la première installation, depuis ce dossier `application/` :

```powershell
cd javascript
npm.cmd ci
if (!(Test-Path .env.local)) { Copy-Item .env.example .env.local }
cd ..
```

Trois processus sont nécessaires, dans trois terminaux PowerShell :

```powershell
# Terminal 1 — PostgreSQL (inutile s’il tourne déjà)
cd javascript
npm.cmd run db:local
```

```powershell
# Terminal 2 — depuis la racine de ce dépôt application/
cd ml-service
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv sync --locked
uv run --locked python -m scripts.export_model
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 3 — depuis la racine de ce dépôt application/
cd javascript
npm.cmd run db:init
npm.cmd run dev -- --hostname 127.0.0.1 --port 3100
```

Ouvrir [l’application](http://127.0.0.1:3100), sélectionner un ticket, cliquer **Auto-tag**, puis **Apply suggestion**. La proposition et le temps de réponse s’affichent avant confirmation. **Dismiss** abandonne ; **Assign tag** permet la saisie manuelle.

Le serveur Next.js utilise `ML_API_URL=http://127.0.0.1:8000` par défaut ; aucune clé fournisseur n’est nécessaire. FastAPI n’accède pas à PostgreSQL. `db:init` conserve les tickets existants. Si le port de PostgreSQL est déjà occupé par cette même base, garder le processus existant et ne pas lancer un second `db:local`. Arrêter chaque processus avec `Ctrl+C`.

La première exécution de `uv sync` peut télécharger Python 3.13.5. Les dépendances restent dans `ml-service/.venv` ; l’environnement du notebook n’est pas modifié.

## Tests

- `ml-service/` : `uv run --locked pytest -q`.
- `javascript/` : `npm.cmd test`, puis `npm.cmd run lint`.
- Avec les trois processus démarrés : `npm.cmd run test:integration` et `npm.cmd run test:e2e` (Edge requis).
- Vérifier directement le service : [état du modèle](http://127.0.0.1:8000/health), [documentation interactive](http://127.0.0.1:8000/docs), route `POST /tags`.

Pour tester l’indisponibilité, arrêter le terminal Python puis cliquer **Auto-tag** : un message doit s’afficher et l’étiquette doit rester inchangée. Relancer Python pour rétablir l’analyse.

Le [workflow de release](.github/workflows/release-model.yml) exporte le pipeline entraîné en `model.pkl`, teste l’API puis joint le modèle, ses métadonnées et ses métriques à une release GitHub publiée. Le lancement manuel produit un artefact téléchargeable sans créer de release. Le workflow devra être poussé sur votre propre dépôt GitHub ; aucun push ni publication n’a été effectué ici.
