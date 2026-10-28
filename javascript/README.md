# Application JavaScript — mission 3 ML

Cette branche remplace l’intégration LLM par le service Python. La version OpenRouter est conservée sur `main`, commit `f41c052`. Le [guide commun](../README.md) décrit les trois processus à démarrer ; le [guide FastAPI](../ml-service/README.md) décrit l’API et l’export.

## Lancer l’application

Depuis `application/javascript` :

```powershell
# Première installation uniquement
npm.cmd ci
# Uniquement si .env.local est absent
if (!(Test-Path .env.local)) { Copy-Item .env.example .env.local }
```

Le fichier privé existe déjà sur le poste : ne pas le remplacer. La variable `ML_API_URL=http://127.0.0.1:8000` est lue par le serveur Next.js, avec cette adresse comme défaut. Les anciennes variables privées de fournisseur, si elles existent localement, sont inutilisées. Aucune clé API n’est requise.

Démarrer PostgreSQL dans un terminal (`npm.cmd run db:local`, ou `npm.cmd run db:start` si Docker fonctionne), puis FastAPI dans `../ml-service` selon son guide. Ne pas démarrer deux bases sur le même port.

```powershell
npm.cmd run db:init
npm.cmd run dev -- --hostname 127.0.0.1 --port 3100
```

Ouvrir [l’application](http://127.0.0.1:3100). `db:init` crée et peuple la table uniquement si elle est absente ; il préserve les tickets existants. `db:reset` est une commande destructive de réinitialisation volontaire.

## Parcours de test

1. Sélectionner un ticket dans **Untagged**.
2. Cliquer **Auto-tag** : chargement visible, bouton désactivé.
3. Vérifier la proposition **ML suggestion** et son temps de réponse. À ce stade le ticket reste sans étiquette.
4. Cliquer **Apply suggestion** : l’étiquette est enregistrée dans PostgreSQL et le ticket quitte **Untagged**. Le retrouver dans sa boîte ou **All Claims**.
5. Sur un autre ticket, essayer **Dismiss**, puis **Assign tag** pour vérifier l’abandon et la saisie manuelle.
6. Arrêter FastAPI avec `Ctrl+C` et réessayer **Auto-tag** : un message d’indisponibilité apparaît sans changer l’étiquette. Redémarrer le service puis réessayer.

La réponse affiche le temps mesuré dans le navigateur, jusqu’à réception de la suggestion ; le rendu visuel peut ajouter un faible délai. La route serveur retourne aussi `elapsedMs` (appel interne HTTP) et `inferenceMs` (inférence Python), ainsi que `modelVersion` pour la traçabilité.

## Fonctionnement

`Claim.jsx` → `POST /api/claims/[id]/suggest-tag` → `POST http://127.0.0.1:8000/tags` avec `user_claim`. Next.js lit le texte complet en base, valide l’étiquette parmi les 18 originales et renvoie la proposition. La confirmation utilise séparément `PUT /api/claims/[id]/tag`.

Le modèle Python est chargé une seule fois. Timeout d’appel interne : 10 s ; timeout navigateur : 15 s. Les demandes simultanées du même ticket sont regroupées, deux tickets distincts maximum par processus. Les erreurs de réseau, HTTP, JSON ou catégorie invalide restent affichées sans enregistrement. Aucun repli vers un fournisseur externe.

Le SDK, les prompts et les modules d’intégration fournisseur ont été retirés de cette version. Next.js 15.5.28, React 19.1.10, CSS Modules et pilote PostgreSQL `pg` sont conservés. La variante `../typescript/` reste le starter inchangé.

## Vérifications automatiques

```powershell
npm.cmd test
npm.cmd run lint
# Base, Python et application démarrés
npm.cmd run test:integration
npm.cmd run test:e2e
```

Les tests navigateur nécessitent Edge. Quatre scénarios simulent la réponse de la route de suggestion pour tester les erreurs ; un cinquième utilise réellement FastAPI et le modèle. Les écritures PostgreSQL sont réelles et limitées à un ticket synthétique créé puis supprimé par les tests. Le contrôle d’intégration effectue 10 requêtes locales et écrit `artifacts/mission3/smoke-live.json` dans le projet d’étude.

## Build de production

```powershell
# Dossier séparé pour pouvoir garder le mode développement ouvert
$env:NEXT_DIST_DIR = '.next-build'
npm.cmd run build
npm.cmd run start -- --hostname 127.0.0.1 --port 3101
```

L’application de production est alors sur le port 3101. Utiliser la même valeur `NEXT_DIST_DIR` lors du build et du start. Dans un nouveau terminal sans cette variable, le mode développement garde son dossier `.next`. Le starter télécharge ses polices Google lors du premier build.

Le prototype reste limité à la boucle locale, sans authentification comme le starter. Ajouter authentification et contrôle des droits avant exposition publique.
