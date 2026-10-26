# ZenAssist — mission 2, intégration LLM

Application JavaScript/React du dépôt officiel [OpenClassrooms dev-ia-P12](https://github.com/OpenClassrooms-Student-Center/dev-ia-P12), base `23c0f63e1cc589488cb1de3574fdebcf4da77a55`. Next.js 15.5.28, React 19.1.10, PostgreSQL, SDK OpenAI compatible OpenRouter. La variante TypeScript reste inchangée.

## Démarrage sous Windows

Dans `application/javascript`, avec Node.js 24 et npm :

```powershell
npm.cmd ci
Copy-Item .env.example .env.local
```

Renseigner `OPENROUTER_API_KEY` dans `.env.local`. Ce fichier privé existe déjà sur le poste : ne pas le remplacer. Ne jamais préfixer cette clé par `NEXT_PUBLIC_`.

Choisir un seul mode PostgreSQL :

```powershell
# Docker Desktop, moteur démarré
npm.cmd run db:start

# OU PostgreSQL local dans .local-postgres, sans Docker.
# Garder ce terminal ouvert ; Ctrl+C arrête le serveur et préserve les données.
npm.cmd run db:local
```

Dans un autre terminal :

```powershell
npm.cmd run db:init
npm.cmd run build
npm.cmd run start -- --hostname 127.0.0.1 --port 3100
```

Ouvrir [l’application](http://127.0.0.1:3100). `db:init` charge les 10 tickets du starter seulement si la table est absente. Toute table existante est préservée. `db:reset`, hérité du starter, **efface les tickets** : uniquement pour une réinitialisation volontaire. En développement : `npm.cmd run dev -- --hostname 127.0.0.1 --port 3100` à la place de build/start.

Les deux modes utilisent la base isolée `zenassist_mission2`, port `55432`, sur `127.0.0.1`. Le mode local utilise PostgreSQL 18.4 ; Docker conserve PostgreSQL 14.18 du starter. Pilote `pg` et requêtes SQL identiques. Ne pas démarrer les deux modes simultanément.

## Utilisation

1. Sélectionner un ticket.
2. Cliquer sur **Auto-tag**, dans `Claim.jsx`. Attente visible, bouton désactivé, requête asynchrone.
3. Relire la proposition. **Apply suggestion** enregistre ; **Dismiss** abandonne. **Assign tag** reste disponible pour l’attribution manuelle.

La proposition seule ne modifie jamais la base. Réponses malformées, étiquettes inconnues, erreurs HTTP et délais dépassés produisent un message visible. Un échec d’enregistrement conserve la proposition pour réessayer sans nouvel appel LLM.

## Modèle et prompt

`OPENROUTER_MODEL=qwen/qwen3.8-flash` par défaut. Les autres configurations évaluées sont `deepseek/deepseek-v4-flash-0731` et `openai/gpt-6-luna`. Un modèle inconnu est refusé, sans remplacement silencieux.

Les prompts sélectionnés en mission 1 sont repris **à l’identique**, avec les 18 étiquettes d’origine. `src/server/llm-config.json` contient leurs textes, empreintes et paramètres. Régénération depuis la racine : `uv run python scripts/export_mission2_config.py`.

Le serveur lit le texte complet du ticket en PostgreSQL, appelle OpenRouter et valide strictement le JSON `{ "label": "..." }`. Limite de 100 000 caractères, sans troncature silencieuse. La réponse interne contient l’étiquette, le modèle, la durée en millisecondes et le coût USD s’il est fourni. Les plafonds de prix sont ceux du catalogue de mission 1 : une hausse ou un paramètre non pris en charge entraîne une erreur explicite.

La clé et le SDK restent côté serveur (`server-only`). Timeout fournisseur : 45 s, aucune relance automatique payante ; timeout navigateur : 55 s. Les appels simultanés sur le même ticket partagent une requête ; au maximum deux tickets sont analysés simultanément par processus.

## Vérifications

```powershell
npm.cmd test
npm.cmd run lint
npm.cmd run build
npm.cmd audit
# Base et application démarrées ; Edge installé
npm.cmd run test:e2e
# Un appel réel facturable, à lancer volontairement
node scripts/smoke-llm.js
```

Les tests navigateur simulent uniquement les réponses de la route LLM pour ne rien facturer ; lecture et enregistrement utilisent PostgreSQL. Ils créent puis suppriment leur propre ticket synthétique. Le contrôle réel sauvegarde son résultat dans `artifacts/mission2/smoke-live.json`, à la racine du projet.

## Périmètre

Prototype local pédagogique, sans authentification comme le starter. Avant exposition publique : authentification, droits, limitation persistante des appels, suivi des coûts et politique de données. Les données envoyées à OpenRouter quittent le poste ; les essais réels utilisent les tickets publics de démonstration. Aucun entraînement ou évaluation de mission 1 n’est relancé. L’intégration ML de mission 3 reste à réaliser.
