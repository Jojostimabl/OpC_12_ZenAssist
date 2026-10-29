# Mission 1, étape 4 — Synthèse et recommandation

Analyse mise à jour le **28 octobre 2026**, avec GPT-6.1 Terra mesuré le 28 octobre et les mesures initiales ML/LLM du **19 octobre 2026**. Ce script de synthèse ne lance aucun entraînement ni appel payant : il relit les résultats enregistrés.

## Décision proposée

**Privilégier la régression logistique pondérée comme aide à l'étiquetage, avec confirmation humaine.**

- Le choix ML a été figé sur le F1 macro de validation : `logistic_regression__balanced`.
- Sur les mêmes 1 000 tickets que les LLM, elle obtient un F1 macro de **0.557**, une accuracy de **72.6 %**, une médiane de **1.761 ms** et un p95 de **3.387 ms**. Le meilleur F1 macro LLM est **0.475** (`openai/gpt-6.1-terra`) ; la meilleure accuracy LLM est **70.6 %** (`openai/gpt-6.1-terra`). Les p95 LLM vont de 1.94 à 6.91 secondes.
- Le SVM est proche en F1 macro (0,556) et a une meilleure accuracy (76,5 %). Il reste une alternative si le besoin métier privilégie les résultats sur le volume global. Ce constat ne remplace pas le critère de sélection fixé ; valider tout changement sur de nouvelles données indépendantes.
- **274 tickets sur 1 000 sont mal classés** par le ML retenu. Ce résultat ne justifie pas une orientation totalement automatique. L'utilisateur doit pouvoir confirmer ou corriger l'étiquette.
- Les mesures ML portent sur le pipeline chargé en mémoire, les mesures LLM incluent le réseau. Le délai clic-vers-affichage et le comportement sous charge restent à tester.

Le test commun couvre 17 catégories. `Virtual currency` n'y est pas évaluée ; deux exemples dans le test complet ne suffisent pas à fiabiliser son score. Les données sans texte et les tickets futurs restent hors du périmètre de preuve. Ne pas fusionner les libellés historiques sans décision métier ni table de correspondance.

## Produit et scénario pédagogique

La recommandation produit découle des scores et latences observés. Le scénario OpenClassrooms impose néanmoins une **mission 2 avec intégration LLM**, puis une **mission 3 avec remplacement par le ML**. Le pilote LLM sera donc une étape pédagogique de transition ; il ne prouve pas que le LLM est le meilleur choix final.

Le pilote de mission 2 a été intégré avec Qwen à partir de la comparaison initiale. Après ajout de Terra, le meilleur F1 macro observé revient à `openai/gpt-6.1-terra`, la meilleure accuracy à `openai/gpt-6.1-terra` et le coût API unitaire le plus bas à `qwen/qwen3.8-flash`. Cette extension de comparaison ne modifie pas le code des intégrations existantes. Aucun seuil d'attente acceptable n'est imposé dans la fiche ; l'établir avec le client avant de figer un choix produit. Les coûts observés ne sont pas remplacés par un dépassement fictif pour reproduire le scénario pédagogique.

Les conditions de réseau/fournisseur des sessions du 19 et du 28 octobre diffèrent ; les latences ne sont pas des mesures simultanées. Les prompts, partitions et identifiants restent identiques, et le nouveau prompt est choisi exclusivement sur le développement.

## Coût d'exploitation mensuel

Hypothèses : 30 jours, pilote de 1 000 tickets/jour (30 000/mois), scénario global de 40 000 tickets/jour (1 200 000/mois, base de 200 clients), un appel par ticket, mêmes prompts, textes, cache et coûts d'inférence que lors de la mesure du modèle (19 ou 28 octobre). Les prix ne sont pas garantis à la date de cette synthèse.

| Scénario | Modèle | Tickets/mois | API, USD | Serveur ML ajouté, USD | Budget avec maintenance, USD |
|---|---|---:|---:|---:|---:|
| pilot | deepseek/deepseek-v4-flash-0731 | 30,000 | 2.61 | 0–0 | 102.61–202.61 |
| pilot | openai/gpt-6-luna | 30,000 | 2.58 | 0–0 | 102.58–202.58 |
| pilot | openai/gpt-6.1-terra | 30,000 | 26.12 | 0–0 | 126.12–226.12 |
| pilot | qwen/qwen3.8-flash | 30,000 | 2.18 | 0–0 | 102.18–202.18 |
| pilot | logistic_regression | 30,000 | 0.00 | 6–12 | 156.00–312.00 |
| global | deepseek/deepseek-v4-flash-0731 | 1,200,000 | 104.51 | 0–0 | 204.51–304.51 |
| global | openai/gpt-6-luna | 1,200,000 | 103.10 | 0–0 | 203.10–303.10 |
| global | openai/gpt-6.1-terra | 1,200,000 | 1044.94 | 0–0 | 1144.94–1244.94 |
| global | qwen/qwen3.8-flash | 1,200,000 | 87.35 | 0–0 | 187.35–287.35 |
| global | logistic_regression | 1,200,000 | 0.00 | 6–12 | 156.00–312.00 |

La colonne de budget inclut une **hypothèse de maintenance** de 2–4 h/mois pour le LLM et 3–6 h/mois pour le ML, valorisées à **50 USD/h**, convention modifiable et non tarif convenu. Elle ne constitue pas une dépense mesurée. Les plages se recouvrent : l'avantage économique global n'est pas établi par la seule absence de facture API ML.

Référence serveur vérifiée le 25 octobre : [DigitalOcean Basic Regular](https://www.digitalocean.com/pricing/droplets), 1 vCPU/1 GiB à 6 USD/mois ou 1 vCPU/2 GiB à 12 USD/mois. **Scénario budgétaire à valider en charge**, sans promesse que ces tailles suffisent. L'entraînement reste séparé du serveur d'inférence. Une ressource existante peut éviter ce serveur supplémentaire si sa capacité est vérifiée.

Coûts communs exclus : hébergement de l'application, base de données et authentification existantes. Taxes, recharge OpenRouter, options de stockage/sauvegarde/observabilité, incidents majeurs et nouvelle collecte d'annotations ne sont pas compris. Les coûts API suivent la [facturation d'usage OpenRouter](https://openrouter.ai/docs/faq), avec frais d'achat de crédits distincts.

Au pilote, les API seules coûtent environ 2.18–26.12 USD/mois, à comparer au serveur ML supplémentaire de 6–12 USD. Au volume global, les API seules représentent environ 87.35–1044.94 USD/mois ; le budget serveur ML pourrait être inférieur si sa capacité suffit. Les coûts humains et la qualité restent déterminants.

## Temps et coût de mise en place

**Estimations de travail à affiner après lecture du dépôt d'intégration.** Un développeur connaît l'application, les accès sont disponibles, l'authentification existe et les données/modèles de mission 1 sont prêts. Les heures excluent refonte métier, achats, attente d'accès et audit réglementaire ; elles ne décrivent pas le temps de calcul de l'entraînement.

| Travail | LLM, h | ML, h |
|---|---:|---:|
| Client fournisseur / API Python et chargement du pipeline | 3–5 | 4–6 |
| Validation et erreurs LLM / export du modèle et CI de release | 3–5 | 4–8 |
| Interface React, chargement et confirmation humaine | 2–4 | 2–4 |
| Tests, déploiement du pilote et vérification produit | 4–6 | 6–10 |
| Suivi, documentation et transmission à l'équipe | 2–4 | 4–6 |
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
