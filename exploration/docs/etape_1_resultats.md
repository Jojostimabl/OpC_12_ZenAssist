# Mission 1, étape 1 — Résultats du 17 octobre 2026

La partie exploration du [notebook unique](../notebooks/01_analyse_et_comparaison.ipynb) a été reprise et exécutée entièrement, avec graphiques et sorties enregistrées. Toutes les étapes de la mission sont désormais réunies dans ce même fichier. Deux tests vérifient le nettoyage entre blocs, les contradictions, la conservation des textes courts, la reproductibilité des partitions et le refus des identifiants dupliqués.

## Périmètre et choix

Entrée : `Consumer Claim`. Cible : `Tag`, avec ses **18 étiquettes d'origine**, sans fusion ou traduction. Le fichier source reste inchangé. L'ancien essai SVM sur 10 catégories est [archivé](../artifacts/history/01_analyse_et_comparaison_initial.json) ; ses scores ne concernent pas le protocole actuel.

Le corpus entier est lu par blocs et stocké en SQLite. Pas de plafond d'exemples par catégorie, de suppression automatique sous 40 caractères ou de troncature du texte. L'empreinte de dédoublonnage utilise Unicode NFC, les espaces normalisés et la casse neutralisée ; les textes des futurs modèles gardent leur casse et leur contenu.

## Bilan exclusif des lignes

| Traitement | Nombre |
|---|---:|
| Lignes source | 1 282 355 |
| Sans texte | 898 791 |
| Identifiant invalide ou étiquette absente parmi les lignes restantes | 0 |
| Sans contenu lexical | 4 |
| Enregistrements dont le texte porte plusieurs étiquettes | 1 924 |
| Doublons cohérents retirés | 17 734 |
| **Lignes conservées** | **363 902** |

Les 1 924 enregistrements contradictoires correspondent à 208 groupes. Leurs textes ne sont pas supprimés de la base locale d'audit ; ils sont exclus des partitions. Les doublons cohérents conservent le premier enregistrement rencontré. La somme des exclusions et des lignes conservées est vérifiée par assertion.

## Résultats exploratoires

- Le CSV comporte 15 colonnes ; 70,1 % des lignes n'ont pas de texte. La couverture varie selon les produits : il faut distinguer le flux de tickets entier et le corpus exploitable.
- 923 textes nettoyés ont moins de 40 caractères ; certains contiennent un signal explicite (« not my debt »). Un seuil arbitraire les aurait supprimés.
- La longueur médiane est de 771 caractères ; 2 356 textes dépassent 6 000 caractères et restent complets.
- Les effectifs nettoyés vont de 14 pour `Virtual currency` à 83 966 pour `Debt collection`.
- Aucun identifiant dupliqué ; les formats de dates non vides sont valides. En revanche, 7 052 dates d'envoi précèdent la réception : anomalie signalée, sans exclusion des textes puisque ces dates ne sont pas prédictives dans le protocole.
- L'historique va du 1er décembre 2011 au 10 mai 2019. Le graphique des textes montre les changements de nomenclature au fil des années : les catégories proches ne sont pas fusionnées automatiquement.

## Partitions

Séparation stratifiée sklearn, seed 42, sans rééquilibrage :

| Jeu | Lignes |
|---|---:|
| Entraînement, environ 70 % | 254 731 |
| Validation, environ 15 % | 54 585 |
| Test, environ 15 % | 54 586 |

Les 18 catégories sont présentes dans chacun des trois jeux. Les identifiants et empreintes de texte sont uniques dans le corpus nettoyé : les doublons normalisés ne traversent pas les partitions. Cela ne garantit pas l'absence de reformulations ou de modèles de lettre proches.

Le lot commun de 1 000 tickets est fixé dans le test et conserve ses proportions. Il ne comprend pas `Virtual currency`. Cette catégorie ne possède que 2 exemples dans le test complet : les futurs rapports devront expliciter ce manque de données et compléter le diagnostic sans annoncer une mesure fiable sur ces quelques exemples.

Le test est réservé à l'évaluation finale ; seuls des textes du train sont relus dans le notebook. Le TF-IDF et les réglages des modèles ne sont pas ajustés à cette étape. La séparation aléatoire ne constitue pas une preuve de généralisation chronologique.

## Métriques et artefacts

Le notebook définit le F1 macro, l'accuracy, le F1 pondéré, les scores par classe, les confusions, la validité des sorties, les erreurs techniques, les latences médiane/p95 et les coûts. Leur calcul sur des modèles appartient aux étapes suivantes.

Complément du 19 octobre 2026 : le notebook détaille maintenant le **temps de réponse par réclamation en millisecondes**, mesuré avec `time.perf_counter`, et son protocole. Il précise les périmètres ML/LLM, la médiane, le p95, la moyenne, les échecs et l'interprétation côté utilisateur. Le tableau reste « à mesurer » jusqu'aux étapes 2 et 3. Le délai clic-vers-affichage sera vérifié lors de l'intégration.

L'utilitaire `src/response_time.py` conserve durées, prédictions et erreurs et produit les statistiques sur toutes les demandes et sur les seules réponses valides. Il n'ajoute ni requête d'échauffement ni reprise automatique. Ses calculs et le maintien des erreurs dans le bilan sont vérifiés par des tests à horloge simulée, sans appel réseau.

Les fichiers régénérables sont dans `artifacts/step1/` : base SQLite, partitions compressées, lot commun, tableaux d'audit, distributions et manifeste. L'empreinte SHA-256 du source est `80db47ac8e817bfd1f0588f785a847f03d5f3052f244c0e47bdf8f312e7b52bc`.

Pour exécuter le notebook sans serveur Jupyter :

```powershell
.venv\Scripts\python.exe scripts/execute_notebook.py
```

Ce script utilise IPython pour les cellules et enregistre les sorties et graphiques dans le notebook. Il ne réalise aucun appel réseau.
