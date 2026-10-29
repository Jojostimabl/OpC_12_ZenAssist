# Mission 1 — exploration et comparaison LLM / ML

Auteur : Joseph Truffinet. Projet démarré en octobre 2026.

Le notebook unique `notebooks/01_analyse_et_comparaison.ipynb` contient les sorties
enregistrées de toute la mission : préparation, quatre LLM (Terra compris), trois
algorithmes ML et référence majoritaire, erreurs, temps de réponse, coûts et recommandation.
Il est consultable directement sur GitHub sans installation ni clé API.

Les modules `src/`, scripts d'exécution, tests, paramètres et dépendances verrouillées
sont inclus, ainsi que les résultats CSV/JSON et les graphiques utilisés par le notebook.
Les applications sont à la racine du dépôt : `main` pour le LLM, `mission-3-ml` pour le ML.

## Réexécution locale

1. Télécharger le dataset OpenClassrooms et le placer **ici**, sous `dataset.csv` :
   https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/2464_D%C3%A9veloppeur_IA/P12/dataset.csv
   Empreinte SHA-256 attendue : `80db47ac8e817bfd1f0588f785a847f03d5f3052f244c0e47bdf8f312e7b52bc`.
2. Depuis ce dossier : `uv sync --locked`, puis `uv run jupyter lab`.
3. Ouvrir le notebook et exécuter les cellules dans l'ordre. La première passe régénère
   automatiquement la base SQLite et les partitions manquantes depuis le CSV.
   Cette préparation complète peut prendre plusieurs minutes et crée des fichiers locaux volumineux.
4. Les étapes LLM/ML lisent les résultats publiés : le notebook ne déclenche aucun appel
   fournisseur ni nouvel entraînement. Les scripts `run_step2.py` et `run_step3.py`
   servent aux exécutions explicites ; le premier nécessite une clé et peut être payant.
5. La toute dernière cellule vérifie le support de présentation livré séparément :
   placer sa version finale dans `livrables/ZenAssist_Mission1_Recommandation.pptx`
   sous ce dossier pour satisfaire cette vérification. Le PowerPoint est en cours de mise à jour
   et ne fait pas partie de cette publication du code.

Le CSV de 639 Mo, la base SQLite, les environnements, clés privées et caches sont exclus.
Les sorties enregistrées sont conservées telles quelles. Les mesures historiques datent du
19 octobre 2026, Terra du 28 octobre ; les latences proviennent de sessions distinctes.

## Tests

`uv run python -m unittest discover -s tests -v`

Le contenu publié est décrit et vérifiable dans `publication_manifest.json`.
