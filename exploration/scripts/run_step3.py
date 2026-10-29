"""Exécuter la comparaison ML locale ; aucune requête API."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ml_evaluation import run_experiment

if __name__=='__main__':
    run_experiment(ROOT)
