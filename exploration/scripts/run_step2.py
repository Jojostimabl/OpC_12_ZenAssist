"""Préparer l'expérience sans frais, ou exécuter les appels avec --run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from llm_evaluation import (MODEL_IDS, MAX_OUTPUT_TOKENS, estimate_costs, load_local_env,
                            prepare_experiment, run_experiment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Envoyer les demandes payantes à OpenRouter')
    parser.add_argument('--models', nargs='+', choices=MODEL_IDS, help='Évaluer seulement ces modèles et conserver les autres résultats')
    args = parser.parse_args()
    key_present = load_local_env(ROOT)
    out = ROOT / 'artifacts/step2'
    catalog = json.loads((ROOT / 'config/llm_models.json').read_text(encoding='utf-8'))
    dev, test, labels, prompts = prepare_experiment(ROOT)
    costs = estimate_costs(dev, test, labels, prompts, catalog)
    costs.to_csv(out / 'estimated_costs.csv', index=False)
    budget = float(__import__('os').getenv('STEP2_BUDGET_USD', '2.00'))
    manifest = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'models': MODEL_IDS, 'development_examples': len(dev), 'test_examples': len(test),
        'prompts': list(prompts), 'max_output_tokens': MAX_OUTPUT_TOKENS,
        'budget_usd': budget, 'max_concurrent_requests': 3, 'sdk_retries': 0,
        'run_models': args.models or MODEL_IDS, 'requests_per_model_in_flight': 1,
        'full_text': True, 'reasoning': 'disabled_when_supported',
        'temperature': '0_only_when_supported', 'input_token_estimation': 'UTF8_bytes/4 to UTF8_bytes/3, plus protocol margin',
        'cost_estimate_low_usd': costs['estimate_low_usd'].sum(),
        'cost_estimate_high_usd': costs['estimate_high_usd'].sum(),
        'source_manifest_sha256': hashlib.sha256((ROOT/'artifacts/step1/dataset_manifest.json').read_bytes()).hexdigest(),
        'partitions_sha256': hashlib.sha256((ROOT/'artifacts/step1/partitions.csv.gz').read_bytes()).hexdigest(),
        'common_ids_sha256': hashlib.sha256((ROOT/'artifacts/step1/test_common_ids.csv').read_bytes()).hexdigest(),
        'labels_without_test_support': sorted(set(labels)-set(test['label'])),
    }
    extension_path = out/'terra_extension.json'
    if extension_path.exists():
        manifest['extension'] = json.loads(extension_path.read_text(encoding='utf-8'))
        for relative, expected in manifest['extension']['frozen_sha256'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
    (out / 'protocol.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(costs.to_string(index=False), flush=True)
    print(f'Total estime : ${costs.estimate_low_usd.sum():.3f} a ${costs.estimate_high_usd.sum():.3f}', flush=True)
    print(f'Cle disponible : {key_present} | reservation maximale : ${budget:.2f}', flush=True)
    if args.run:
        run_experiment(ROOT, dev, test, labels, prompts, catalog, budget=budget, model_ids=args.models)
        subprocess.run([sys.executable, str(ROOT/'scripts/finalize_step2.py')], check=True)
    else:
        print('Preparation seule : aucun appel payant.', flush=True)


if __name__ == '__main__':
    main()
