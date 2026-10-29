from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from llm_evaluation import (BudgetError, EvaluationStore, classify_one, export_results, metrics, request_payload, run_experiment)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.base = (ROOT/'artifacts/test_tmp').resolve()
        assert self.base.is_relative_to(ROOT)
        self.base.mkdir(parents=True, exist_ok=True)
        self.model = {'id': 'vendor/model', 'pricing': {'prompt': '0.0000001', 'completion': '0.0000005'},
                      'supported_parameters': ['max_tokens', 'response_format', 'reasoning'],
                      'reasoning': {'mandatory': False, 'supported_efforts': ['none', 'low']}}

    def test_budget_and_uncertain_call_block_before_sending(self):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            assert Path(tmp).resolve().is_relative_to(ROOT)
            store = EvaluationStore(Path(tmp)/'cache.sqlite', .01)
            store.reserve('one', .008)
            with self.assertRaises(BudgetError):
                store.reserve('one', .008)
            with self.assertRaises(BudgetError):
                store.reserve('two', .008)

    def test_resume_does_not_call_api_again(self):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            assert Path(tmp).resolve().is_relative_to(ROOT)
            store = EvaluationStore(Path(tmp)/'cache.sqlite', 1)
            client = Mock()
            client.chat.completions.create.return_value.model_dump.return_value = {
                'model': 'vendor/model', 'choices': [{'finish_reason': 'stop', 'message': {'content': '{"label":"A"}'}}],
                'usage': {'cost': .0001, 'prompt_tokens': 10, 'completion_tokens': 5}}
            row = next(pd.DataFrame({'complaint_id': ['1'], 'text': ['a complaint'], 'label': ['A']}).itertuples(index=False))
            first = classify_one(client, store, self.model, 'base', 'Classify', row, ['A','B'], 'development')
            second = classify_one(client, store, self.model, 'base', 'Classify', row, ['A','B'], 'development')
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first['status'], 'valid')
            self.assertEqual(first['cost_usd'], .0001)

    def test_request_avoids_unsupported_temperature_and_caps_prices(self):
        request = request_payload(self.model, 'Classify', 'text', ['A','B'])
        self.assertNotIn('temperature', request)
        self.assertEqual(request['extra_body']['reasoning'], {'effort': 'none'})
        self.assertAlmostEqual(request['extra_body']['provider']['max_price']['prompt'], .1)

    def test_failures_stay_in_quality_and_latency_scores(self):
        records = [
            {'expected':'A','prediction':'A','status':'valid','response_ms':100,'cost_usd':.1,'prompt_tokens':2,'completion_tokens':1},
            {'expected':'B','prediction':None,'status':'error','response_ms':1000,'cost_usd':None,'prompt_tokens':None,'completion_tokens':None},
        ]
        result = metrics(records, ['A','B','C'])
        json.dumps(result)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['coverage'], .5)
        self.assertAlmostEqual(result['p95_ms'], 955)
        self.assertEqual(result['unknown_cost_calls'], 1)
        self.assertEqual(result['absent_labels'], 'C')


    def test_cache_write_price_is_reserved_before_sending(self):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            model = {**self.model, 'pricing': {**self.model['pricing'], 'input_cache_write': '0.001'}}
            store = EvaluationStore(Path(tmp)/'cache.sqlite', .01)
            client = Mock()
            row = next(pd.DataFrame({'complaint_id':['1'], 'text':['Complaint'], 'label':['A']}).itertuples(index=False))
            with self.assertRaises(BudgetError):
                classify_one(client, store, model, 'base', 'Classify', row, ['A'], 'test')
            client.chat.completions.create.assert_not_called()

    def test_adding_a_model_preserves_old_results_and_never_replays_them(self):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            root = Path(tmp)
            out = root/'artifacts/step2'
            out.mkdir(parents=True)
            store = EvaluationStore(out/'requests.sqlite', 1)
            old_model = {**self.model, 'id':'openai/gpt-6-luna'}
            new_model = {**self.model, 'id':'openai/gpt-6.1-terra'}
            response = {'model':'mock', 'choices':[{'finish_reason':'stop','message':{'content':'{"label":"A"}'}}],
                        'usage':{'cost':.0001,'prompt_tokens':10,'completion_tokens':5}}
            old_client = Mock()
            old_client.chat.completions.create.return_value.model_dump.return_value = response
            old_row = next(pd.DataFrame({'complaint_id':['old'], 'text':['Old complaint'], 'label':['A']}).itertuples(index=False))
            old = classify_one(old_client, store, old_model, 'base', 'Classify', old_row, ['A','B'], 'test')
            export_results(out, [old], ['A','B'], expected_test=1)
            development = pd.DataFrame({'complaint_id':['dev'], 'text':['Development'], 'label':['A']})
            test = pd.DataFrame({'complaint_id':['new'], 'text':['New complaint'], 'label':['A']})
            prompts = {'labels_only':'Classify', 'definitions':'Classify with definitions', 'few_shot':'Classify with examples'}
            client = Mock()
            client.chat.completions.create.return_value.model_dump.return_value = response
            with patch('llm_evaluation.load_local_env', return_value=True), patch.dict('os.environ', {'OPENROUTER_API_KEY':'test-only'}), patch('openai.OpenAI', return_value=client):
                run_experiment(root, development, test, ['A','B'], prompts, [new_model], budget=1, model_ids=[new_model['id']])
            self.assertEqual(client.chat.completions.create.call_count, 4)
            frame = pd.read_csv(out/'predictions.csv')
            self.assertEqual(len(frame), 5)
            self.assertIn(old['key'], set(frame.key))
            preserved = next(record for record in store.records() if record['key']==old['key'])
            self.assertEqual(preserved, old)
            client.reset_mock()
            with patch('llm_evaluation.load_local_env', return_value=True), patch.dict('os.environ', {'OPENROUTER_API_KEY':'test-only'}), patch('openai.OpenAI', return_value=client):
                run_experiment(root, development, test, ['A','B'], prompts, [new_model], budget=1, model_ids=[new_model['id']])
            client.chat.completions.create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
