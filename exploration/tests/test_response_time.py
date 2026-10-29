import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from response_time import benchmark_response_time


class ResponseTimeTests(unittest.TestCase):
    def test_durations_and_failed_requests_are_kept(self):
        samples = pd.DataFrame({'complaint_id': ['a', 'b', 'c'], 'text': ['valid', 'invalid', 'failure']})

        def predict(text):
            if text == 'failure':
                raise TimeoutError()
            return 'Mortgage' if text == 'valid' else 'unknown'

        # Demandes de 100 ms, 300 ms et 1 000 ms : pas d'attente réelle dans le test.
        with patch('response_time.perf_counter', side_effect=[0, .1, 1, 1.3, 2, 3]):
            measurements, summary = benchmark_response_time(predict, samples, is_valid=lambda x: x == 'Mortgage')
        self.assertEqual(measurements['statut'].tolist(), ['valide', 'format_invalide', 'erreur'])
        self.assertEqual(measurements.iloc[-1]['erreur_type'], 'TimeoutError')
        all_requests = summary.iloc[0]
        self.assertEqual(all_requests['n'], 3)
        self.assertAlmostEqual(all_requests['mediane_ms'], 300)
        self.assertAlmostEqual(all_requests['p95_ms'], 930)
        self.assertAlmostEqual(all_requests['taux_sortie_valide'], 1/3)
        self.assertAlmostEqual(summary.iloc[1]['p95_ms'], 100)

    def test_all_failed_has_no_fabricated_success_latency(self):
        samples = pd.DataFrame({'complaint_id': ['a'], 'text': ['failure']})
        with patch('response_time.perf_counter', side_effect=[0, 1]):
            _, summary = benchmark_response_time(lambda text: None, samples, is_valid=lambda x: False)
        self.assertEqual(summary.iloc[1]['n'], 0)
        self.assertTrue(pd.isna(summary.iloc[1]['p95_ms']))


if __name__ == '__main__':
    unittest.main()
