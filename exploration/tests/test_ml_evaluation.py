from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ml_evaluation import quality_scores, select_candidates


class MLProtocolTests(unittest.TestCase):
    def test_selection_refuses_test_and_ignores_unconverged_candidate(self):
        rows = [
            {'phase':'validation','model':'svc','candidate':'bad','converged':False,
             'f1_macro_present':.99,'accuracy':.99},
            {'phase':'validation','model':'svc','candidate':'ok','converged':True,
             'f1_macro_present':.7,'accuracy':.8},
        ]
        self.assertEqual(select_candidates(rows).iloc[0].candidate,'ok')
        rows[1]['phase']='test'
        with self.assertRaises(ValueError): select_candidates(rows)

    def test_common_score_excludes_absent_class_and_keeps_failed_predictions(self):
        result = quality_scores(['A','B'],['A','__NO_PREDICTION__'],['A','B','C'])
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['f1_macro_present'],.5)
        self.assertEqual(result['absent_labels'],'C')


if __name__=='__main__':
    unittest.main()
