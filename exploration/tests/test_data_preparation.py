"""Vérifier les exclusions, doublons globaux et partitions sur un CSV synthétique."""
import csv
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_preparation import prepare_corpus, create_partitions


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1]
        self.temp_root = (self.workspace / 'artifacts' / 'test_tmp').resolve()
        assert self.temp_root.is_relative_to(self.workspace)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def write_fixture(self, root, extra):
        rows = [
            {'Complaint ID': str(1000 + 30 * c + i), 'Tag': label,
             'Consumer Claim': f'Claim for product {label} with distinct case {i}',
             'Date received': '05/10/2019'}
            for c, label in enumerate(['A', 'B', 'C']) for i in range(30)
        ]
        rows += [dict(zip(['Complaint ID', 'Tag', 'Consumer Claim', 'Date received'], x)) for x in extra]
        source = root / 'source.csv'
        with source.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return source

    def test_global_cleaning_and_reproducible_partitions(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root = Path(tmp).resolve()
            assert root.is_relative_to(self.workspace)
            source = self.write_fixture(root, [
                ('2000', 'A', 'Help', 'bad date'),
                ('2001', 'A', '  HELP\n ', '05/10/2019'),
                ('2002', 'A', 'Contradictory complaint', '05/10/2019'),
                ('2003', 'B', 'contradictory   complaint', '05/10/2019'),
                ('2004', 'A', ' XXXX XX 123 !!! ', '05/10/2019'),
                ('2005', 'A', '   ', '05/10/2019'),
                ('2006', '', 'Missing category', '05/10/2019'),
                ('bad', 'A', 'Valid claim invalid id', '05/10/2019'),
            ])
            output = root / 'result'
            meta, _, distribution, _, _, manifest = prepare_corpus(source, output, chunksize=7)
            self.assertEqual(len(meta), 91)
            self.assertEqual(meta['label'].nunique(), 3)
            self.assertEqual(distribution.loc['', 'source'], 1)
            self.assertEqual(manifest['audit']['doublons_coherents'], 1)
            self.assertEqual(manifest['audit']['textes_contradictoires'], 2)
            self.assertEqual(manifest['audit']['sans_contenu_lexical'], 1)
            self.assertEqual(manifest['issues']['dates_reception_invalides_non_vides'], 1)
            self.assertEqual(meta['n_chars'].min(), 4)
            self.assertTrue(meta['text_key'].is_unique)
            partitions, common = create_partitions(meta, output, common_size=6)
            again, common_again = create_partitions(meta, output, common_size=6)
            self.assertTrue(partitions.equals(again))
            self.assertTrue(common.equals(common_again))
            self.assertTrue(common['partition'].eq('test').all())
            self.assertTrue(partitions.groupby('label')['partition'].nunique().eq(3).all())
            with closing(sqlite3.connect(output / 'complaints.sqlite')) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM partitions').fetchone()[0], 91)

    def test_duplicate_ids_stop_preparation(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root = Path(tmp).resolve()
            assert root.is_relative_to(self.workspace)
            source = self.write_fixture(root, [('1000', 'A', 'Another complaint', '05/10/2019')])
            with self.assertRaisesRegex(ValueError, 'Identifiants dupliqués'):
                prepare_corpus(source, root / 'result', chunksize=7)


if __name__ == '__main__':
    unittest.main()
