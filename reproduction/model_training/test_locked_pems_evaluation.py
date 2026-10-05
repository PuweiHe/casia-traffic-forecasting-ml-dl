"""Regression checks that prevent opening test data before evaluation gates."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import locked_pems_evaluation as evaluation


class EvaluationGateTests(unittest.TestCase):
    def run_case(self, mode, provenance, collect_error=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'test_provenance_ledger.json').write_text(json.dumps({'status': provenance}))
            rows = [{'trial': trial, 'validation_mae_mph': score} for trial, score in
                    [('stgcn_reference', 2.0), ('stgcn_search_005', 1.8)]]
            with patch.object(evaluation, 'ROOT', root), patch.object(sys, 'argv', ['evaluation', mode]), \
                 patch.object(evaluation, 'collect', side_effect=collect_error, return_value=(rows, 'hash')), \
                 patch.object(evaluation, 'load_development') as load:
                with self.assertRaises(ValueError):
                    evaluation.main()
                load.assert_not_called()
                self.assertFalse((root / 'runs/final_pems_evaluation_v1/locked_selection.json').exists())

    def test_unreviewed_history_blocks_test_data(self):
        self.run_case('evaluate', 'provenance_review_required')

    def test_unfrozen_selection_blocks_test_data(self):
        self.run_case('evaluate', 'reviewed_independent_with_disclosed_scope')

    def test_missing_confirmation_blocks_freeze(self):
        self.run_case('freeze', 'reviewed_independent_with_disclosed_scope',
                      ValueError('Missing completed seed73 artifact'))


if __name__ == '__main__':
    unittest.main()
