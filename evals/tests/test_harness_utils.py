from __future__ import annotations

import unittest
from pathlib import Path

from evals.run_phase1_eval import build_loso_folds, ensure_output_root, fold_label


class HarnessUtilsTests(unittest.TestCase):
    def test_fold_label(self) -> None:
        self.assertEqual(fold_label(0), "A")
        self.assertEqual(fold_label(2), "C")
        self.assertEqual(fold_label(30), "F31")

    def test_build_loso_folds(self) -> None:
        paths = [
            Path("/tmp/a.state"),
            Path("/tmp/b.state"),
            Path("/tmp/c.state"),
        ]
        folds = build_loso_folds(paths)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(len(fold.train_states), 2)
            self.assertNotIn(fold.heldout_state, fold.train_states)

    def test_output_root_created_under_input_path(self) -> None:
        output = ensure_output_root(Path("/tmp/pokemon-red-mistral-evals-test"))
        self.assertTrue(output.exists())
        self.assertTrue(output.is_dir())


if __name__ == "__main__":
    unittest.main()
