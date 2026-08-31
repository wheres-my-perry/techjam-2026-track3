from __future__ import annotations

import unittest
from pathlib import Path


class RepositoryImportTests(unittest.TestCase):
    def test_active_entrypoint_uses_standalone_v16_1(self) -> None:
        import main
        from v16_1_clean import UserOptimizedTransformer

        self.assertIs(main.UserOptimizedTransformer, UserOptimizedTransformer)

    def test_tool_root_resolves_to_repository(self) -> None:
        from tools.matrix_runner import ROOT

        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])

    def test_v19_candidate_package_imports(self) -> None:
        from candidates.v19.cuda_fp16_checkpoint import UserOptimizedTransformer
        from candidates.v19.parallel_batch_v161 import (
            UserOptimizedTransformer as ParallelTransformer,
        )

        self.assertTrue(callable(UserOptimizedTransformer))
        self.assertTrue(callable(ParallelTransformer))


if __name__ == "__main__":
    unittest.main()
