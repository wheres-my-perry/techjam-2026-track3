from __future__ import annotations

import unittest

from tools.submission_preflight import ROOT, verify_submission


class SubmissionManifestTests(unittest.TestCase):
    def test_frozen_source_and_evidence_match_manifest(self) -> None:
        result = verify_submission(ROOT)
        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertEqual(result["locked_file_count"], 7)
        self.assertEqual(result["release_tag"], "techjam-2026-final-v16.1")


if __name__ == "__main__":
    unittest.main()
