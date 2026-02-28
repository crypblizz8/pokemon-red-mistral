from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pokemon.campaign_log import append_campaign_log_entry, campaign_log_report


class CampaignLogTests(unittest.TestCase):
    def test_report_defaults_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "campaign_log.json"
            report = campaign_log_report(path)
            self.assertEqual(report["real_battles"], 0)
            self.assertEqual(report["simulations"], 0)
            self.assertEqual(report["combined"], 0)
            self.assertEqual(report["entries"], 0)

    def test_append_updates_totals_and_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "campaign_log.json"
            append_campaign_log_entry(
                path,
                kind="real_battle",
                count=3,
                source="phase1",
                metadata={"model": "mistral-large-latest"},
            )
            append_campaign_log_entry(
                path,
                kind="simulation",
                count=7,
                source="phase1_eval_loso",
            )
            report = campaign_log_report(path)
            self.assertEqual(report["real_battles"], 3)
            self.assertEqual(report["simulations"], 7)
            self.assertEqual(report["combined"], 10)
            self.assertEqual(report["entries"], 2)


if __name__ == "__main__":
    unittest.main()
