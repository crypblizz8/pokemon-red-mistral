from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import run


def _ok_check(name: str) -> run.CheckResult:
    return run.CheckResult(name=name, passed=True, message="ok")


class _FakeEmulator:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def stop(self) -> None:
        return None


class _FakeAgent:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class Phase5RunModeTests(unittest.TestCase):
    def _build_args(self, tmpdir: str, extra: list[str] | None = None) -> object:
        parser = run.build_parser()
        results_path = Path(tmpdir) / "phase5_results.json"
        timeline_path = Path(tmpdir) / "phase5_timeline.jsonl"
        cmd = [
            "--phase5",
            "--phase5-policy-mode",
            "heuristic",
            "--phase5-start-state",
            str(Path(tmpdir) / "before_brock.state"),
            "--phase5-route-script",
            str(Path(tmpdir) / "route.json"),
            "--phase5-results-path",
            str(results_path),
            "--phase5-timeline-path",
            str(timeline_path),
        ]
        if extra:
            cmd.extend(extra)
        return parser.parse_args(cmd)

    def _fake_emulator_module(self) -> types.ModuleType:
        module = types.ModuleType("pokemon.emulator")
        module.PokemonEmulator = _FakeEmulator
        return module

    def test_phase5_strict_blocks_when_strength_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._build_args(tmpdir, extra=["--phase5-strength-check", "strict"])
            with patch("run.check_python_version", side_effect=lambda **_: _ok_check("python")), patch(
                "run.check_dependencies", side_effect=lambda: _ok_check("deps")
            ), patch(
                "run.check_file_exists_nonempty", side_effect=lambda *_: _ok_check("file")
            ), patch(
                "run.check_nav_state_ready", side_effect=lambda *_: _ok_check("nav")
            ), patch(
                "run.load_route_script", return_value={"name": "phase5_route", "steps": [1], "targets": {}}
            ), patch(
                "run.run_phase5_strength_probe",
                return_value={
                    "mode": "strict",
                    "min_level": 16,
                    "sample_runs": 10,
                    "required_pass_rate": 0.7,
                    "sample_size": 10,
                    "observed_pass_rate": 0.2,
                    "avg_turns": 11.0,
                    "active_level": 12,
                    "active_species_id": 176,
                    "party_species_ids": [176],
                    "meets_level_gate": False,
                    "meets_empirical_gate": False,
                    "passed": False,
                },
            ), patch("run.run_phase4_route") as mock_route_exec:
                rc = run.run_phase5(args)

            self.assertEqual(rc, 1)
            self.assertFalse(mock_route_exec.called)
            payload = json.loads(Path(args.phase5_results_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_reason"], "strength_gate_failed")
            self.assertEqual(payload["strength_gate"]["passed"], False)

    def test_phase5_warn_continues_and_marks_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._build_args(tmpdir, extra=["--phase5-strength-check", "warn"])
            with patch("run.check_python_version", side_effect=lambda **_: _ok_check("python")), patch(
                "run.check_dependencies", side_effect=lambda: _ok_check("deps")
            ), patch(
                "run.check_file_exists_nonempty", side_effect=lambda *_: _ok_check("file")
            ), patch(
                "run.check_nav_state_ready", side_effect=lambda *_: _ok_check("nav")
            ), patch(
                "run.load_route_script", return_value={"name": "phase5_route", "steps": [1], "targets": {}}
            ), patch(
                "run.run_phase5_strength_probe",
                return_value={
                    "mode": "warn",
                    "min_level": 16,
                    "sample_runs": 10,
                    "required_pass_rate": 0.7,
                    "sample_size": 10,
                    "observed_pass_rate": 0.6,
                    "avg_turns": 8.0,
                    "active_level": 15,
                    "active_species_id": 176,
                    "party_species_ids": [176],
                    "meets_level_gate": False,
                    "meets_empirical_gate": False,
                    "passed": False,
                },
            ), patch(
                "run.run_phase4_route",
                return_value={
                    "run_status": "success",
                    "target": "brock_badge",
                    "target_reached": True,
                    "party_constraint_passed": True,
                    "active_species_id_final": 176,
                    "active_level_final": 17,
                    "party_species_ids_final": [176],
                    "failure_reason": "",
                },
            ), patch(
                "run.MistralBattleAgent", _FakeAgent
            ), patch.dict(
                "sys.modules",
                {"pokemon.emulator": self._fake_emulator_module()},
            ):
                rc = run.run_phase5(args)

            self.assertEqual(rc, 0)
            payload = json.loads(Path(args.phase5_results_path).read_text(encoding="utf-8"))
            self.assertTrue(payload["strength_gate_warning"])
            self.assertEqual(payload["strength_gate_warning_reason"], "strength_gate_failed")
            self.assertEqual(payload["strength_gate"]["passed"], False)

    def test_phase5_success_writes_expected_result_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._build_args(tmpdir, extra=["--phase5-strength-check", "strict"])
            with patch("run.check_python_version", side_effect=lambda **_: _ok_check("python")), patch(
                "run.check_dependencies", side_effect=lambda: _ok_check("deps")
            ), patch(
                "run.check_file_exists_nonempty", side_effect=lambda *_: _ok_check("file")
            ), patch(
                "run.check_nav_state_ready", side_effect=lambda *_: _ok_check("nav")
            ), patch(
                "run.load_route_script", return_value={"name": "phase5_route", "steps": [1], "targets": {}}
            ), patch(
                "run.run_phase5_strength_probe",
                return_value={
                    "mode": "strict",
                    "min_level": 16,
                    "sample_runs": 10,
                    "required_pass_rate": 0.7,
                    "sample_size": 10,
                    "observed_pass_rate": 0.8,
                    "avg_turns": 7.5,
                    "active_level": 18,
                    "active_species_id": 176,
                    "party_species_ids": [176],
                    "meets_level_gate": True,
                    "meets_empirical_gate": True,
                    "passed": True,
                },
            ), patch(
                "run.run_phase4_route",
                return_value={
                    "run_status": "success",
                    "target": "brock_badge",
                    "target_reached": True,
                    "party_constraint_passed": True,
                    "active_species_id_final": 176,
                    "active_level_final": 18,
                    "party_species_ids_final": [176],
                    "failure_reason": "",
                },
            ), patch(
                "run.MistralBattleAgent", _FakeAgent
            ), patch.dict(
                "sys.modules",
                {"pokemon.emulator": self._fake_emulator_module()},
            ):
                rc = run.run_phase5(args)

            self.assertEqual(rc, 0)
            payload = json.loads(Path(args.phase5_results_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["required_species_id"], 176)
            self.assertIn("strength_gate", payload)
            self.assertTrue(payload["strength_gate"]["passed"])
            self.assertEqual(payload["target"], "brock_badge")


if __name__ == "__main__":
    unittest.main()
