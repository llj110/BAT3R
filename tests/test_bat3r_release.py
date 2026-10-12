"""CPU checks for namespace migration, numerical parity, and CLI dispatch."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from bat3r.bat3r_update import BayesianRecurrentUnit, UpdateController
from dust3r.bayesian_ttt import BayesianRecurrentUnit as LegacyUnit
from bat3r_eval import build_command


class BAT3RReleaseTests(unittest.TestCase):
    def test_legacy_import_and_state_dict(self):
        self.assertIs(LegacyUnit, BayesianRecurrentUnit)
        baseline = ROOT / "tests" / "_baseline_bayesian_ttt.py"
        if not baseline.exists():
            self.skipTest("Numerical parity requires a saved pre-migration module")
        spec = importlib.util.spec_from_file_location("baseline_bayesian", baseline)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        torch.manual_seed(7)
        original = old.BayesianRecurrentUnit(8, 8).eval()
        migrated = BayesianRecurrentUnit(8, 8).eval()
        migrated.load_state_dict(original.state_dict(), strict=True)
        mu, obs = torch.randn(2, 8, 5), torch.randn(2, 8, 5)
        var = torch.full_like(mu, 0.5)
        relevance = torch.sigmoid(torch.randn(2, 1, 5))
        for conf in [torch.rand(2, 3, 4) + 1, torch.rand(2, 1, 5) + 1, torch.rand(2, 5) + 1]:
            expected = original(mu, var, obs, conf, relevance)
            actual = migrated(mu, var, obs, conf, relevance)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(a).all())
            self.assertTrue((actual[1] > 0).all())

    def test_controller_confidence_layouts(self):
        ctl = UpdateController(8).eval()
        innovation = torch.rand(2, 1, 5)
        relevance = torch.ones(2, 1, 5)
        confidence = torch.rand(2, 5, 1) + 1
        a = ctl(innovation, confidence, relevance)
        b = ctl(innovation, confidence.transpose(1, 2), relevance)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
            self.assertEqual(x.shape, (2, 1, 5))

    def test_bat3r_and_legacy_model_share_classes(self):
        from bat3r.bat3r_model import BAT3R, BAT3RConfig
        from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig
        from bat3r.bat3r_inference import inference_recurrent_lighter
        from dust3r.inference import inference_recurrent_lighter as legacy_inference
        self.assertIs(BAT3R, ARCroco3DStereo)
        self.assertIs(BAT3RConfig, ARCroco3DStereoConfig)
        self.assertIs(inference_recurrent_lighter, legacy_inference)

    def test_dispatch_defaults_and_explicit_overrides(self):
        for task in ("relpose", "video_depth", "mv_recon", "ply"):
            _, launcher, forwarded = build_command(["--task", task])
            self.assertTrue(launcher.is_file())
            self.assertEqual(forwarded[forwarded.index("--model_update_type") + 1], "bat3r")
        _, _, forwarded = build_command(["--task", "relpose", "--model_update_type=cut3r", "--output_dir", "custom"])
        self.assertNotIn("--model_update_type", forwarded)
        self.assertEqual(forwarded.count("--output_dir"), 1)

    def test_checkpoint_location_does_not_hide_code(self):
        from add_ckpt_path import add_path_to_dust3r
        add_path_to_dust3r("/external/checkpoints/model.pth")
        self.assertIn(str(ROOT / "src"), sys.path)


if __name__ == "__main__":
    unittest.main()
