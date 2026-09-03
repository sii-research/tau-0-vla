"""Regression coverage for flow-matching time sampling."""

from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import torch


_UTILS_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "tau0_vla"
    / "models"
    / "vision_language_action_models"
    / "tau0_vla"
    / "utils.py"
)
_SPEC = importlib.util.spec_from_file_location("tau0_vla_utils_under_test", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)
sample_beta = _UTILS.sample_beta


class SampleBetaTest(unittest.TestCase):
    def test_samples_match_beta_moments(self) -> None:
        alpha, beta = 1.5, 1.0
        torch.manual_seed(20260903)
        samples = sample_beta(alpha, beta, 100_000, "cpu")

        expected_mean = alpha / (alpha + beta)
        expected_variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

        self.assertAlmostEqual(samples.mean().item(), expected_mean, delta=0.01)
        self.assertAlmostEqual(samples.var(unbiased=False).item(), expected_variance, delta=0.01)
        self.assertTrue(torch.all((samples >= 0) & (samples <= 1)))

    def test_sample_shape_and_device_are_preserved(self) -> None:
        samples = sample_beta(2.0, 3.0, 17, torch.device("cpu"))

        self.assertEqual(samples.shape, (17,))
        self.assertEqual(samples.device.type, "cpu")
        self.assertEqual(samples.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
