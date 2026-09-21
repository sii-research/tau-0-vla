"""Regression coverage for CLI overrides merged onto a YAML training config."""

import tempfile
import unittest
from pathlib import Path

from transformers.integrations import get_reporting_integration_callbacks

from tau0_vla.configs.training_config import TrainingArguments
from tau0_vla.utils.utils import load_config_from_yaml

_YAML = """\
experiment:
  run_name: overrides_probe
model_args:
  model_name_or_path: /nonexistent/checkpoint
data_args:
  config_name: overrides_probe
training_args:
  output_dir: outputs/
  report_to: none
"""


class CliOverrideTest(unittest.TestCase):
    def merged_training_args(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(_YAML)
            return load_config_from_yaml(str(path), overrides=overrides)["training_args"]

    def test_report_to_none_reaches_transformers_as_a_value_it_accepts(self):
        # scripts/validate_gpu_release.sh passes --report_to none. Coerced to a
        # real Python None it becomes [None] and the callback lookup rejects it.
        training_args = TrainingArguments(**self.merged_training_args(report_to="none"))

        self.assertEqual(training_args.report_to, [])
        self.assertEqual(get_reporting_integration_callbacks(training_args.report_to), [])

    def test_report_to_none_matches_the_yaml_only_path(self):
        from_yaml = TrainingArguments(**self.merged_training_args())
        from_cli = TrainingArguments(**self.merged_training_args(report_to="none"))

        self.assertEqual(from_cli.report_to, from_yaml.report_to)

    def test_fields_without_a_string_sentinel_still_coerce_to_python_none(self):
        # deepspeed is typed ``dict | str | None``; "none" there must stay None,
        # or the path loader would look for a config file literally named "none".
        self.assertIsNone(self.merged_training_args(deepspeed="none")["deepspeed"])

    def test_unrelated_coercions_are_untouched(self):
        merged = self.merged_training_args(max_steps="7", bf16="true", run_name="probe-2")

        self.assertEqual(merged["max_steps"], 7)
        self.assertIs(merged["bf16"], True)
        self.assertEqual(merged["run_name"], "probe-2")


if __name__ == "__main__":
    unittest.main()
