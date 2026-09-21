import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from tau0_world_model.common import PREFIX, import_vendor, read_pairs

import_vendor()
from modules.model_edit import Step1XEdit, original_format_state_dict


class Contracts(unittest.TestCase):
    def test_only_known_unused_v1p2_weights_are_omitted(self):
        model = Step1XEdit.__new__(Step1XEdit)
        torch.nn.Module.__init__(model)
        model.params = SimpleNamespace(version="v1p2")
        state = {
            "active.weight": torch.empty(2, device="meta"),
            "text_token_mapping.weight": torch.empty(4096, 3584, device="meta"),
            "text_token_mapping.bias": torch.empty(4096, device="meta"),
        }
        with self.assertWarns(UserWarning):
            result = original_format_state_dict(model, state)
        self.assertEqual(set(result), {"active.weight"})
        self.assertEqual(len(state), 3)
        del state["text_token_mapping.bias"]
        with self.assertRaises(ValueError):
            original_format_state_dict(model, state)

    def test_portable_pairs_preserve_prompt_and_reject_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4)).save(root / "start.png")
            Image.new("RGB", (4, 4)).save(root / "target.png")
            row = {"id": "pair_1", "image": "start.png", "target": "target.png",
                   "instruction": PREFIX + "Pick up the cup."}
            path = root / "pairs.jsonl"
            line = json.dumps(row) + "\n"
            path.write_text(line)
            rows = read_pairs(path, training=True)
            self.assertEqual(rows[0]["prompt"], row["instruction"])
            self.assertEqual(rows[0]["image"], str(root / "start.png"))
            path.write_text(line * 2)
            with self.assertRaises(ValueError):
                read_pairs(path, training=True)


if __name__ == "__main__":
    unittest.main()
