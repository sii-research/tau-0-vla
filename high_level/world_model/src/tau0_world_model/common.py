"""Dataset and model paths shared by training and inference."""
import hashlib
import json
import re
import sys
from pathlib import Path

PREFIX = "In a robotic workspace, a dual-arm robot: "
VENDOR = Path(__file__).parent / "_vendor" / "step1x"


def import_vendor():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def prompt(instruction):
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")
    return instruction if instruction.startswith(PREFIX) else PREFIX + instruction


def read_pairs(path, training=False):
    path = Path(path).resolve()
    rows, ids = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row.get("id", "")) or row["id"] in ids:
            raise ValueError("Each id must be unique and contain only letters, digits, '_' or '-'")
        ids.add(row["id"])
        row["prompt"] = prompt(row["instruction"])
        for key in (("image", "target") if training else ("image",)):
            row[key] = str((path.parent / row[key]).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        rows.append(row)
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def model_files(root):
    root = Path(root).resolve()
    files = {"dit_path": root / "step1x-edit-v1p2.safetensors",
             "ae_path": root / "vae.safetensors",
             "qwen2vl_model_path": root / "Qwen2.5-VL-7B-Instruct"}
    for value in files.values():
        if not value.exists():
            raise FileNotFoundError(value)
    return {key: str(value) for key, value in files.items()}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()
