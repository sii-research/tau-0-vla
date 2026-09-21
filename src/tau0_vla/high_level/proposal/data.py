"""Portable JSONL samples with media paths relative to the manifest."""
import hashlib
import json
from pathlib import Path

from PIL import Image

from .format import answer_text, messages, ordered_images, task_type


def read_samples(path, require_answers=False):
    path = Path(path).resolve()
    rows, seen = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f"Missing or duplicate string id on line {number}")
        seen.add(sid)
        task_type(row.get("task_type", "full_qa"))
        paths = [str((path.parent / p).resolve()) for p in ordered_images(row["images"])]
        for p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        row["images"] = dict(zip(("head", "left", "right"), paths))
        messages(row, paths)  # Validate prompt before loading weights.
        if require_answers:
            answer_text(row)
        rows.append(row)
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def load_images(row, height=480, width=640):
    result = []
    for path in ordered_images(row["images"]):
        with Image.open(path) as image:
            result.append(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
