#!/usr/bin/env python3
"""Prepare a goal-image request from a selected proposal result."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--head-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.prediction.read_text().splitlines() if line.strip()]
    chosen = [row for row in rows if row["id"] == args.id]
    if len(chosen) != 1 or not chosen[0].get("format_valid") or not chosen[0].get("subtask"):
        raise ValueError("Select exactly one valid proposal")
    subtask = chosen[0]["subtask"]
    if subtask.strip().lower() in ("(done)", "(fail)", "(empty)"):
        raise ValueError("Terminal/status output has no goal image to generate")
    if not args.head_image.is_file():
        raise FileNotFoundError(args.head_image)
    with args.output.open("x") as handle:
        handle.write(json.dumps({"id": "goal", "image": str(args.head_image.resolve()),
                                 "instruction": subtask}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
