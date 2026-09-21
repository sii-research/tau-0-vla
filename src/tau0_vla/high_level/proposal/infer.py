"""Run proposal inference on a portable JSONL manifest."""
import argparse
import json
from pathlib import Path

from .data import read_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    rows = read_samples(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)
    from .runtime import Proposal
    policy = Proposal(args.model, args.adapter, args.device, args.max_new_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row["id"], **policy.predict(row)}, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
