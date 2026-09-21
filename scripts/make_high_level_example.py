#!/usr/bin/env python3
"""Copy the bundled high-level inference and fine-tuning examples."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("high_level/examples"))
    parser.add_argument("--output", type=Path, default=Path("outputs/high_level_example"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ("proposal.jsonl", "proposal_sft.jsonl", "world_model.jsonl", "world_model_sft.jsonl"):
        if not (args.data / name).is_file():
            raise FileNotFoundError(args.data / name)
    shutil.copytree(args.data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
