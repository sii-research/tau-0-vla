"""Resumable evaluation: exact-match statistics and complete per-sample outputs."""
import argparse
import json
from pathlib import Path

from .data import read_samples, sha256
from .format import answer_text, parse_output


def weight_binding(path):
    path = Path(path).resolve()
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights: {path}")
    return {p.name: sha256(p) for p in files + sorted(path.glob("*.json")) + sorted(path.glob("*.jinja"))}


def read_results(path, samples):
    allowed = {row["id"]: row for row in samples}
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            sid = row["id"]
            if sid not in allowed or sid in records:
                raise ValueError(f"Unexpected/duplicate result ID: {sid}")
            if row["ground_truth"] != answer_text(allowed[sid]):
                raise ValueError(f"Ground truth mismatch: {sid}")
            records[sid] = row
    return records


def metrics(records):
    result = {"samples": len(records), "exact_match_is_semantic_accuracy": False}
    for key in ("format_valid", "subtask_exact_match", "think_exact_match", "memory_exact_match"):
        values = [row[key] for row in records if row.get(key) is not None]
        result[key] = {"count": sum(values), "denominator": len(values),
                       "rate": sum(values) / len(values) if values else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    samples = read_samples(args.input, require_answers=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    binding = {
        "input_sha256": sha256(args.input), "model": weight_binding(args.model),
        "adapter": weight_binding(args.adapter) if args.adapter else None,
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "attention": "eager", "image_resize_hw": [480, 640],
        "camera_order": ["head", "left", "right"], "enable_thinking": False,
        "media_sha256": {path: sha256(path) for path in sorted(
            {path for row in samples for path in row["images"].values()}
        )},
    }
    binding_path = args.output.with_suffix(".binding.json")
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError("Resume binding differs: data, model or inference parameters changed")
    elif args.output.exists():
        raise ValueError("Existing results lack a binding; refusing unsafe resume")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    records = read_results(args.output, samples)
    pending = [row for row in samples if row["id"] not in records]
    if pending:
        from .runtime import Proposal
        policy = Proposal(args.model, args.adapter, args.device, args.max_new_tokens)
        with args.output.open("a") as handle:
            for sample in pending:
                pred = policy.predict(sample)
                gt_text = answer_text(sample)
                gt = parse_output(gt_text, sample.get("task_type", "full_qa"))
                record = {"id": sample["id"], "input": sample, "ground_truth": gt_text, **pred}
                for field, output_field in (("think", "think"), ("memory", "memory_out"), ("subtask", "subtask")):
                    record[f"{field}_exact_match"] = (
                        pred["format_valid"] and pred[output_field] == gt[output_field]
                    ) if gt[output_field] is not None else None
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records[sample["id"]] = record
    if len(records) != len(samples):
        raise ValueError("Incomplete evaluation")
    result = {"complete": True, **metrics(list(records.values()))}
    args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
