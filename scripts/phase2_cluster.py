#!/usr/bin/env python3
import argparse
import json

from question_pipeline.cluster import run_phase2_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()
    result = run_phase2_cluster(
        f"{args.input_dir}/phase1_cleaned.jsonl" if not args.input_dir.endswith(".jsonl") else args.input_dir,
        args.output_dir,
        strategy=args.strategy,
        threshold=args.threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
