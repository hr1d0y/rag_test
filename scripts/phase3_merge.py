#!/usr/bin/env python3
import argparse
import json

from question_pipeline.merge import run_phase3_merge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    result = run_phase3_merge(
        f"{args.input_dir}/phase2_clusters.jsonl" if not args.input_dir.endswith(".jsonl") else args.input_dir,
        args.output_dir,
        model=args.model,
        host=args.ollama_host,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
