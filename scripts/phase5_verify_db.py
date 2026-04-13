#!/usr/bin/env python3
import argparse
import json

from question_pipeline.verify_db import run_phase5_verify_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_input", required=True)
    parser.add_argument("--truth_input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    result = run_phase5_verify_db(
        args.cleaned_input,
        args.truth_input,
        args.output_dir,
        model=args.model,
        host=args.ollama_host,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
