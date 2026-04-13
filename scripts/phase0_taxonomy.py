#!/usr/bin/env python3
import argparse
import json

from question_pipeline.taxonomy import run_phase0_taxonomy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--auto_classify", action="store_true")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    result = run_phase0_taxonomy(
        args.input_dir,
        args.output_dir,
        auto_classify=args.auto_classify,
        llm_model=args.model,
        ollama_host=args.ollama_host,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
