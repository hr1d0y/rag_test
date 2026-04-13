#!/usr/bin/env python3
import argparse
import json

from question_pipeline.runner import estimate_pipeline_cost, run_pipeline, setup_data_dir
from question_pipeline.utils import load_raw_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--phase", type=int)
    parser.add_argument("--from_phase", type=int)
    parser.add_argument("--topic")
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--merge_model", default="qwen2.5:32b")
    parser.add_argument("--verify_models", nargs="*", default=["qwen2.5:32b", "llama3.1:8b", "mistral:7b"])
    parser.add_argument("--verify_db_model", default="qwen2.5:7b")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--auto_classify", action="store_true")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    if args.setup:
        setup_data_dir(args.data_dir)
        print(json.dumps({"setup": "ok", "data_dir": args.data_dir}, ensure_ascii=False, indent=2))
        return

    if args.estimate:
        row_count = len(load_raw_questions(f"{args.data_dir}/raw"))
        print(json.dumps(estimate_pipeline_cost(row_count), ensure_ascii=False, indent=2))
        return

    result = run_pipeline(
        data_dir=args.data_dir,
        phase=args.phase,
        from_phase=args.from_phase,
        run_all=args.all,
        topic=args.topic,
        strategy=args.strategy,
        threshold=args.threshold,
        merge_model=args.merge_model,
        verify_models=args.verify_models,
        verify_db_model=args.verify_db_model,
        auto_classify=args.auto_classify,
        ollama_host=args.ollama_host,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
