from __future__ import annotations

from typing import Dict, List

from .llm import compare_db_to_truth
from .utils import emit, load_checkpoint, normalize_whitespace, read_jsonl, save_checkpoint, write_jsonl


def _exact_match(question_row: Dict, truth_row: Dict) -> bool:
    return normalize_whitespace(question_row.get("answer", "")).lower() == normalize_whitespace(
        truth_row.get("final_answer", truth_row.get("canonical_answer", ""))
    ).lower()


def run_phase5_verify_db(
    cleaned_input_file: str,
    truth_file: str,
    output_dir: str,
    model: str = "qwen2.5:7b",
    host: str = "http://127.0.0.1:11434",
    progress_callback=None,
) -> Dict:
    questions = read_jsonl(cleaned_input_file)
    truth_rows = read_jsonl(truth_file)
    truth_by_fingerprint = {}
    for row in truth_rows:
        for item in row.get("items", []):
            truth_by_fingerprint[item.get("fingerprint")] = row
    checkpoint = load_checkpoint(output_dir, "phase5_verify_db")
    completed_ids = set(checkpoint.get("completed_ids", []))
    reports = {row["db_id"]: row for row in read_jsonl(f"{output_dir}/phase5_verification.jsonl")}
    quick_checks = 0
    llm_checks = 0
    for question in questions:
        question_id = question.get("id")
        if question_id in completed_ids:
            continue
        truth_row = truth_by_fingerprint.get(question.get("fingerprint"))
        if not truth_row:
            reports[question_id] = {
                "db_id": question_id,
                "status": "review",
                "severity": "medium",
                "errors": [{"type": "missing_truth_match", "detail": "No truth entry matched this fingerprint.", "correction": ""}],
            }
        elif _exact_match(question, truth_row):
            quick_checks += 1
            reports[question_id] = {"db_id": question_id, "status": "valid", "severity": "low", "errors": []}
        else:
            llm_checks += 1
            comparison = compare_db_to_truth(model, question, truth_row, host)
            reports[question_id] = {"db_id": question_id, **comparison}
        completed_ids.add(question_id)
        save_checkpoint(output_dir, "phase5_verify_db", completed_ids, {"model": model})
        emit(progress_callback, f"Verified DB row {question_id}.", phase=5)
    output_path = f"{output_dir}/phase5_verification.jsonl"
    write_jsonl(output_path, list(reports.values()))
    return {
        "verified_count": len(reports),
        "quick_checks": quick_checks,
        "llm_checks": llm_checks,
        "output_file": output_path,
    }
