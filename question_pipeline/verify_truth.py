from __future__ import annotations

from typing import Dict, List

from .llm import verify_truth_with_llm
from .utils import emit, load_checkpoint, majority_vote, read_jsonl, save_checkpoint, write_jsonl


def run_phase4_verify_truth(
    input_file: str,
    output_dir: str,
    models: List[str],
    host: str = "http://127.0.0.1:11434",
    progress_callback=None,
) -> Dict:
    rows = read_jsonl(input_file)
    checkpoint = load_checkpoint(output_dir, "phase4_verify_truth")
    completed_ids = set(checkpoint.get("completed_ids", []))
    verified_rows = {row["cluster_id"]: row for row in read_jsonl(f"{output_dir}/phase4_truth.jsonl")}
    for row in rows:
        cluster_id = row["cluster_id"]
        if cluster_id in completed_ids:
            continue
        votes = [verify_truth_with_llm(model, row, host) for model in models]
        winner = majority_vote(votes, vote_key="passed")
        final_answer = winner.get("corrected_answer") or row.get("canonical_answer", "")
        verification_state = "verified" if winner.get("passed") else winner.get("verdict", "review")
        verified_row = {
            **row,
            "verification_votes": votes,
            "verification_state": verification_state,
            "final_answer": final_answer,
            "review_required": verification_state == "review",
        }
        verified_rows[cluster_id] = verified_row
        completed_ids.add(cluster_id)
        save_checkpoint(output_dir, "phase4_verify_truth", completed_ids, {"models": models})
        emit(progress_callback, f"Verified cluster {cluster_id} using {len(models)} models.", phase=4)
    output_path = f"{output_dir}/phase4_truth.jsonl"
    write_jsonl(output_path, list(verified_rows.values()))
    return {"verified_count": len(verified_rows), "output_file": output_path, "models": models}
