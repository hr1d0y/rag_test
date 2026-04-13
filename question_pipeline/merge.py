from __future__ import annotations

from typing import Dict, List

from .llm import merge_cluster_with_llm
from .utils import emit, load_checkpoint, read_jsonl, save_checkpoint, write_jsonl


def run_phase3_merge(
    input_file: str,
    output_dir: str,
    model: str = "qwen2.5:32b",
    host: str = "http://127.0.0.1:11434",
    progress_callback=None,
) -> Dict:
    clusters = read_jsonl(input_file)
    checkpoint = load_checkpoint(output_dir, "phase3_merge")
    completed_ids = set(checkpoint.get("completed_ids", []))
    merged_entries: List[Dict] = []
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        if cluster_id in completed_ids:
            continue
        merged = merge_cluster_with_llm(model, cluster, host)
        merged_entry = {**cluster, **merged, "merge_model": model}
        merged_entries.append(merged_entry)
        completed_ids.add(cluster_id)
        save_checkpoint(output_dir, "phase3_merge", completed_ids, {"model": model})
        emit(progress_callback, f"Merged cluster {cluster_id} with {cluster['size']} items.", phase=3)
    existing = read_jsonl(f"{output_dir}/phase3_merged.jsonl")
    by_id = {row["cluster_id"]: row for row in existing}
    for row in merged_entries:
        by_id[row["cluster_id"]] = row
    rows = list(by_id.values())
    write_jsonl(f"{output_dir}/phase3_merged.jsonl", rows)
    return {"merged_count": len(rows), "output_file": f"{output_dir}/phase3_merged.jsonl", "model": model}
