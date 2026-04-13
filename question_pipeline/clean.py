from __future__ import annotations

from typing import Dict, List

from .utils import dump_csv, emit, load_raw_questions, normalize_record, write_jsonl


def run_phase1_clean(input_dir: str, output_dir: str, progress_callback=None) -> Dict:
    rows = load_raw_questions(input_dir)
    emit(progress_callback, f"Loaded {len(rows)} raw rows for cleaning.", phase=1)
    seen = {}
    cleaned = []
    duplicate_count = 0
    bijoy_count = 0
    for row in rows:
        normalized = normalize_record(row)
        if normalized["bijoy_detected"]:
            bijoy_count += 1
        fingerprint = normalized["fingerprint"]
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen[fingerprint] = normalized["id"]
        cleaned.append(normalized)
    jsonl_path = f"{output_dir}/phase1_cleaned.jsonl"
    csv_path = f"{output_dir}/phase1_cleaned.csv"
    write_jsonl(jsonl_path, cleaned)
    dump_csv(csv_path, cleaned)
    emit(progress_callback, f"Phase 1 completed. {len(cleaned)} rows kept, {duplicate_count} duplicates removed.", phase=1)
    return {
        "row_count": len(cleaned),
        "duplicate_count": duplicate_count,
        "bijoy_detected_count": bijoy_count,
        "output_file": jsonl_path,
        "output_csv": csv_path,
    }
