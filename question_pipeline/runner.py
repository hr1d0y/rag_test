from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .clean import run_phase1_clean
from .cluster import run_phase2_cluster
from .taxonomy import run_phase0_taxonomy
from .merge import run_phase3_merge
from .verify_db import run_phase5_verify_db
from .verify_truth import run_phase4_verify_truth
from .utils import emit, ensure_dir, read_jsonl, write_jsonl


def setup_data_dir(data_dir: str):
    data = Path(data_dir)
    for name in ["raw", "categorized", "cleaned", "clustered", "merged", "truth", "verification", "reports"]:
        ensure_dir(data / name)
    return data


def estimate_pipeline_cost(row_count: int, gpu_hourly_cost: float = 1.5) -> Dict:
    phase3_hours = max(1.0, row_count / 160000)
    phase4_hours = max(1.5, row_count / 100000)
    phase5_hours = max(0.6, row_count / 250000)
    total_hours = phase3_hours + phase4_hours + phase5_hours
    return {
        "row_count": row_count,
        "phase3_hours": round(phase3_hours, 2),
        "phase4_hours": round(phase4_hours, 2),
        "phase5_hours": round(phase5_hours, 2),
        "total_hours": round(total_hours, 2),
        "estimated_cloud_cost": round(total_hours * gpu_hourly_cost, 2),
    }


def run_pipeline(
    data_dir: str,
    phase: Optional[int] = None,
    from_phase: Optional[int] = None,
    run_all: bool = False,
    topic: Optional[str] = None,
    strategy: str = "hybrid",
    threshold: float = 0.82,
    merge_model: str = "qwen2.5:32b",
    verify_models: Optional[List[str]] = None,
    verify_db_model: str = "qwen2.5:7b",
    auto_classify: bool = False,
    ollama_host: str = "http://127.0.0.1:11434",
    progress_callback=None,
) -> Dict:
    data = setup_data_dir(data_dir)
    verify_models = verify_models or ["qwen2.5:32b", "llama3.1:8b", "mistral:7b"]

    if run_all:
        phases = [0, 1, 2, 3, 4, 5]
    elif from_phase is not None:
        phases = [p for p in [0, 1, 2, 3, 4, 5] if p >= from_phase]
    elif phase is not None:
        phases = [phase]
    else:
        phases = [0, 1, 2, 3, 4, 5]

    results = {}
    categorized_file = data / "categorized" / "phase0_taxonomy.jsonl"
    cleaned_file = data / "cleaned" / "phase1_cleaned.jsonl"
    clustered_file = data / "clustered" / "phase2_clusters.jsonl"
    merged_file = data / "merged" / "phase3_merged.jsonl"
    truth_file = data / "truth" / "phase4_truth.jsonl"
    working_cleaned_file = cleaned_file

    for current_phase in phases:
        emit(progress_callback, f"Running phase {current_phase}...", phase=current_phase)
        if current_phase == 0:
            results["phase0"] = run_phase0_taxonomy(
                str(data / "raw"),
                str(data / "categorized"),
                auto_classify=auto_classify,
                llm_model=verify_db_model,
                ollama_host=ollama_host,
                progress_callback=progress_callback,
            )
        elif current_phase == 1:
            source_dir = data / "raw"
            if categorized_file.exists():
                # Use categorized jsonl as source if available.
                source_dir = data / "raw"
            results["phase1"] = run_phase1_clean(str(source_dir), str(data / "cleaned"), progress_callback=progress_callback)
            if categorized_file.exists():
                categorized_rows = read_jsonl(categorized_file)
                cleaned_rows = read_jsonl(cleaned_file)
                by_id = {row["id"]: row for row in categorized_rows}
                enriched = [{**by_id.get(row["id"], {}), **row} for row in cleaned_rows]
                write_jsonl(cleaned_file, enriched)
            if topic:
                topic_lower = topic.lower()
                filtered = [
                    row for row in read_jsonl(cleaned_file)
                    if topic_lower in " ".join(
                        [
                            str(row.get("topic", "")),
                            str(row.get("parent", "")),
                            str(row.get("child", "")),
                            str(row.get("leaf", "")),
                            str(row.get("question", "")),
                        ]
                    ).lower()
                ]
                working_cleaned_file = data / "cleaned" / "phase1_cleaned_topic.jsonl"
                write_jsonl(working_cleaned_file, filtered)
                results["topic_filter"] = {"topic": topic, "row_count": len(filtered), "output_file": str(working_cleaned_file)}
        elif current_phase == 2:
            results["phase2"] = run_phase2_cluster(
                str(working_cleaned_file),
                str(data / "clustered"),
                strategy=strategy,
                threshold=threshold,
                progress_callback=progress_callback,
            )
        elif current_phase == 3:
            results["phase3"] = run_phase3_merge(
                str(clustered_file),
                str(data / "merged"),
                model=merge_model,
                host=ollama_host,
                progress_callback=progress_callback,
            )
        elif current_phase == 4:
            results["phase4"] = run_phase4_verify_truth(
                str(merged_file),
                str(data / "truth"),
                models=verify_models,
                host=ollama_host,
                progress_callback=progress_callback,
            )
        elif current_phase == 5:
            results["phase5"] = run_phase5_verify_db(
                str(working_cleaned_file),
                str(truth_file),
                str(data / "verification"),
                model=verify_db_model,
                host=ollama_host,
                progress_callback=progress_callback,
            )
    return results
