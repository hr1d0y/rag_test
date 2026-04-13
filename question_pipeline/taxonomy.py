from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .llm import classify_with_llm
from .utils import emit, ensure_dir, load_raw_questions, write_jsonl


DEFAULT_TAXONOMY = {
    "ইতিহাস": {
        "বাংলাদেশের ইতিহাস": {
            "মুক্তিযুদ্ধ": ["মুক্তিযুদ্ধ", "স্বাধীনতা যুদ্ধ", "1971", "১৯৭১", "সেক্টর", "রাজাকার"],
            "ভাষা আন্দোলন": ["ভাষা আন্দোলন", "একুশে", "1952", "১৯৫২", "রাষ্ট্রভাষা"],
            "ব্রিটিশ শাসন": ["ব্রিটিশ", "ইস্ট ইন্ডিয়া", "লর্ড কার্জন", "বঙ্গভঙ্গ"],
        },
        "বিশ্ব ইতিহাস": {
            "বিশ্বযুদ্ধ": ["বিশ্বযুদ্ধ", "world war", "হিটলার", "ভার্সাই", "নাজি"],
            "প্রাচীন সভ্যতা": ["মিশর", "সুমের", "মেসোপটেমিয়া", "ইন্দাস", "রোম", "গ্রিস"],
        },
    },
    "গণিত": {
        "বীজগণিত": {
            "সমীকরণ": ["সমীকরণ", "equation", "x =", "quadratic", "দ্বিঘাত"],
            "সূচক-লগ": ["log", "লগ", "সূচক", "exponent"],
        },
        "জ্যামিতি": {
            "ত্রিভুজ": ["triangle", "ত্রিভুজ", "কোণ", "angle"],
            "বৃত্ত": ["circle", "বৃত্ত", "ব্যাসার্ধ", "radius"],
        },
    },
    "ইংরেজি": {
        "Grammar": {
            "Tense": ["tense", "present", "past", "future"],
            "Parts of Speech": ["noun", "verb", "adjective", "adverb", "pronoun"],
        }
    },
    "General": {
        "Mixed": {
            "General Knowledge": ["বাংলাদেশ", "সংবিধান", "রাজধানী", "নদী", "bcs"],
        }
    },
}


def taxonomy_prompt(taxonomy: Dict) -> str:
    lines = ["Classify the question into the following taxonomy:"]
    for parent, children in taxonomy.items():
        lines.append(parent)
        for child, leaves in children.items():
            lines.append(f"  - {child}")
            for leaf in leaves:
                lines.append(f"    - {leaf}")
    return "\n".join(lines)


def classify_rule_based(question: str, taxonomy: Dict) -> Optional[Dict]:
    lowered = question.lower()
    for parent, children in taxonomy.items():
        for child, leaves in children.items():
            for leaf, keywords in leaves.items():
                if any(keyword.lower() in lowered for keyword in keywords):
                    return {"parent": parent, "child": child, "leaf": leaf}
    return None


def run_phase0_taxonomy(
    input_dir: str,
    output_dir: str,
    auto_classify: bool = False,
    llm_model: str = "qwen2.5:7b",
    ollama_host: str = "http://127.0.0.1:11434",
    progress_callback=None,
) -> Dict:
    emit(progress_callback, "Loading raw questions...", phase=0)
    questions = load_raw_questions(input_dir)
    taxonomy = DEFAULT_TAXONOMY
    prompt = taxonomy_prompt(taxonomy)
    categorized = []
    unresolved = 0
    for row in questions:
        classification = classify_rule_based(row.get("question", ""), taxonomy)
        if not classification and auto_classify:
            classification = classify_with_llm(llm_model, row.get("question", ""), prompt, ollama_host)
        if not classification:
            classification = {"parent": "Unclassified", "child": "Unclassified", "leaf": "Unclassified"}
            unresolved += 1
        categorized.append({**row, **classification})
    output_dir_path = ensure_dir(output_dir)
    write_jsonl(output_dir_path / "phase0_taxonomy.jsonl", categorized)
    grouped = defaultdict(list)
    for row in categorized:
        grouped[row["leaf"]].append(row["id"])
    emit(progress_callback, f"Phase 0 completed. Classified {len(categorized)} rows with {unresolved} unresolved.", phase=0)
    return {
        "row_count": len(categorized),
        "unresolved_count": unresolved,
        "leaf_counts": {leaf: len(ids) for leaf, ids in grouped.items()},
        "output_file": str(output_dir_path / "phase0_taxonomy.jsonl"),
    }
