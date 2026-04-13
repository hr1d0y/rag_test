from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional


def ollama_generate(model: str, prompt: str, host: str = "http://127.0.0.1:11434", temperature: float = 0.1) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def parse_json_response(raw_text: str, default=None):
    text = raw_text.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return default
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return default


def classify_with_llm(model: str, question: str, taxonomy_prompt: str, host: str) -> Optional[Dict]:
    prompt = (
        f"{taxonomy_prompt}\n\n"
        "Return JSON with keys parent, child, leaf.\n"
        f"Question: {question}"
    )
    parsed = parse_json_response(ollama_generate(model, prompt, host=host), default={})
    if isinstance(parsed, dict) and parsed.get("parent") and parsed.get("leaf"):
        return {
            "parent": parsed.get("parent", "").strip(),
            "child": parsed.get("child", "").strip(),
            "leaf": parsed.get("leaf", "").strip(),
        }
    return None


def merge_cluster_with_llm(model: str, cluster: Dict, host: str) -> Dict:
    joined_explanations = "\n\n".join(
        [
            f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', '')}\nExplanation: {item.get('explanation', '')}"
            for item in cluster.get("items", [])
        ]
    )
    prompt = f"""You are merging duplicate or near-duplicate educational QA entries.
Preserve every factual detail. Do not drop dates, names, formulas, examples, or qualifiers.
Respond only with valid JSON using this schema:
{{
  "canonical_question": "best merged question",
  "canonical_answer": "best merged answer",
  "merged_explanation": "fully merged explanation",
  "key_facts": ["fact 1", "fact 2"],
  "quality_notes": "short note"
}}

Entries:
{joined_explanations}
"""
    parsed = parse_json_response(ollama_generate(model, prompt, host=host), default={})
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "canonical_question": parsed.get("canonical_question") or cluster.get("representative_question", ""),
        "canonical_answer": parsed.get("canonical_answer") or cluster.get("representative_answer", ""),
        "merged_explanation": parsed.get("merged_explanation") or cluster.get("representative_explanation", ""),
        "key_facts": parsed.get("key_facts") or [],
        "quality_notes": parsed.get("quality_notes") or "",
    }


def verify_truth_with_llm(model: str, merged_entry: Dict, host: str) -> Dict:
    prompt = f"""You are verifying whether a merged educational entry is correct.
Return only JSON:
{{
  "passed": true,
  "verdict": "verified | corrected | review",
  "corrected_answer": "answer if correction is needed, else empty string",
  "reason": "short reason"
}}

Question: {merged_entry.get("canonical_question", "")}
Answer: {merged_entry.get("canonical_answer", "")}
Explanation: {merged_entry.get("merged_explanation", "")}
"""
    parsed = parse_json_response(ollama_generate(model, prompt, host=host), default={})
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "model": model,
        "passed": bool(parsed.get("passed", False)),
        "verdict": parsed.get("verdict", "review"),
        "corrected_answer": parsed.get("corrected_answer", ""),
        "reason": parsed.get("reason", ""),
    }


def compare_db_to_truth(model: str, question_row: Dict, truth_entry: Dict, host: str) -> Dict:
    prompt = f"""Compare a database QA entry against a verified source-of-truth entry.
Return only JSON:
{{
  "status": "valid | invalid | review",
  "severity": "low | medium | critical",
  "errors": [
    {{
      "type": "wrong_answer | incomplete_answer | explanation_mismatch | unsupported",
      "detail": "short detail",
      "correction": "short correction"
    }}
  ]
}}

DB question: {question_row.get("question", "")}
DB answer: {question_row.get("answer", "")}
DB explanation: {question_row.get("explanation", "")}

Truth question: {truth_entry.get("canonical_question", "")}
Truth answer: {truth_entry.get("final_answer", truth_entry.get("canonical_answer", ""))}
Truth explanation: {truth_entry.get("merged_explanation", "")}
"""
    parsed = parse_json_response(ollama_generate(model, prompt, host=host), default={})
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "status": parsed.get("status", "review"),
        "severity": parsed.get("severity", "medium"),
        "errors": parsed.get("errors", []),
    }
