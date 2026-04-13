from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
MATH_SYMBOL_MAP = {
    "×": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "²": "^2",
    "³": "^3",
    "√": "sqrt",
    "π": "pi",
}
DEFAULT_RAW_COLUMNS = ["id", "db_id", "question_id", "question", "answer", "explanation", "topic"]


@dataclass
class ProgressUpdate:
    message: str
    phase: Optional[int] = None


def emit(progress_callback, message: str, phase: Optional[int] = None):
    if progress_callback:
        progress_callback(ProgressUpdate(message=message, phase=phase))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Dict]):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_get(row: Dict, *keys, default=""):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _parse_txt_questions(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    rows: List[Dict] = []
    for idx, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        question = lines[0]
        answer = ""
        explanation = ""
        topic = ""
        for line in lines[1:]:
            lowered = line.lower()
            if lowered.startswith("answer:") or lowered.startswith("ans:"):
                answer = line.split(":", 1)[1].strip() if ":" in line else line
            elif lowered.startswith("explanation:") or lowered.startswith("solution:"):
                explanation = line.split(":", 1)[1].strip() if ":" in line else line
            elif lowered.startswith("topic:") or lowered.startswith("subject:"):
                topic = line.split(":", 1)[1].strip() if ":" in line else line
            elif not answer:
                answer = line
            elif not explanation:
                explanation = line
        rows.append(
            {
                "id": f"{path.stem}-{idx}",
                "question": question,
                "answer": answer,
                "explanation": explanation,
                "topic": topic,
                "source_file": path.name,
                "raw": {"text_block": block},
            }
        )
    return rows


def load_raw_questions(input_dir: str | Path) -> List[Dict]:
    input_dir = Path(input_dir)
    rows: List[Dict] = []
    for path in sorted(input_dir.glob("*")):
        if path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl(path))
            continue
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows.extend(payload)
            continue
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for idx, row in enumerate(csv.DictReader(handle)):
                    question_id = _row_get(row, "id", "db_id", "question_id", default=f"{path.stem}-{idx+1}")
                    rows.append(
                        {
                            "id": str(question_id),
                            "question": str(_row_get(row, "question", "prompt", "stem")).strip(),
                            "answer": str(_row_get(row, "answer", "correct_answer", "option_answer")).strip(),
                            "explanation": str(_row_get(row, "explanation", "solution", "reason")).strip(),
                            "topic": str(_row_get(row, "topic", "subject", "category")).strip(),
                            "source_file": path.name,
                            "raw": {key: row.get(key, "") for key in row.keys()},
                        }
                    )
            continue
        if path.suffix.lower() == ".txt":
            rows.extend(_parse_txt_questions(path))
    return [row for row in rows if row.get("question")]


def dump_csv(path: str | Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    path = Path(path)
    ensure_dir(path.parent)
    if not fieldnames:
        discovered = set()
        for row in rows:
            discovered.update(row.keys())
        fieldnames = sorted(discovered)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def contains_bangla(text: str) -> bool:
    return bool(text and re.search(r"[\u0980-\u09FF]", text))


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([।!?])", r"\1", text)
    text = re.sub(r"([।!?])([^\s\n])", r"\1 \2", text)
    return text.strip()


def normalize_latex_math(text: str) -> str:
    normalized = text
    for source, target in MATH_SYMBOL_MAP.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", normalized)
    normalized = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = re.sub(r"\\times", "*", normalized)
    normalized = re.sub(r"\\div", "/", normalized)
    return normalized


def _bijoy_converter(text: str) -> Optional[str]:
    try:
        from bijoy2unicode import converter  # type: ignore

        return converter.Unicode(text)
    except Exception:
        return None


def looks_like_bijoy(text: str) -> bool:
    if not text:
        return False
    ascii_ratio = sum(1 for char in text if ord(char) < 128) / max(1, len(text))
    strange_pattern = re.search(r"[Avkª¯Íœ±¼½¾]", text)
    return ascii_ratio > 0.6 and bool(strange_pattern)


def maybe_convert_bijoy(text: str) -> tuple[str, bool]:
    if not looks_like_bijoy(text):
        return text, False
    converted = _bijoy_converter(text)
    if converted and contains_bangla(converted):
        return converted, True
    return text, False


def normalize_bangla_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(BENGALI_DIGIT_MAP)
    normalized = normalize_latex_math(normalized)
    normalized = normalize_whitespace(normalized)
    return normalized


def normalize_record(record: Dict) -> Dict:
    question, question_bijoy = maybe_convert_bijoy(str(record.get("question", "")))
    answer, answer_bijoy = maybe_convert_bijoy(str(record.get("answer", "")))
    explanation, explanation_bijoy = maybe_convert_bijoy(str(record.get("explanation", "")))
    question = normalize_bangla_text(question)
    answer = normalize_bangla_text(answer)
    explanation = normalize_bangla_text(explanation)
    fingerprint = fingerprint_record(question, answer)
    return {
        **record,
        "question": question,
        "answer": answer,
        "explanation": explanation,
        "fingerprint": fingerprint,
        "bijoy_detected": bool(question_bijoy or answer_bijoy or explanation_bijoy),
    }


def fingerprint_record(question: str, answer: str) -> str:
    payload = f"{normalize_whitespace(question).lower()}||{normalize_whitespace(answer).lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return values[low]
    fraction = idx - low
    return values[low] * (1 - fraction) + values[high] * fraction


def checkpoint_path(output_dir: str | Path, phase_name: str) -> Path:
    return ensure_dir(output_dir) / f"{phase_name}_checkpoint.json"


def load_checkpoint(output_dir: str | Path, phase_name: str) -> Dict:
    return read_json(checkpoint_path(output_dir, phase_name), default={"completed_ids": [], "meta": {}})


def save_checkpoint(output_dir: str | Path, phase_name: str, completed_ids: Iterable[str], meta: Optional[Dict] = None):
    write_json(
        checkpoint_path(output_dir, phase_name),
        {"completed_ids": list(completed_ids), "meta": meta or {}},
    )


def majority_vote(items: List[Dict], vote_key: str = "passed") -> Dict:
    if not items:
        return {}
    tally: Dict[bool, int] = {}
    for item in items:
        tally[bool(item.get(vote_key))] = tally.get(bool(item.get(vote_key)), 0) + 1
    winner = max(tally.items(), key=lambda item: item[1])[0]
    winner_items = [item for item in items if bool(item.get(vote_key)) == winner]
    return winner_items[0] if winner_items else items[0]
