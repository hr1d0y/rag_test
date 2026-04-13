from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_CONFIGS = {
    "multilingual": ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"],
    "bangla_primary": ["csebuetnlp/banglabert"],
    "hybrid": [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "csebuetnlp/banglabert",
    ],
}


@lru_cache(maxsize=8)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=4)
def load_auto_model(model_name: str):
    return AutoModel.from_pretrained(model_name)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _merge_embeddings(parts: List[np.ndarray]) -> np.ndarray:
    if len(parts) == 1:
        return _normalize(parts[0])
    if all(part.shape[1] == parts[0].shape[1] for part in parts):
        merged = sum(parts) / len(parts)
        return _normalize(merged)
    merged = np.concatenate([_normalize(part) for part in parts], axis=1)
    return _normalize(merged)


def embed_texts(texts: List[str], strategy: str = "multilingual") -> np.ndarray:
    model_names = MODEL_CONFIGS.get(strategy, MODEL_CONFIGS["multilingual"])
    encoded_parts = []
    for model_name in model_names:
        try:
            model = load_model(model_name)
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            encoded_parts.append(vectors.astype("float32"))
            continue
        except Exception:
            pass

        tokenizer = load_tokenizer(model_name)
        model = load_auto_model(model_name)
        batches = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            with torch.no_grad():
                outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batches.append(pooled.cpu().numpy())
        vectors = np.concatenate(batches, axis=0)
        encoded_parts.append(_normalize(vectors.astype("float32")))
    return _merge_embeddings(encoded_parts).astype("float32")
