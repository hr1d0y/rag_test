from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

from .embeddings import embed_texts
from .utils import emit, percentile, read_jsonl, write_jsonl


@dataclass
class UnionFind:
    size: int

    def __post_init__(self):
        self.parent = list(range(self.size))
        self.rank = [0] * self.size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def _search_neighbors(vectors: np.ndarray, top_k: int = 25):
    if faiss is not None:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        scores, neighbors = index.search(vectors, min(top_k, len(vectors)))
        return scores, neighbors
    similarity = vectors @ vectors.T
    top_indices = np.argsort(-similarity, axis=1)[:, : min(top_k, len(vectors))]
    top_scores = np.take_along_axis(similarity, top_indices, axis=1)
    return top_scores, top_indices


def compute_adaptive_threshold(vectors: np.ndarray, base_threshold: float) -> float:
    if len(vectors) < 8:
        return base_threshold
    sample_count = min(1000, len(vectors))
    sample_vectors = vectors[:sample_count]
    sim = sample_vectors @ sample_vectors.T
    upper = sim[np.triu_indices_from(sim, k=1)]
    if len(upper) == 0:
        return base_threshold
    p90 = percentile(upper.tolist(), 0.90)
    if p90 > 0.92:
        return min(0.95, base_threshold + 0.03)
    if p90 < 0.75:
        return max(0.70, base_threshold - 0.02)
    return base_threshold


def cluster_leaf_rows(rows: List[Dict], strategy: str, base_threshold: float) -> List[Dict]:
    texts = [
        " ".join(
            [
                row.get("question", ""),
                row.get("answer", ""),
                row.get("explanation", ""),
            ]
        ).strip()
        for row in rows
    ]
    vectors = embed_texts(texts, strategy=strategy)
    threshold = compute_adaptive_threshold(vectors, base_threshold)
    scores, neighbors = _search_neighbors(vectors)
    uf = UnionFind(len(rows))
    for idx in range(len(rows)):
        for score, neighbor in zip(scores[idx], neighbors[idx]):
            if idx == int(neighbor):
                continue
            if float(score) >= threshold:
                uf.union(idx, int(neighbor))
    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[uf.find(idx)].append(row)
    clusters = []
    for cluster_idx, items in enumerate(grouped.values(), start=1):
        representative = items[0]
        clusters.append(
            {
                "cluster_id": f"{representative.get('leaf', 'leaf')}-{cluster_idx}",
                "parent": representative.get("parent", ""),
                "child": representative.get("child", ""),
                "leaf": representative.get("leaf", ""),
                "threshold": threshold,
                "strategy": strategy,
                "size": len(items),
                "representative_question": representative.get("question", ""),
                "representative_answer": representative.get("answer", ""),
                "representative_explanation": representative.get("explanation", ""),
                "items": items,
            }
        )
    return clusters


def run_phase2_cluster(
    input_file: str,
    output_dir: str,
    strategy: str = "hybrid",
    threshold: float = 0.82,
    progress_callback=None,
) -> Dict:
    rows = read_jsonl(input_file)
    leaves = defaultdict(list)
    for row in rows:
        leaves[row.get("leaf", "Unclassified")].append(row)
    emit(progress_callback, f"Clustering {len(rows)} rows across {len(leaves)} leaf topics...", phase=2)
    all_clusters: List[Dict] = []
    for leaf, leaf_rows in leaves.items():
        leaf_clusters = cluster_leaf_rows(leaf_rows, strategy, threshold)
        all_clusters.extend(leaf_clusters)
        emit(progress_callback, f"Leaf `{leaf}` clustered into {len(leaf_clusters)} groups.", phase=2)
    output_path = f"{output_dir}/phase2_clusters.jsonl"
    write_jsonl(output_path, all_clusters)
    return {
        "cluster_count": len(all_clusters),
        "row_count": len(rows),
        "output_file": output_path,
        "strategy": strategy,
        "threshold": threshold,
    }
