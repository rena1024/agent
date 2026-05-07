from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RerankResult:
    hits: List[Dict[str, Any]]
    backend: str  # "cross-encoder" | "none"


class CrossEncoderReranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None
    ):
        from sentence_transformers import CrossEncoder

        # CrossEncoder will use torch under the hood.
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, hits: List[Dict[str, Any]], top_n: int
    ) -> RerankResult:
        pairs: List[Tuple[str, str]] = []
        for h in hits:
            pairs.append((query, str(h.get("text", ""))))
        scores = self.model.predict(pairs)
        scored = []
        for h, s in zip(hits, scores):
            nh = dict(h)
            nh["rerank_score"] = float(s)
            scored.append(nh)
        scored.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
        return RerankResult(hits=scored[:top_n], backend="cross-encoder")


# Process-wide cache to avoid re-loading weights for every query.
_RERANKER_CACHE: dict[tuple[str, Optional[str]], CrossEncoderReranker] = {}


def _get_reranker(model_name: str, device: Optional[str]) -> CrossEncoderReranker:
    key = (model_name, device)
    rr = _RERANKER_CACHE.get(key)
    if rr is not None:
        return rr
    rr = CrossEncoderReranker(model_name=model_name, device=device)
    _RERANKER_CACHE[key] = rr
    return rr


def rerank_hits(
    query: str,
    hits: List[Dict[str, Any]],
    *,
    top_n: int = 2,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: Optional[str] = None,
) -> RerankResult:
    """
    Rerank retrieved hits using a cross-encoder reranker.
    If dependencies/model are unavailable, returns hits unchanged.
    """
    if not hits:
        return RerankResult(hits=[], backend="none")

    try:
        reranker = _get_reranker(model_name=model_name, device=device)
        return reranker.rerank(query, hits, top_n=top_n)
    except Exception:  # noqa: BLE001
        return RerankResult(hits=hits[:top_n], backend="none")
