"""
Collect retrieval distance/rerank score distributions for threshold tuning.

Usage:
  python scripts/collect_retrieval_stats.py --input eval/queries.jsonl --output eval/stats.jsonl

Input jsonl format (one per line):
  {"q": "...", "label": 1}   # label optional; 1=answerable from internal docs, 0=not
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        yield json.loads(s)


def _quantiles(xs: list[float], ps: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    ys = sorted(xs)
    out: dict[str, float] = {}
    n = len(ys)
    for p in ps:
        if n == 1:
            out[f"p{int(p * 100)}"] = float(ys[0])
            continue
        # linear interpolation
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            out[f"p{int(p * 100)}"] = float(ys[lo])
        else:
            w = idx - lo
            out[f"p{int(p * 100)}"] = float(ys[lo] * (1 - w) + ys[hi] * w)
    return out


@dataclass
class Row:
    q: str
    label: int | None
    status: str
    best_distance: float | None
    gap_distance: float | None
    rerank_backend: str | None
    top1_rerank_score: float | None
    margin_rerank_score: float | None
    n_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "label": self.label,
            "status": self.status,
            "best_distance": self.best_distance,
            "gap_distance": self.gap_distance,
            "rerank_backend": self.rerank_backend,
            "top1_rerank_score": self.top1_rerank_score,
            "margin_rerank_score": self.margin_rerank_score,
            "n_hits": self.n_hits,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="jsonl file with {'q':..., 'label':...}"
    )
    ap.add_argument("--output", default="", help="optional jsonl output path")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rerank-top-n", type=int, default=5)
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument(
        "--no-rerank", action="store_true", help="disable cross-encoder rerank"
    )
    ap.add_argument(
        "--torch-threads", type=int, default=2, help="limit torch CPU threads"
    )
    ap.add_argument(
        "--progress-every", type=int, default=1, help="print progress every N queries"
    )
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output) if args.output else None

    # Reduce CPU contention on laptops.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(max(1, int(args.torch_threads)))
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass

    # Import after env/thread settings.
    from tools.retrieval import Retrieval

    tool = Retrieval()

    rows: list[Row] = []
    items = list(_iter_jsonl(inp))
    total = len(items)
    for idx, item in enumerate(items, start=1):
        q = str(item.get("q") or item.get("query") or "").strip()
        if not q:
            continue
        label = item.get("label", None)
        try:
            label = int(label) if label is not None else None
        except Exception:
            label = None

        t0 = time.perf_counter()
        tool_input = {
            "query": q,
            "top_k": args.top_k,
            "rerank_top_n": args.rerank_top_n,
            "rerank_model": args.rerank_model,
            "disable_rerank": bool(args.no_rerank),
            # We only need scores/distances; skip neighbor expansion to speed up.
            "expand_neighbors": False,
        }
        res = tool.run(tool_input, trace_id="eval")
        dt_ms = int((time.perf_counter() - t0) * 1000)
        if args.progress_every > 0 and (idx % args.progress_every == 0):
            print(f"[{idx}/{total}] {dt_ms}ms status={res.get('status')} q={q[:40]}")

        status = str(res.get("status", ""))
        stats = res.get("retrieval_stats") or {}
        best = stats.get("best_distance", None)
        gap = stats.get("gap_distance", None)
        try:
            best_f = float(best) if best is not None else None
        except Exception:
            best_f = None
        try:
            gap_f = float(gap) if gap is not None else None
        except Exception:
            gap_f = None

        hits = res.get("output") if isinstance(res, dict) else None
        hits = hits if isinstance(hits, list) else []
        scores = [h.get("rerank_score") for h in hits if isinstance(h, dict)]
        scores_f = []
        for s in scores:
            try:
                scores_f.append(float(s))
            except Exception:
                pass
        scores_f.sort(reverse=True)
        top1 = scores_f[0] if scores_f else None
        margin = (scores_f[0] - scores_f[1]) if len(scores_f) >= 2 else None

        rows.append(
            Row(
                q=q,
                label=label,
                status=status,
                best_distance=best_f,
                gap_distance=gap_f,
                rerank_backend=res.get("rerank_backend"),
                top1_rerank_score=top1,
                margin_rerank_score=margin,
                n_hits=len(hits),
            )
        )

    if outp:
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    def show_group(name: str, rs: list[Row]):
        bests = [r.best_distance for r in rs if isinstance(r.best_distance, float)]
        gaps = [r.gap_distance for r in rs if isinstance(r.gap_distance, float)]
        top1s = [
            r.top1_rerank_score for r in rs if isinstance(r.top1_rerank_score, float)
        ]
        margins = [
            r.margin_rerank_score
            for r in rs
            if isinstance(r.margin_rerank_score, float)
        ]
        print(f"\n== {name} (n={len(rs)}) ==")
        print("best_distance:", _quantiles(bests, [0.1, 0.5, 0.9]))
        print("gap_distance :", _quantiles(gaps, [0.1, 0.5, 0.9]))
        print("top1_score  :", _quantiles(top1s, [0.1, 0.5, 0.9]))
        print("margin_score:", _quantiles(margins, [0.1, 0.5, 0.9]))

    labels = {r.label for r in rows if r.label is not None}
    if labels:
        for lab in sorted(labels):
            show_group(f"label={lab}", [r for r in rows if r.label == lab])
    else:
        show_group("all", rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
