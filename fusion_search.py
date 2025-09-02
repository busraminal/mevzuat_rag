# -*- coding: utf-8 -*-
"""
fusion_search.py
- BM25 bag-of-words skoru + FAISS (embedding) skorunu birleştirir.
- Birleştirme: RRF (Reciprocal Rank Fusion) veya normalize ağırlıklı toplam.
- Chunk metinleri için corpus: index.chunks.jsonl / .json

Gereksinim:
    pip install rank_bm25
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
from nlp_helpers import normalize_text, expand_query_terms, tokenize

def _load_chunks(index_dir: Path) -> Dict[str, str]:
    """
    index.chunks.jsonl / .json → {chunk_id: text}
    Beklenen alanlar: {"id": "...", "text": "..."} veya {"chunk_id":"...","content":"..."}
    """
    candidates = [
        index_dir / "index.chunks.jsonl",
        index_dir / "index.chunks.json",
        index_dir / "chunks.jsonl",
        index_dir / "chunks.json",
    ]
    for p in candidates:
        if p.exists():
            data = {}
            if p.suffix == ".jsonl":
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip(): continue
                    obj = json.loads(line)
                    cid = obj.get("id") or obj.get("chunk_id")
                    txt = obj.get("text") or obj.get("content") or ""
                    if cid and txt:
                        data[str(cid)] = txt
            else:
                arr = json.loads(p.read_text(encoding="utf-8"))
                for obj in arr:
                    cid = obj.get("id") or obj.get("chunk_id")
                    txt = obj.get("text") or obj.get("content") or ""
                    if cid and txt:
                        data[str(cid)] = txt
            if data:
                return data
    raise FileNotFoundError("Chunk haritası bulunamadı (index.chunks.json[l])")

class FusionSearcher:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.chunks: Dict[str, str] = _load_chunks(self.index_dir)
        self.doc_ids: List[str] = list(self.chunks.keys())
        self.doc_tokens: List[List[str]] = [tokenize(normalize_text(self.chunks[cid])) for cid in self.doc_ids]
        self.bm25 = BM25Okapi(self.doc_tokens)

    def bm25_search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        terms = expand_query_terms(query)
        q_tokens = list(terms)
        scores = self.bm25.get_scores(q_tokens)
        pairs = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)[:k]
        return [(cid, float(sc)) for cid, sc in pairs]

    @staticmethod
    def _to_ranks(pairs: List[Tuple[str, float]], higher_is_better=True) -> Dict[str, int]:
        rev = sorted(pairs, key=lambda x: x[1], reverse=not higher_is_better)
        return {cid: i+1 for i, (cid, _) in enumerate(rev)}

    def fuse(
        self,
        faiss_hits: List[Tuple[str, float]],
        bm25_hits:  List[Tuple[str, float]],
        method: str = "rrf",
        alpha: float = 0.7,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        method='rrf'  → Reciprocal Rank Fusion (k=60)
        method='wsum' → 0-1 normalize + alpha*faiss + (1-alpha)*bm25
        """
        if method == "rrf":
            k = 60
            r_faiss = self._to_ranks(faiss_hits, higher_is_better=True)
            r_bm25  = self._to_ranks(bm25_hits,  higher_is_better=True)
            all_ids = set(r_faiss) | set(r_bm25)
            fused = {}
            for cid in all_ids:
                s = 0.0
                if cid in r_faiss: s += 1.0 / (k + r_faiss[cid])
                if cid in r_bm25:  s += 1.0 / (k + r_bm25[cid])
                fused[cid] = s
            return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        def _normalize(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
            if not pairs: return {}
            vals = [s for _, s in pairs]
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return {cid: 0.0 for cid, _ in pairs}
            return {cid: (s - mn) / (mx - mn) for cid, s in pairs}

        n_faiss = _normalize(faiss_hits)
        n_bm25  = _normalize(bm25_hits)
        all_ids = set(n_faiss) | set(n_bm25)
        fused = {cid: alpha*n_faiss.get(cid,0.0) + (1-alpha)*n_bm25.get(cid,0.0) for cid in all_ids}
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
