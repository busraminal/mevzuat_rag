# adapters.py
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

__all__ = ["BM25Adapter", "FAISSAdapter", "HybridRetriever", "rrf_merge"]

class BaseRetriever:
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        raise NotImplementedError

class BM25Adapter(BaseRetriever):
    def __init__(self, bm25: BM25Okapi, corpus_meta: List[Dict[str, Any]]):
        self.bm25 = bm25
        self.meta = corpus_meta  # meta[i]["text"] doküman gövdesi

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:max(k, 50)]
        out = []
        for i in idxs:
            m = self.meta[i]
            out.append({
                "text": m["text"],
                "meta": {k:v for k,v in m.items() if k != "text"},
                "score": float(scores[i])
            })
        return out

class FAISSAdapter(BaseRetriever):
    def __init__(self, index: faiss.Index, encoder: SentenceTransformer, corpus_meta: List[Dict[str, Any]]):
        self.index = index
        self.encoder = encoder
        self.meta = corpus_meta  # sıra FAISS vektör sırasıyla birebir

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        qv = self.encoder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.asarray(qv, dtype=np.float32), max(k, 50))
        D, I = D[0], I[0]
        out = []
        for idx, dist in zip(I, D):
            if idx < 0:
                continue
            m = self.meta[idx]
            out.append({
                "text": m["text"],
                "meta": {k:v for k,v in m.items() if k != "text"},
                "score": float(dist)
            })
        return out

def rrf_merge(results_list: List[List[Dict[str, Any]]], k: int = 20, k_rrf: int = 60) -> List[Dict[str, Any]]:
    scores: Dict[str, Dict[str, Any]] = {}
    for results in results_list:
        for rank, item in enumerate(results):
            meta = item.get("meta", {})
            cid = meta.get("chunk_id") or f"{meta.get('doc_id','?')}:{meta.get('page','?')}:{hash(item['text'])}"
            if cid not in scores:
                scores[cid] = {"item": item, "score": 0.0}
            scores[cid]["score"] += 1.0 / (k_rrf + rank + 1)
    merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [m["item"] | {"rrf_score": float(m["score"])} for m in merged[:k]]

class HybridRetriever(BaseRetriever):
    def __init__(self, bm25: BM25Adapter, faiss_adp: FAISSAdapter, k_rrf: int = 60):
        self.bm25 = bm25
        self.faiss = faiss_adp
        self.k_rrf = k_rrf

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(self.bm25.search, query, max(k, 50))
            f2 = ex.submit(self.faiss.search, query, max(k, 50))
            r1, r2 = f1.result(), f2.result()
        return rrf_merge([r1, r2], k=k, k_rrf=self.k_rrf)
