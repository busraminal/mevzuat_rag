# pipeline.py
# Uçtan uca soru-cevap akışı: HybridRetriever -> (CrossEncoder) Rerank -> Answerer
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Not: Burada adapters ve reranker dosyalarındaki sınıfların imzalarına güveniyoruz.
# HybridRetriever: retrieve(query: str, k: int) -> List[dict]  (alternatif: search(...))
# MiniLMReranker:  rerank(query: str, candidates: List[dict], top_k: int) -> List[dict]
# Answerer:        answer(query: str, contexts: List[dict], max_context: int) -> str

# pipeline.py (QAConfig)
class QAConfig:
    k_hybrid: int = 20
    rerank_top: int = 8
    context_top: int = 4         # <-- 4 bağlam yeterli
    text_key: str = "text"
    min_chars: int = 20        # çok kısa/boş parçaları atmak için alt sınır

class QAEngine:
    def __init__(self, hybrid_retriever, reranker, answerer, cfg: Optional[QAConfig] = None):
        """
        :param hybrid_retriever: HybridRetriever örneği (BM25+FAISS+RRF)
        :param reranker: MiniLMReranker (CrossEncoder tabanlı)
        :param answerer: Answerer (LLM generate wrapper)
        :param cfg: QAConfig
        """
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.answerer = answerer
        self.cfg = cfg or QAConfig()

    # ---- İç yardımcılar -----------------------------------------------------
    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Hybrid retriever’dan adayları alır. API imzası farklı ise tolere eder.
        Dönüş: [{"id": int, "text": str, "page": int, "file": str, "score": float, ...}, ...]
        """
        k = self.cfg.k_hybrid
        # Bazı implementasyonlar retrieve(), bazıları search() kullanır
        if hasattr(self.retriever, "retrieve"):
            cands = self.retriever.retrieve(query, k=k)
        elif hasattr(self.retriever, "search"):
            cands = self.retriever.search(query, k=k)
        else:
            raise RuntimeError("HybridRetriever 'retrieve' ya da 'search' metoduna sahip değil.")

        # Basit temizlik
        out: List[Dict[str, Any]] = []
        for c in cands or []:
            text = c.get(self.cfg.text_key) or ""
            if not isinstance(text, str):
                continue
            if len(text.strip()) < self.cfg.min_chars:
                continue
            out.append(c)
        return out

    def _rerank(self, query: str, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """CrossEncoder ile yeniden sıralama (mevcutsa)."""
        if self.reranker is None:
            return cands[: self.cfg.rerank_top]
        return self.reranker.rerank(query, cands, top_k=self.cfg.rerank_top, text_key=self.cfg.text_key)

    # ---- Dış API -------------------------------------------------------------
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Soruya yanıt üretir ve kaynakları döndürür.
        return: {
            "answer": str,
            "contexts": [ {file, page, text, ce_score?, score?, rank}, ... ]
        }
        """
        # 1) Retrieve
        cands = self._retrieve(query)

        # 2) Rerank (CrossEncoder) — yoksa hızlıca ilk N
        reranked = self._rerank(query, cands)

        # 3) LLM cevabı
        answer = self.answerer.answer(query, reranked, max_context=self.cfg.context_top)

        return {
            "answer": answer,
            "contexts": reranked[: self.cfg.context_top],
        }


# ------------------ CLI Duman Testi (opsiyonel) ------------------
if __name__ == "__main__":
    # Burada boot.build_engine’i çağırıp hızlı bir test yapıyoruz.
    try:
        from boot import build_engine  # dairesel bağımlılığı önlemeye dikkat; sadece CLI’da çağrılır
        eng = build_engine()
        print("[i] QAEngine hazır. Çıkış için q/quit.")
        while True:
            try:
                q = input("\nSoru: ").strip()
            except EOFError:
                break
            if not q or q.lower() in {"q", "quit", "exit"}:
                break
            out = eng.ask(q)
            print("\n=== Yanıt ===")
            print(out["answer"])
            print("\n=== Kaynaklar ===")
            for i, c in enumerate(out["contexts"], 1):
                f = c.get("file", "?")
                p = c.get("page", "?")
                cs = c.get("ce_score", c.get("score"))
                print(f"{i}. {f} | sayfa {p} | ce_score/score={cs}")
    except Exception as e:
        print("Duman testi atlandı/başarısız:", e)
