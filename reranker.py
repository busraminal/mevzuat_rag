# reranker.py
# CrossEncoder tabanlı yeniden sıralayıcı (reranker)
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Tuple, Optional
import math

try:
    import numpy as np
except Exception:  # numpy yoksa çalışma anında hata vermesin
    np = None


def _chunked(iterable: Iterable, n: int):
    """Basit chunk'lama (predict batch'leri)."""
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


class MiniLMReranker:
    """
    Sentence-Transformers CrossEncoder ile rerank.
    Kullanım:
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        rer = MiniLMReranker(ce)
        out = rer.rerank(query, docs, top_k=10)
    """
    def __init__(self, cross_encoder, batch_size: int = 64, normalize: bool = False):
        """
        :param cross_encoder: sentence_transformers.CrossEncoder örneği
        :param batch_size: predict sırasında kullanılacak batch boyutu
        :param normalize: True ise skorları 0..1 aralığına sigmoid ile sıkıştırır
        """
        self.model = cross_encoder
        self.batch_size = max(1, int(batch_size))
        self.normalize = bool(normalize)

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Modelin predict'ini güvenli şekilde (batch'li) çalıştırır."""
        scores: List[float] = []
        # CrossEncoder.predict zaten batch'i kendi yönetiyor ama
        # bazı ortamlarda stabilite için manuel batch tercih ediliyor.
        for chunk in _chunked(pairs, self.batch_size):
            s = self.model.predict(chunk)  # -> np.ndarray veya list
            if isinstance(s, list):
                scores.extend(float(x) for x in s)
            else:
                # numpy array ise
                scores.extend([float(x) for x in s.tolist()])
        if self.normalize:
            # Sigmoid normalizasyon (opsiyonel)
            scores = [1.0 / (1.0 + math.exp(-x)) for x in scores]
        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        :param query: Kullanıcı sorgusu
        :param candidates: [{"text": "...", "id": ..., "score": ..., "file": ..., "page": ...}, ...]
        :param top_k: Kaç adet döndürülecek
        :param text_key: Metnin bulunduğu alan adı
        :return: En iyi top_k aday; her elemana ce_score ve rank eklenir
        """
        if not candidates:
            return []

        # Eksik text'leri filtrele
        pool = [c for c in candidates if c.get(text_key)]
        if not pool:
            return []

        # (query, passage) çiftlerini hazırla
        pairs = [(query, c[text_key]) for c in pool]

        # Skorla
        scores = self._score_pairs(pairs)

        # Skorları ekleyip sırala
        for c, s in zip(pool, scores):
            c["ce_score"] = float(s)
        pool.sort(key=lambda x: x["ce_score"], reverse=True)

        # top_k ve rank ata (1-based)
        top = pool[: max(1, int(top_k))]
        for i, c in enumerate(top, start=1):
            c["rank"] = i
        return top


# ---- Hızlı duman testi ----
if __name__ == "__main__":
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        rr = MiniLMReranker(ce, batch_size=32, normalize=False)
        q = "Yönetmeliğin amacı nedir?"
        docs = [
            {"id": 1, "text": "MADDE 1 – Bu yönetmeliğin amacı ..."},
            {"id": 2, "text": "Tanımlar başlıklı madde, bu yönetmelikte geçen terimleri açıklar."},
            {"id": 3, "text": "Yürürlük maddesi, yayım tarihinde yürürlüğe girer."},
        ]
        out = rr.rerank(q, docs, top_k=2)
        for d in out:
            print(d["id"], d["ce_score"], d["text"][:60].replace("\n", " "))
    except Exception as e:
        print("Duman testi atlandı/başarısız:", e)
