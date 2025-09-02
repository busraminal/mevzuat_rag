# persistence.py
# -*- coding: utf-8 -*-
"""
Chunk'ları kalıcı saklama ve indeksleri (FAISS + BM25) yazma / yükleme yardımcıları
====================================================================================

Bu modül; indeksleme hattının "kalıcı veri katmanı" gibidir:
- chunks.jsonl (ham metin + meta)
- index.faiss   (dense; E5 embedding'leri ile cosine ~ dot-product)
- bm25_index.pkl (sparse; bag-of-words)
dosyalarını üretir ve geri yükler.

Tasarım notları
---------------
1) E5 Prompt Disiplini:
   - Passage (korpus) için encode ederken "passage: {text}" formatı kullanılır.
     (Eğer metin zaten "passage: " ile başlıyorsa tekrar eklemeyiz.)
   - Sorgu tarafında (pdf_gap_checker / fusion_search) "query: {text}" kullanılmalı.

2) Sıra Bütünlüğü:
   - `index_from_chunks(...)` FAISS'e embedding'leri **chunk listesinin aynı sırası** ile yazar.
     Böylece `data/chunks.jsonl` satır N ↔ FAISS vektör N eşleşmesi korunur.

3) BM25:
   - Basit bir tokenizer var; dil-özel geliştirme (stopword, stemmer) istersen burada yapılır.

4) Dosya Güvenliği:
   - JSONL yazımlarında atomic temp dosyası kullanılır (yarım dosya kalmasın).

Kullanım
--------
- make_indices.py içinde index_from_chunks(all_chunks) çağrıları buraya düşer.
- fusion_search / pdf_gap_checker bu modülün ürettiği dosyaları okuyarak çalışır.
"""

from __future__ import annotations

import os
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================ YOL / AYAR SABİTLERİ ============================

DATA_DIR    = Path("data")
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"       # ham text + meta (kalıcı)
FAISS_PATH  = DATA_DIR / "index.faiss"        # dense index (FAISS)
BM25_PATH   = DATA_DIR / "bm25_index.pkl"     # sparse index (pickle)
EMB_MODEL   = "intfloat/multilingual-e5-large"  # E5-büyük çokdilli (iyi kalite)


# =============================== DİR/IO YARDIMCILAR ==========================

def ensure_dirs() -> None:
    """data/ vb. klasörleri garantiye al."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, lines: Iterable[str]) -> None:
    """
    JSONL gibi 'satır satır' dosyalar için güvenli yazım.
    Önce *.tmp'ye yazar; başarılıysa atomik rename ile hedefe taşır.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln)
            if not ln.endswith("\n"):
                f.write("\n")
    os.replace(tmp, path)


# ================================= CHUNKS I/O ================================

def save_chunks_jsonl(chunks: List[Dict[str, Any]], path: Path = CHUNKS_PATH) -> None:
    """
    Chunk formatı (örnek):
      {"id": 0, "text": "MADDE 1 ...", "page": 1, "file": "kanun.pdf"}
    Not: Burada metne "passage:" ÖNEKİ EKLEMEYİZ (ham saklıyoruz).
         Embedding sırasında önek eklenir (bkz. build_embeddings).
    """
    ensure_dirs()
    lines = (json.dumps(ch, ensure_ascii=False) for ch in chunks)
    atomic_write_text(path, lines)


def load_chunks_jsonl(path: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    """JSONL'den chunk listesini geri yükle."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ========================= EMBEDDINGS / FAISS (DENSE) ========================

def _ensure_passage_prefix(text: str) -> str:
    """
    E5 passage prompt disiplini: "passage: {text}"
    Metin zaten 'passage:' ile başlıyorsa ikinci kez eklemeyelim.
    """
    t = text.strip()
    if t.lower().startswith("passage:"):
        return t
    return f"passage: {t}"


def build_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = EMB_MODEL,
    batch_size: int = 32
) -> np.ndarray:
    """
    Chunk metinlerini E5 ile encode eder.
    - Normalize_embeddings=True → cosine ~ dot-product için uygun.
    - float32 → FAISS gerekliliği.
    - Sıra → chunks sırası ile aynı tutulur.
    """
    model = SentenceTransformer(model_name)

    # Passage önekini burada ekliyoruz (ham saklama ile arama disiplinini ayırmak daha sağlıklı)
    texts = [_ensure_passage_prefix(ch["text"]) for ch in chunks]

    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # FAISS, C-contiguous float32 bekler
    emb = np.ascontiguousarray(emb.astype("float32"))
    return emb


def save_faiss(embeddings: np.ndarray, path: Path = FAISS_PATH) -> None:
    """
    Embedding matrisini (N x D) bir IndexFlatIP içine yazar.
    IndexFlatIP: normalize vektörlerle dot = cosine.
    """
    ensure_dirs()
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # sırayla eklenir → satır N = vektör N
    faiss.write_index(index, str(path))


def load_faiss(path: Path = FAISS_PATH) -> faiss.Index:
    """FAISS indeksini diskteki dosyadan yükle."""
    return faiss.read_index(str(path))


# ================================ BM25 (SPARSE) ==============================

_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+")

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Basit TR dostu tokenize:
    - Harf/rakam dışını bölücü sayar; unicode TR karakterlerini de kapsar.
    - İstersen burada stop-word/stem ekleyebilirsin.
    """
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def build_bm25(chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    BM25 Okapi modelini kurar ve meta ile birlikte döndürür.
    BM25, tokenized doküman listesi alır → skorlamak için query de tokenized edilmelidir.
    """
    tokenized_docs = [tokenize_for_bm25(ch["text"]) for ch in chunks]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, chunks  # meta = chunks (id/page/file/text) aynı sırada


def save_bm25(bm25: BM25Okapi, meta: List[Dict[str, Any]], path: Path = BM25_PATH) -> None:
    """
    BM25 modelini ve eşleşen meta listesini tek pickle içine at.
    Not: BM25Okapi pickle'lanabilir (rank_bm25: saf Python).
    """
    ensure_dirs()
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bm25(path: Path = BM25_PATH) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """pickle'dan BM25 ve meta'yı (dokümanlar) geri yükle."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["meta"]


# ============================ END-TO-END ÜRETİCİ ============================

def index_from_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = EMB_MODEL,
    batch_size: int = 32
) -> None:
    """
    Verilen chunk listesini kalıcı depolara yazar:
      - data/chunks.jsonl   (ham metin + meta)
      - data/index.faiss    (E5 embedding'leri; IndexFlatIP)
      - data/bm25_index.pkl (BM25 + meta)

    ÖNEM: Sıra bütünlüğü korunur (FAISS satır N ↔ chunks[N]).
    """
    # 1) Ham chunk metni/metadata → JSONL
    save_chunks_jsonl(chunks)

    # 2) Dense embeddings → FAISS
    emb = build_embeddings(chunks, model_name=model_name, batch_size=batch_size)
    save_faiss(emb)

    # 3) Sparse → BM25 (tokenize + pickle)
    bm25, meta = build_bm25(chunks)
    save_bm25(bm25, meta)


def reload_all() -> Tuple[List[Dict[str, Any]], faiss.Index, BM25Okapi]:
    """
    Diskten hepsini geri yükle:
      - chunks.jsonl → chunks
      - index.faiss  → FAISS index
      - bm25_index.pkl → BM25 + meta (meta gerekirse ayrıca döndürülebilir)
    """
    chunks = load_chunks_jsonl()
    faiss_index = load_faiss()
    bm25, _meta = load_bm25()
    return chunks, faiss_index, bm25


# ================================ HIZLI DEMO ================================

if __name__ == "__main__":
    # Küçük bir örnek kurulum
    demo_chunks = [
        {"id": 0, "file": "yönetmelik.pdf", "page": 1, "text": "MADDE 1 – Bu yönetmeliğin amacı..."},
        {"id": 1, "file": "yönetmelik.pdf", "page": 2, "text": "MADDE 2 – Tanımlar..."},
        {"id": 2, "file": "tebliğ.pdf",     "page": 5, "text": "MADDE 5 – Uygulama esasları..."},
    ]

    index_from_chunks(demo_chunks)

    chunks, faiss_index, bm25 = reload_all()
    print(f"[OK] Yüklenen chunk sayısı: {len(chunks)} | FAISS ntotal={faiss_index.ntotal}")
