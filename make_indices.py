# make_indices.py
# -*- coding: utf-8 -*-
"""
PDF → CHUNKS → (chunks.jsonl, index.faiss, bm25_index.pkl, meta.jsonl)
=====================================================================

Amaç
----
- data/pdfs/indexed/ altındaki PDF'leri okuyup, sayfa bazlı temizlemek,
- sabit uzunluklu (ve MADDE başlığına duyarlı) chunk'lara bölmek,
- bu chunk listesini `persistence.index_from_chunks(...)` fonksiyonuna verip
  * FAISS vektör indeksi (index.faiss),
  * BM25 dizini (bm25_index.pkl),
  * chunk metadatası (data/chunks.jsonl)
  üretmek.

Notlar
------
- Bu dosya **sadece chunk üretimi ve indeks çağrısını** yapar. Embedding ve BM25
  oluşturma detayları `persistence.py` içindedir (aynı proje kökünde olmalı).
- `pdf_gap_checker.py` ve `fusion_search.py` bu çıktı dosyalarını kullanır.
- `chunks.jsonl` satır sırası → FAISS vektör sırasıyla **aynı olmalıdır**. 
  (Genelde `index_from_chunks` böyle garanti eder.)

Gerekenler
----------
- pypdf veya pymupdf(fitz)
- sentence-transformers, faiss-cpu (persistence tarafı için)
- rank_bm25 (BM25 için)
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterable

# Kalıcı indeksleme/klasör yardımcıları (aynı proje kökünde olmalı)
from persistence import index_from_chunks, ensure_dirs

# -------------------------------------------------------------------
# DİZİN / GİRİŞ ÇIKTI SABİTLERİ
# -------------------------------------------------------------------
# Proje kökü: bu dosya ile aynı klasör
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data"

# PDF'lerin toplanacağı klasör (indekslemek istediklerinizi buraya koyun)
PDF_DIR: Path = DATA_DIR / "pdfs" / "indexed"

# Eski akışlarla uyum için meta dosyası (özet mapping)
META_PATH: Path = DATA_DIR / "meta.jsonl"

# -------------------------------------------------------------------
# CHUNK'LAMA AYARLARI (varsayılan)
# -------------------------------------------------------------------
CHUNK_CHARS_DEFAULT   = 800   # Her chunk'ın yaklaşık karakter uzunluğu
CHUNK_OVERLAP_DEFAULT = 100   # Komşu chunk'lar arası bindirme (bağlam kaybını azaltır)
MAX_PAGE_CHARS        = 500_000  # Çok büyük/bozuk sayfaları sınırlama

# "MADDE 1", "MADDE 2A" vb. başlıklarını yakalamak için basit desen
MADDE_RE = re.compile(r"\b(MADDE\s*\d+[A-Z]?)\b", re.IGNORECASE)


# ==============================
# Yardımcı: Temel metin normalize
# ==============================
def _normalize(text: str) -> str:
    """
    Basit metin temizliği:
    - BOM / kontrol karakterleri / çoklu boşluk
    - CRLF → LF
    - aşırı boş satırları azaltma
    """
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =====================================
# PDF Okuma: PyMuPDF → pypdf fallback
# =====================================
def _read_pdf_pymupdf(p: Path) -> List[str]:
    """
    PyMuPDF (fitz) ile her sayfayı düz metin olarak al.
    Hızlı ve genellikle daha iyi yerleşim verir.
    """
    import fitz  # PyMuPDF
    doc = fitz.open(p)
    pages: List[str] = []
    for i in range(len(doc)):
        t = doc.load_page(i).get_text("text") or ""
        pages.append(t[:MAX_PAGE_CHARS])
    doc.close()
    return pages

def _read_pdf_pypdf(p: Path) -> List[str]:
    """
    pypdf ile metin çıkarımı. PyMuPDF yoksa veya hata verirse buraya düşer.
    """
    from pypdf import PdfReader
    reader = PdfReader(str(p))
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t[:MAX_PAGE_CHARS])
    return pages

def read_pdf_pages(p: Path) -> List[str]:
    """
    PDF → sayfa metin listesi.
    Önce PyMuPDF dener, hata/veri yoksa pypdf'e düşer.
    """
    try:
        return _read_pdf_pymupdf(p)
    except Exception:
        return _read_pdf_pypdf(p)


# =========================================
# Chunk üretimi (MADDE duyarlı sabit pencereler)
# =========================================
def _yield_chunks_from_text(txt: str, chunk_chars: int, overlap: int) -> Iterable[str]:
    """
    Sabit uzunlukta pencereler üretir; fakat pencere içinde 'MADDE ...' başlığı
    bulunursa kesim yerini bu başlığa yaklaştırır (böylece madde ortasından bölmek
    yerine madde başında başlatmaya çalışır).
    """
    n = len(txt)
    if n <= chunk_chars:
        # Kısa metinler tek parça
        yield txt
        return

    i = 0
    while i < n:
        j = min(i + chunk_chars, n)
        window = txt[i:j]

        # Pencere içinde MADDE başlığı varsa, kesimi başlığa hizalamayı dene
        matches = list(MADDE_RE.finditer(window))
        if matches:
            # En sondaki başlığı bul; bir sonraki pencerenin başını buraya kaydır
            last_m = matches[-1]
            if last_m.start() > 0 and (j < n):
                j = i + last_m.start()
                window = txt[i:j]

        if window.strip():
            yield window.strip()

        if j >= n:
            break
        # Overlap ile bir miktar geri sar (bağlam devamı)
        i = max(0, j - overlap)

def chunk_pages(pages: List[str], *, chunk_chars: int, overlap: int) -> List[str]:
    """
    Tüm sayfalar için normalize → chunk.
    """
    chunks: List[str] = []
    for page_text in pages:
        t = _normalize(page_text)
        if not t:
            continue
        for ch in _yield_chunks_from_text(t, chunk_chars, overlap):
            chunks.append(ch)
    return chunks


# =========================================
# Tek PDF → Çoklu chunk kaydı (id/file/page/text)
# =========================================
def build_chunks_for_pdf(pdf_path: Path, start_id: int, *, chunk_chars: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Verilen PDF için:
      - sayfaları oku,
      - normalize et,
      - chunk'lara böl ve
      - her bir chunk'a artan bir 'id' ata.
    """
    pages = read_pdf_pages(pdf_path)
    out: List[Dict[str, Any]] = []
    cid = start_id
    for page_no, page_text in enumerate(pages, start=1):
        page_text = _normalize(page_text)
        if not page_text:
            continue
        for ch in _yield_chunks_from_text(page_text, chunk_chars, overlap):
            out.append({
                "id": cid,             # (int) — FAISS sırasıyla birebir aynı tutulur
                "file": pdf_path.name, # (str) — kaynak PDF ismi
                "page": page_no,       # (int) — PDF sayfa numarası (1-indexed)
                "text": ch             # (str) — chunk metni (ham metin; 'passage:' ekini persistence ekleyebilir)
            })
            cid += 1
    return out


# =========================================
# Backward-compat: meta.jsonl (isteğe bağlı)
# =========================================
def write_meta_jsonl(chunks: List[Dict[str, Any]], path: Path = META_PATH) -> None:
    """
    Eski akışlar için basit bir meta eşleme (her satır: text / page / file / id).
    `pdf_gap_checker.py` doğrudan bunu kullanmıyor; ancak başka yerlerde
    hafif referans olarak işine yarayabilir.
    """
    ensure_dirs()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({
                "text": ch["text"],
                "page": ch.get("page"),
                "file": ch.get("file"),
                "id": ch.get("id")
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# =========================================
# CLI / MAIN
# =========================================
def main() -> None:
    """
    - data/pdfs/indexed/ içindeki PDF'leri gez,
    - chunk'la,
    - 'persistence.index_from_chunks' ile FAISS + BM25 + chunks.jsonl üret.
    """
    parser = argparse.ArgumentParser(description="PDF'leri chunk'la ve FAISS+BM25 indeksleri üret.")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=str(PDF_DIR),
        help="İndekslenecek PDF klasörü (varsayılan: data/pdfs/indexed)"
    )
    parser.add_argument(
        "--chunk_chars",
        type=int,
        default=CHUNK_CHARS_DEFAULT,
        help=f"Chunk hedef uzunluğu (varsayılan: {CHUNK_CHARS_DEFAULT})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP_DEFAULT,
        help=f"Chunk bindirmesi (varsayılan: {CHUNK_OVERLAP_DEFAULT})"
    )
    args = parser.parse_args()

    # 1) Klasörleri garantiye al
    ensure_dirs()

    # 2) PDF klasörünü doğrula
    pdf_dir = Path(args.pdf_dir)
    assert pdf_dir.exists(), f"PDF klasörü yok: {pdf_dir}"

    # 3) PDF listesini topla
    all_pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    if not all_pdfs:
        print(f"[!] {pdf_dir} içinde PDF bulunamadı.")
        return

    print(f"[i] PDF sayısı: {len(all_pdfs)}")

    # 4) Tüm PDF'lerden chunk listesi oluştur
    all_chunks: List[Dict[str, Any]] = []
    cur_id = 0
    for pdf in all_pdfs:
        print(f"[i] İşleniyor: {pdf.name}")
        pdf_chunks = build_chunks_for_pdf(
            pdf,
            start_id=cur_id,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap
        )
        all_chunks.extend(pdf_chunks)
        cur_id = all_chunks[-1]["id"] + 1 if all_chunks else 0
        print(f"    -> {len(pdf_chunks)} chunk")

    print(f"[i] Toplam chunk: {len(all_chunks)}")

    # 5) (Opsiyonel) meta.jsonl — yalnızca geriye dönük uyum/diagnostic için
    write_meta_jsonl(all_chunks, META_PATH)
    print(f"[✓] Yazıldı: {META_PATH}")

    # 6) KALICI İNDEKSLER
    #    - persistence.index_from_chunks, aşağıdaki dosyaları yazar:
    #        * data/chunks.jsonl      : FAISS sırasına eşit satır sıralı chunk kayıtları
    #        * data/index.faiss       : FAISS IndexFlatIP (veya persistence içinde seçilen tip)
    #        * data/bm25_index.pkl    : BM25 dizini (Fusion için)
    #
    #    ÖNEM: 'chunks.jsonl' satır sırasının FAISS ile birebir aynı olmasını persistence sağlar.
    index_from_chunks(all_chunks)
    print("[✓] Yazıldı: data/chunks.jsonl, data/index.faiss, data/bm25_index.pkl")

    print("\n[OK] İndeksleme tamam. Şimdi 'pdf_gap_checker.py' ile yeni bir PDF karşılaştırabilirsiniz.")


if __name__ == "__main__":
    main()
