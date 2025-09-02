# -*- coding: utf-8 -*-
"""
PDF -> Mevzuat Kıyas (Eksik / Çelişki / Uyum)
=============================================
- Önkoşul: Mevzuat korpusu E5 (passage:) ile embedlenip FAISS IndexFlatIP'e yazılmış olmalı
  (index.faiss + index.chunks.jsonl / .json). "chunks" dosyasının satır sırası FAISS vektör sırasıyla
  aynı olmalıdır (make_indices.py genelde böyle üretir).

- Bu araç: Yeni yüklenen PDF'yi chunk'lar, her chunk'ı E5 (query:) ile embedler,
  FAISS'te aratır ve (opsiyonel) BM25 ile füzyon yapar; eşik/kurallara göre sınıflar,
  sayısal/kural/madde farklarını tespit eder.

Kullanım:
    pip install -U sentence-transformers faiss-cpu pymupdf rich rank_bm25
    # (PyMuPDF yoksa otomatik pypdf'e düşer)
    python pdf_gap_checker.py --pdf path/to/new.pdf --index data/index.faiss \
        --out reports/ --topk 3 --hi 0.80 --lo 0.60 \
        --fusion on --fusion_method rrf --alpha 0.7 --bm25_k 120 --faiss_mult 3

Çıktılar:
- {out}/report_YYYYMMDD_HHMMSS.json  (özet + bulgular)
- {out}/report_YYYYMMDD_HHMMSS.csv   (tablo)
- Konsolda özet kartlar + ilk bulgular

Notlar:
- Füzyon kapalıysa (varsayılan off) eski FAISS-only davranışı korunur.
- Füzyon açık ama modüller kurulu değilse (fusion_search / nlp_helpers) otomatik FAISS-only'a düşer.
"""
import unicodedata  # TR aksanlarını ASCII'ye indirgemek için (yüzde->yuzde vb.)
import os, re, json, csv, argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------
# (OPSİYONEL) Füzyon arama modülleri — yoksa FAISS-only'a düşer
# -------------------------------------------------------------
FUSION_AVAILABLE = True
try:
    from fusion_search import FusionSearcher
except Exception:
    FUSION_AVAILABLE = False


# =============================================================
# 1) PDF OKUMA (PyMuPDF hızlı; yoksa pypdf'e düş)
# =============================================================
def read_pdf_pages(pdf_path: Path) -> List[str]:
    """
    Girdi PDF dosyasını sayfa sayfa okur ve düz metin listesi döndürür.
    Önce PyMuPDF (fitz) denenir; başarısız olursa pypdf'e düşülür.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(len(doc)):
            text = doc.load_page(i).get_text("text") or ""
            pages.append(text)
        doc.close()
        return pages
    except Exception:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = []
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
        return pages


# =============================================================
# 2) TEMİZLİK ve CHUNKING
# =============================================================
SPACES_RE = re.compile(r"[ \t\r\f\v]+")

def clean_text(t: str) -> str:
    """PDF metnini normalize eder (boşluklar, NBSP, satır sonları…)."""
    t = t.replace("\u00a0", " ")
    t = SPACES_RE.sub(" ", t)
    t = re.sub(r"\s+\n\s+", "\n", t)
    return " ".join(t.split()).strip()

def chunk_pages(pages: List[str], chunk_chars: int = 900, overlap: int = 120) -> List[Dict[str, Any]]:
    """
    Sayfaları yaklaşık 'chunk_chars' uzunlukta parçalar.
    Overlap, cümlelerin bölünmesinden doğan bağlam kaybını azaltır.
    """
    chunks = []
    global_idx = 0
    for pi, page in enumerate(pages, start=1):
        txt = clean_text(page)
        if not txt:
            continue
        start = 0
        while start < len(txt):
            end = min(start + chunk_chars, len(txt))
            piece = txt[start:end]
            if piece.strip():
                global_idx += 1
                cid = f"p{pi:03d}_c{global_idx:04d}"
                chunks.append({"id": cid, "idx": global_idx, "page": pi, "text": piece})
            if end == len(txt):
                break
            start = max(0, end - overlap)
    return chunks


# =============================================================
# 3) FAISS İNDEKS + CHUNK MAP (id/text) YÜKLEME
# =============================================================
def load_faiss_with_map(index_path: Path):
    """
    FAISS indeksini ve 'chunks' haritasını (id + text) yükler.
    Dönüş:
        index: FAISS nesnesi
        corpus_texts: List[str]  (FAISS sırasına göre)
        corpus_ids:   List[str]  (FAISS sırasına göre)
        id2idx:       Dict[str,int]  (chunk_id -> FAISS index)
    """
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index bulunamadı: {index_path}")
    index = faiss.read_index(str(index_path))

    # chunks haritasını bul (jsonl/json)
    candidates = [
        index_path.with_name("index.chunks.jsonl"),
        index_path.with_name("index.chunks.json"),
        index_path.with_name("chunks.jsonl"),
        index_path.with_name("chunks.json"),
    ]
    map_path = None
    for p in candidates:
        if p.exists():
            map_path = p
            break
    if not map_path:
        raise FileNotFoundError("Chunk map yok: index.chunks.jsonl|json / chunks.jsonl|json (make_indices.py ile üret)")

    corpus_texts: List[str] = []
    corpus_ids:   List[str] = []

    # Not: Burada OKUMA SIRASI, FAISS vektör sırasıyla aynı varsayılır.
    if map_path.suffix == ".jsonl":
        for line in map_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id") or obj.get("chunk_id")
            txt = obj.get("text") or obj.get("content") or ""
            # 'passage: ' prefixini metinden temizleyelim (görüntüleme kolaylığı)
            if txt.startswith("passage: "):
                txt = txt[len("passage: "):]
            if cid is None:
                # id yoksa FAISS pozisyonuna göre üret (geriye dönük uyum)
                cid = str(len(corpus_ids))
            corpus_ids.append(str(cid))
            corpus_texts.append(txt)
    else:
        arr = json.loads(map_path.read_text(encoding="utf-8"))
        for obj in arr:
            cid = obj.get("id") or obj.get("chunk_id")
            txt = obj.get("text") or obj.get("content") or ""
            if txt.startswith("passage: "):
                txt = txt[len("passage: "):]
            if cid is None:
                cid = str(len(corpus_ids))
            corpus_ids.append(str(cid))
            corpus_texts.append(txt)

    id2idx = {cid: i for i, cid in enumerate(corpus_ids)}
    return index, corpus_texts, corpus_ids, id2idx


# =============================================================
# 4) SİNYAL ÇIKARIMI (tutar/%/süre/tarih/madde & kural yönü)
# =============================================================
# Para: **para birimi şart** (631 gibi kimlikler tutar sanılmasın)
AMOUNT_RE = re.compile(
    r"(?:₺|TL|TRY)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|"
    r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*(?:₺|TL|TRY)",
    re.IGNORECASE,
)
PERCENT_RE  = re.compile(r"%\s*\d+(?:[.,]\d+)?")
DURATION_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(gün|hafta|ay|yıl|saat|dk|dakika)\b", re.IGNORECASE)
DATE_RE     = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
# Madde/bent (1.1, 1.2.3 gibi) – para/yüzdeye yakın ondalıkları ele
ARTICLE_RE  = re.compile(r"\b\d+(?:\.\d+){1,2}\b")

# Basit kural kelimeleri (pozitif/negatif)
POS_TOKENS = {"izin verilir", "serbest", "mümkündür", "uygulanır", "geçerlidir", "zorunludur"}
NEG_TOKENS = {"yasaktır", "uygulanmaz", "geçersizdir", "yapılamaz", "men edilir", "yasak", "değildir", "muaf"}

def _norm_lower(s: str) -> str:
    return s.lower().replace("ı", "i").replace("İ", "i")

def extract_signals(s: str) -> Dict[str, Any]:
    """
    Satır içinden rakamsal/kural yönü sinyallerini çıkartır.
    -> amounts/percents/durations/dates/article_nums/pos/neg
    """
    text = s
    amounts   = [m.group(0).strip() for m in AMOUNT_RE.finditer(text)]
    percents  = [m.group(0).strip() for m in PERCENT_RE.finditer(text)]
    durations = [(v, u.lower()) for (v, u) in DURATION_RE.findall(text)]
    dates     = DATE_RE.findall(text)

    # Madde/bent numarası — para/% yakınındaki ondalıkları dışarıda tut
    article_nums = []
    for m in ARTICLE_RE.finditer(text):
        a, b = m.span()
        near = text[max(0, a - 4): min(len(text), b + 4)]
        if ("%" in near) or ("TL" in near.upper()) or ("₺" in near.upper()):
            continue
        article_nums.append(m.group(0))

    return {
        "amounts": amounts,
        "percents": percents,
        "durations": durations,
        "dates": dates,
        "article_nums": article_nums,
        "pos": [tok for tok in POS_TOKENS if tok in _norm_lower(text)],
        "neg": [tok for tok in NEG_TOKENS if tok in _norm_lower(text)],
    }

def numeric_set(sig: Dict[str, Any]) -> Dict[str, List[str]]:
    """Karşılaştırma için basit normalize edilmiş listeler döndürür."""
    def norm_num(x: str) -> str:
        return x.replace(" ", "")
    return {
        "amounts":  [norm_num(x) for x in sig["amounts"]],
        "percents": [norm_num(x) for x in sig["percents"]],
        "dur_raw":  ["{} {}".format(v, u.lower()) for (v, u) in sig["durations"]],
        "dates":    sig["dates"],
        # article_nums'ı olduğu gibi bırakıyoruz (string karşılaştırma)
    }

def has_conflict(sigA: Dict[str, Any], sigB: Dict[str, Any]) -> Tuple[bool, str]:
    """
    PDF (A) ve Mevzuat (B) sinyallerini karşılaştırır; bir fark/çelişki notu üretir.
    - tutar/yüzde/süre/tarih/madde-bent/kural yönü
    """
    A = numeric_set(sigA)
    B = numeric_set(sigB)

    # Tutar farkı (para birimi şart)
    if sigA["amounts"] and sigB["amounts"] and set(A["amounts"]) != set(B["amounts"]):
        return True, f"tutar farkı: {A['amounts']} vs {B['amounts']}"

    # Yüzde farkı
    if A["percents"] and B["percents"] and set(A["percents"]) != set(B["percents"]):
        return True, f"yüzde farkı: {A['percents']} vs {B['percents']}"

    # Süre farkı (aynı birimde sayı farklılığı)
    durA_units = [d.split(" ", 1)[1] for d in A["dur_raw"]]
    durB_units = [d.split(" ", 1)[1] for d in B["dur_raw"]]
    common_units = set(durA_units).intersection(durB_units)
    if common_units:
        for unit in common_units:
            valsA = sorted([d.split(" ", 1)[0] for d in A["dur_raw"] if d.endswith(unit)])
            valsB = sorted([d.split(" ", 1)[0] for d in B["dur_raw"] if d.endswith(unit)])
            if valsA and valsB and set(valsA) != set(valsB):
                return True, f"süre farkı ({unit}): {valsA} vs {valsB}"

    # Tarih farkı
    if A["dates"] and B["dates"] and set(A["dates"]) != set(B["dates"]):
        return True, f"tarih farkı: {A['dates']} vs {B['dates']}"

    # Madde/bent farkı (diğerleri yoksa)
    if sigA["article_nums"] and sigB["article_nums"] and set(sigA["article_nums"]) != set(sigB["article_nums"]):
        return True, f"madde/bent farkı: {sigA['article_nums']} vs {sigB['article_nums']}"

    # Kural yönü (pozitif/negatif çelişkisi)
    posA, negA = set(sigA["pos"]), set(sigA["neg"])
    posB, negB = set(sigB["pos"]), set(sigB["neg"])
    if (posA and negB) or (negA and posB):
        return True, "kural yönü farkı (yasak/serbest veya uygulanır/uygulanmaz)"

    return False, ""


# =============================================================
# 5) EŞİK ve ŞİDDET/GÜVEN
# =============================================================
def classify(sim: float, hi: float, lo: float) -> str:
    """
    Benzerliğe göre kaba sınıflama:
      sim >= hi          → candidate_aligned (sonra kural kontrolü yapılır)
      sim <  lo          → missing
      aradaki            → ambiguous
    """
    if sim >= hi:
        return "candidate_aligned"
    if sim < lo:
        return "missing"
    return "ambiguous"

def severity_for(diff_note: str) -> str:
    """Fark notuna göre basit önem derecesi."""
    if any(k in diff_note for k in ["tutar", "yüzde", "süre", "kural", "madde", "bent"]):
        return "kritik"
    return "orta"

def confidence(sim: float, conflict: bool) -> float:
    """
    Basit güven modeli:
    - benzerlik 0.6 ağırlık
    - çelişki bulunmuşsa +0.4, aksi halde +0.2 (minimum güven)
    """
    base = max(0.0, min(1.0, sim))
    return round(0.6 * base + (0.4 if conflict else 0.2), 2)


# =============================================================
# 6) ANA AKIŞ (FAISS + opsiyonel FÜZYON)
# =============================================================
def run(
    pdf_path: Path,
    index_path: Path,
    topk: int,
    hi: float,
    lo: float,
    out_dir: Path,
    fusion_on: bool = False,
    fusion_method: str = "rrf",
    alpha: float = 0.7,
    bm25_k: int = 120,
    faiss_mult: int = 3,
) -> Tuple[Path, Path]:
    """
    - PDF'i chunk'lar, her chunk için embedding hesaplar.
    - FAISS araması yapar; fusion_on=True ise BM25 ile birleştirir.
    - En iyi adayı alıp kurallı fark kontrolü yapar; CSV/JSON yazar.
    """
    # 0) Model ve indeksleri yükle
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    index, corpus_texts, corpus_ids, id2idx = load_faiss_with_map(index_path)

    # Füzyon araması gerekiyorsa BM25 hazırla (yoksa FAISS-only)
    fs = None
    if fusion_on and FUSION_AVAILABLE:
        try:
            fs = FusionSearcher(index_dir=index_path.parent)
        except Exception:
            fs = None  # beklenmedik durumda FAISS-only'a düş

    # 1) PDF -> chunk
    pages  = read_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_chars=900, overlap=120)

    findings: List[Dict[str, Any]] = []

    # 2) Her chunk için arama
    for ch in chunks:
        # 2.1) E5 sorgu embedding'i
        q = f"query: {ch['text']}"
        qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)

        # 2.2) FAISS araması (daha geniş al, füzyon daraltır)
        faiss_k = max(topk, topk * max(1, int(faiss_mult)))
        D, I = index.search(qv, faiss_k)
        sims = D[0].tolist()
        idxs = I[0].tolist()

        # Hit list → (chunk_id, faiss_sim) formuna çevir
        faiss_hits: List[Tuple[str, float]] = []
        for pos, sim in zip(idxs, sims):
            if pos < 0 or pos >= len(corpus_ids):
                continue
            faiss_hits.append((corpus_ids[pos], float(sim)))

        # 2.3) BM25 + Füzyon (opsiyonel)
        fused_hits: List[Tuple[str, float]] = []
        if fs is not None:
            # BM25 tarafını geniş alalım; sonra fuse top_k=topk
            bm25_hits = fs.bm25_search(ch["text"], k=max(bm25_k, topk * 5))
            fused_hits = fs.fuse(
                faiss_hits=faiss_hits,
                bm25_hits=bm25_hits,
                method=fusion_method,
                alpha=alpha,
                top_k=topk,
            )

        # 2.4) Son aday listesi: füzyon varsa fused, yoksa FAISS ilk 'topk'
        final_hits = fused_hits if fused_hits else faiss_hits[:topk]

        # 2.5) En iyi aday (id, skor)
        if final_hits:
            best_id, fused_score = final_hits[0]
            # 'benzerlik' alanı için FAISS benzerliğini tercih ediyoruz (eşiklerle uyumlu)
            faiss_sim_map = dict(faiss_hits)
            best_sim = float(faiss_sim_map.get(best_id, fused_score))
            best_idx = int(id2idx.get(best_id, -1))
            best_match = corpus_texts[best_idx] if 0 <= best_idx < len(corpus_texts) else ""
        else:
            # Hiç aday yoksa
            best_sim = 0.0
            best_idx = -1
            best_match = ""

        # 2.6) Durumu belirle (yalın)
        status0 = classify(best_sim, hi, lo)

        # 2.7) Ön izleme için topk adayları hazırlanır
        def _mk_top_entry(rank: int, cid: str, score: float) -> Dict[str, Any]:
            idx = int(id2idx.get(cid, -1))
            snip = corpus_texts[idx] if 0 <= idx < len(corpus_texts) else ""
            return {
                "rank": rank,
                # Not: Burada 'similarity' alanına FAISS benzerliği varsa onu basıyoruz,
                # yoksa füzyon skorunu yazıyoruz (sadece görüntü amaçlı).
                "similarity": float(dict(faiss_hits).get(cid, score)),
                "index": idx,
                "chunk_id": cid,
                "snippet": (snip[:200] + ("…" if len(snip) > 200 else "")) if snip else "",
            }

        topk_matches = [_mk_top_entry(r+1, cid, sc) for r, (cid, sc) in enumerate(final_hits)]

        # 2.8) Çatışma kontrolü (yalnızca anlamca yakın adaylarda yap)
        diff_note = ""
        status = status0
        sev = "düşük"
        if status0 in {"candidate_aligned", "ambiguous"} and best_match:
            sig_pdf = extract_signals(ch["text"])
            sig_law = extract_signals(best_match)
            is_conflict, note = has_conflict(sig_pdf, sig_law)
            if is_conflict:
                status = "conflict"
                diff_note = note
                sev = severity_for(note)
            else:
                status = "aligned" if status0 == "candidate_aligned" else "weak_aligned"
                sev = "düşük"

        # 2.9) Güven skoru
        conf = confidence(best_sim, status == "conflict")

        # 2.10) Bulguyu kaydet
        findings.append({
            "status": status,                      # aligned / weak_aligned / conflict / missing / ambiguous
            "severity": sev,
            "confidence": conf,
            "similarity": round(float(best_sim), 3),
            "pdf_page": ch["page"],
            "pdf_chunk_id": ch["id"],
            "pdf_idx": ch["idx"],                  # PDF global chunk sırası
            "law_idx": best_idx,                   # FAISS en iyi eşleşme index’i (int)
            "pdf_snippet": (ch["text"][:220] + ("…" if len(ch["text"]) > 220 else "")),
            "law_snippet": (best_match[:220] + ("…" if len(best_match) > 220 else "")) if best_match else "",
            "diff_note": diff_note,
            "topk_matches": topk_matches,          # JSON'da görünür; CSV'ye yazılmıyor
        })

    # 3) Özet istatistik
    def count(st: str) -> int:
        return sum(1 for f in findings if f["status"] == st)

    summary = {
        "aligned": count("aligned"),
        "weak_aligned": count("weak_aligned"),
        "conflict": count("conflict"),
        "missing": count("missing"),
        "ambiguous": count("ambiguous"),
        "total_chunks": len(findings),
        "thresholds": {"hi": hi, "lo": lo, "topk": topk},
        "fusion": {
            "enabled": bool(fs is not None),
            "method": fusion_method if fs is not None else "faiss_only",
            "alpha": alpha,
            "bm25_k": bm25_k,
            "faiss_mult": faiss_mult,
        },
    }

    # 4) Çıktılar
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = out_dir / f"report_{stamp}.json"
    cpath = out_dir / f"report_{stamp}.csv"

    with jpath.open("w", encoding="utf-8") as w:
        json.dump({"summary": summary, "findings": findings}, w, ensure_ascii=False, indent=2)

    # CSV: Analyzer'ın beklediği kolonlar
    cols = [
        "status","severity","confidence","similarity",
        "pdf_page","pdf_chunk_id","pdf_idx","law_idx",
        "pdf_snippet","law_snippet","diff_note"
    ]
    with cpath.open("w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=cols)
        writer.writeheader()
        for f in findings:
            row = {k: f.get(k, "") for k in cols}
            writer.writerow(row)

    # 5) Konsola kısa özet (Rich varsa görsel)
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        cons = Console()
        cons.rule("[bold]Özet")
        cons.print(summary)
        cons.rule("[bold]İlk 10 bulgu")
        t = Table()
        t.add_column("Durum")
        t.add_column("Güven")
        t.add_column("Sim")
        t.add_column("Sayfa")
        t.add_column("PDF Parça")
        t.add_column("Mevzuat Eşleşme")
        for f in findings[:10]:
            t.add_row(
                f["status"], str(f["confidence"]), str(f["similarity"]),
                str(f["pdf_page"]),
                f["pdf_snippet"][:80],
                f["law_snippet"][:80]
            )
        cons.print(t)
        cons.print(Panel.fit(f"[green]JSON[/green]: {jpath}\n[yellow]CSV[/yellow]: {cpath}", title="Raporlar"))
    except Exception:
        print("JSON:", jpath)
        print("CSV :", cpath)

    return jpath, cpath


# =============================================================
# 7) CLI
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True, help="Yeni PDF yolu")
    ap.add_argument("--index", type=Path, required=True, help="Mevzuat FAISS index (index.faiss)")
    ap.add_argument("--topk", type=int, default=3, help="Her chunk için rapora girecek aday sayısı")
    ap.add_argument("--hi", type=float, default=0.80, help="Yüksek benzerlik eşiği (aligned adayı)")
    ap.add_argument("--lo", type=float, default=0.60, help="Düşük benzerlik eşiği (missing)")

    # Füzyon opsiyonları
    ap.add_argument("--fusion", choices=["on","off"], default="off",
                    help="BM25+FAISS füzyon (on/off). Modül yoksa FAISS-only'a düşer.")
    ap.add_argument("--fusion_method", choices=["rrf","wsum"], default="rrf",
                    help="Füzyon yöntemi: rrf (Reciprocal Rank Fusion) | wsum (normalize ağırlıklı toplam)")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="wsum için ağırlık (alpha*FAISS + (1-alpha)*BM25)")
    ap.add_argument("--bm25_k", type=int, default=120,
                    help="BM25 tarafında alınacak aday sayısı (füzyondan önce)")
    ap.add_argument("--faiss_mult", type=int, default=3,
                    help="FAISS tarafında topk * faiss_mult kadar aday al, sonra füzyonla daralt")

    ap.add_argument("--out", type=Path, default=Path("reports"), help="Rapor çıktıları klasörü")
    args = ap.parse_args()

    fusion_on = (args.fusion == "on")

    # PDF kıyasını çalıştır
    _, csv_path = run(
        pdf_path=args.pdf,
        index_path=args.index,
        topk=args.topk,
        hi=args.hi,
        lo=args.lo,
        out_dir=args.out,
        fusion_on=fusion_on,
        fusion_method=args.fusion_method,
        alpha=args.alpha,
        bm25_k=args.bm25_k,
        faiss_mult=args.faiss_mult,
    )

    # Bittiğinde rapor analizörü (MD+XLSX+HTML+DOCX) otomatik tetiklenir
    try:
        import subprocess
        subprocess.run(
            ["python", "report_analyzer.py", "--csv", str(csv_path), "--out", str(args.out)],
            check=True
        )
    except Exception as e:
        print(f"[Auto] report_analyzer çalıştırılamadı: {e}")

if __name__ == "__main__":
    main()
