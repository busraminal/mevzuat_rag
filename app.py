# app.py
# -*- coding: utf-8 -*-
"""
Mevzuat RAG UI - FastAPI uygulaması
-----------------------------------

Bu servis; PDF yükleyip "gap checker" ile mevzuat karşılaştırması yapar,
çıktı CSV'yi "report_analyzer.py" ile kullanıcı dostu raporlara dönüştürür ve
sonuçları /result sayfasında Jinja2 template ile gösterir.

Öne çıkanlar (bu sürüm):
- /analyze: PDF'i alır, indeksle kıyaslar, rapor dosyalarını üretir ve /result'a yönlendirir
  * (yeni) --law_version parametresini report_analyzer'a iletir (Form alanı veya settings).
  * (yeni) Kısa Rapor'u analyzer üretmezse Excel'den otomatik derler.
- /result : Excel (xlsx) içindeki sheet adlarını otomatik bulur, sayfa-bazlı veri yapısı kurar,
  * (yeni) "Kısa Rapor" linki için short md dosyasını da şablona geçirir.
- /reports: statik rapor dosyaları (Excel/MD/HTML/DOCX/Short-MD) servis edilir.
- /download/latest-short: son Kısa Rapor'u indirir.
- İş akışı uçları: /work/assign, /work/status, /work/note
- Geri bildirim ucu: /feedback (CSV’ye yazar)
- Fix-it taslak: /fixit/draft (E-koduna göre düzeltme metni taslağı üretir)
- Simülasyon: /simulate/recheck (kullanıcı değerleriyle basit yeniden kontrol)
  * (yeni) Backend, TR/EN tip anahtarlarını (para/yüzde/tarih/süre/metin) koruyucu katmanla kabul eder.

Ayrıca:
- E_CODE_DICT + STATUS_TR sözlükleri: UI'da Türkçe etiket ve kod açıklamaları için şablona geçilir
- Kod satırlarının tamamı ayrıntılı yorumlarla açıklandı
"""

from __future__ import annotations

# --- Standart kütüphaneler
import sys, re, ast, json, csv, unicodedata  # unicodedata: TR aksanlarını ASCII'ye indirgemek için
from datetime import datetime
from pathlib import Path
import glob, shutil, subprocess, tempfile
from typing import Optional, List, Dict, Any, Tuple

# --- 3. parti
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- Proje modülleri
from config import settings
from boot import build_engine
from pipeline import QAEngine


# ============================================================================
# 1) APP ve CORS kurulumu
# ============================================================================
app = FastAPI(title="Mevzuat RAG UI", version="1.2")

# Basit CORS: iç ağ/yerel kullanım için her şeyi serbest bırakıyoruz.
# Üretimde domain-kısıtlaması uygulanması önerilir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 2) Yol sabitleri (mutlak)
# ============================================================================
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"        # yüklenen PDF'lerin kopyalandığı klasör
REPORT_DIR: Path = ROOT / "reports"      # bütün rapor çıktı dosyalarının üretildiği klasör

# FAISS indeks yolu (config.settings içinden gelebilir; yoksa varsayılan)
_FAISS_DEFAULT = DATA_DIR / "index.faiss"
FAISS_INDEX: Path = Path(getattr(settings, "FAISS_INDEX", str(_FAISS_DEFAULT))).resolve()

# Basit kalıcı veri dosyaları (iş akışı/geri bildirim için)
WORKLOG_FILE: Path = DATA_DIR / "worklog.json"    # atama/durum/notlar JSON
FEEDBACK_DIR: Path = REPORT_DIR                   # geri bildirim CSV'leri buraya düşecek

# Klasörleri garantiye al (yoksa oluştur)
PDF_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Statik rapor klasörünü mount et (Excel/MD/HTML/DOCX/Short-MD indirme için)
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")

# Jinja2 şablonları (templates/ klasörü)
templates = Jinja2Templates(directory=str(ROOT / "templates"))


# ============================================================================
# 3) UI SÖZLÜKLERİ (TR) – (ŞABLONA GEÇİLECEK)
#    E-kodları sözlüğü ve durumların Türkçe karşılıkları
# ============================================================================
E_CODE_DICT: Dict[str, Dict[str, str]] = {
    "E101": {"ad": "Tutar farkı",
             "aciklama": "Parasal değerler uyuşmuyor. PDF tutarını mevzuata göre güncelleyin; birimi/ayraçları normalize edin."},
    "E102": {"ad": "Yüzde farkı",
             "aciklama": "Yüzde/oran değerleri farklı. PDF oranını mevzuata hizalayın; % ↔ oran yazımını tek biçime getirin."},
    "E103": {"ad": "Süre farkı",
             "aciklama": "Aynı birimde süreler farklı. Süreyi mevzuata göre düzeltin; birimleri standardize edin (gün/ay/yıl)."},
    "E104": {"ad": "Tarih farkı",
             "aciklama": "Tarihler uyuşmuyor (başlangıç/bitiş/yürürlük). İlgili tarih(leri) mevzuat geçerliliğine göre güncelleyin."},
    "E105": {"ad": "Madde/bent atfı farklı",
             "aciklama": "PDF’deki madde/bent ile mevzuat farklı. Doğru maddeye güncelleyin ya da metni o maddeye taşıyın."},
    "E106": {"ad": "Kural yönü çelişkisi",
             "aciklama": "izin/uygulanır ↔ yasak/uygulanmaz çelişkisi var. Mevzuatı esas alarak cümleyi doğru yönle yeniden yazın."},
    "E107": {"ad": "Birim uyuşmazlığı",
             "aciklama": "Ör. m² ↔ ha, gün ↔ ay. Değeri doğru birime çevirerek eşitleyin."},
    "E108": {"ad": "Düşük benzerlik",
             "aciklama": "Anlamsal eşleşme zayıf. Metni bağlamsallaştırın (madde kodu/anahtar kelime) ve yeniden deneyin."},
    "E109": {"ad": "Eksik içerik",
             "aciklama": "Mevzuattaki hüküm PDF’de yok. İlgili hükmü ekleyin veya referans verin."},
    "E110": {"ad": "Belirsiz eşleşme",
             "aciklama": "Adaylar aynı seviyede. İnsan doğrulaması yapın; metni daha belirgin yazın."},
    "E111": {"ad": "Kısmi uyum",
             "aciklama": "Terimler/ifadeler tam örtüşmüyor. Mevzuat terminolojisiyle güçlendirin."},
    "E112": {"ad": "Versiyon/yürürlük sorunu",
             "aciklama": "Yanlış mevzuat versiyonuna atıf. Doğru yürürlük tarihli sürüme hizalayın."},
    "E113": {"ad": "OCR/format hatası",
             "aciklama": "PDF okuma/şablon hatası. Kaynağı düzeltin; makine-okur biçime getirin."},
    "E000": {"ad": "Genel/Belirsiz",
             "aciklama": "Fark türü net değil. İnsan doğrulaması önerilir."},
}

STATUS_TR: Dict[str, str] = {
    "conflict": "Çatışma",
    "missing": "Eksik",
    "low-similarity": "Düşük benzerlik",
    "ambiguous": "Belirsiz",
    "weak_aligned": "Kısmi uyum",
    "aligned": "Uyumlu",
}


# ============================================================================
# 4) Opsiyonel: RAG Engine örneği (Q/A)
# ============================================================================
engine: Optional[QAEngine] = None

@app.on_event("startup")
def _startup() -> None:
    """Uygulama açılırken QA motorunu kurmayı dener.
    Başarısız olursa /ask endpoint'i 503 verir."""
    global engine
    try:
        engine = build_engine()
    except Exception as e:
        engine = None
        print(f"[startup] QA Engine yüklenemedi: {e!r}")


# ============================================================================
# 5) API Şemaları (Pydantic)
# ============================================================================
class Q(BaseModel):
    """Basit Q/A isteği şeması (opsiyonel)."""
    query: str
    k: Optional[int] = None
    top_k: Optional[int] = None

# --- İş akışı / geri bildirim / fix-it / simülasyon için basit şemalar
class AssignIn(BaseModel):
    page: int
    chunk_id: str
    assignee: str

class StatusIn(BaseModel):
    page: int
    chunk_id: str
    status: str  # "open" | "resolved" | "bypass" vb.

class NoteIn(BaseModel):
    page: int
    chunk_id: str
    note: str

class FeedbackIn(BaseModel):
    page: int
    chunk_id: str
    verdict: str        # "up" | "down" | "neutral"
    reason: Optional[str] = None
    code: Optional[str] = None   # E101 vb. (opsiyonel)

class FixitIn(BaseModel):
    page: int
    chunk_id: str
    explain_code: Optional[str] = None
    pdf_snippet: Optional[str] = None
    law_snippet: Optional[str] = None
    diff_note: Optional[str] = None

class SimIn(BaseModel):
    type: Optional[str] = None   # "money" | "percent" | "date" | "duration" | "text"
    pdf_value: Optional[str] = None
    law_value: Optional[str] = None


# ============================================================================
# 6) Sağlık ve Q/A endpoint'leri
# ============================================================================
@app.get("/health")
def health() -> Dict[str, Any]:
    """Temel sağlık kontrolü + indeks yolları"""
    return {
        "status": "ok",
        "faiss": str(FAISS_INDEX),
        "bm25": getattr(settings, "BM25_INDEX", None),
        "engine_ready": engine is not None,
    }

@app.post("/ask")
def ask(q: Q):
    """Opsiyonel: RAG motoru ile soru-cevap.
    (Projede zorunlu değil; hazır yapıyı koruyoruz.)"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    k = q.k or getattr(settings, "TOP_K", 8)
    top_k = q.top_k or getattr(settings, "RERANK_TOP", 8)
    ans = engine.ask(q.query, k=k, top_k=top_k)
    return {"answer": ans}


# ============================================================================
# 7) Basit UI giriş sayfası
# ============================================================================
@app.get("/", response_class=HTMLResponse)
def index_html(request: Request):
    """index.html şablonunu döndürür (yükleme formu vs.)."""
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================================
# 8) Yardımcılar (komut çalıştırma, dosya bulma, diff ayrıştırma, basit KV-store)
# ============================================================================
def _run(cmd: List[str], cwd: Path | None = None) -> None:
    """Komutu çalıştırır, başarısız olursa detaylı 500 döndürür."""
    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "cmd": cmd,
                "code": res.returncode,
                "stdout": res.stdout[-4000:],  # son N karakter
                "stderr": res.stderr[-4000:],
            },
        )

def _latest(pattern: str) -> str:
    """Verilen glob pattern'ine uyan EN SON dosyanın yolunu döndürür."""
    files = glob.glob(pattern)
    if not files:
        raise HTTPException(status_code=404, detail=f"Eşleşen dosya yok: {pattern}")
    return max(files, key=lambda p: Path(p).stat().st_mtime)

def _parse_diff(note: str):
    """'[...] vs [...]' şeklindeki diff_note'tan iki liste çıkartır.

    Örnek:
      diff_note = "madde/bent farkı: ['1.1','2'] vs ['1.2','2']"
      -> (['1.1','2'], ['1.2','2'])
    """
    if not note or "vs" not in note:
        return [], []
    parts = note.split("vs", 1)

    def _extract(s: str):
        m = re.search(r"\[(.*?)\]", s, flags=re.S)
        if not m:
            return []
        raw = "[" + m.group(1) + "]"
        try:
            return ast.literal_eval(raw)
        except Exception:
            # Güvenli ayrıştırma başarısızsa kaba böl/parçala
            return [x.strip().strip("'").strip('"') for x in m.group(1).split(",")]

    return _extract(parts[0]), _extract(parts[1])

def _mismatch_pairs(pdf_vals, law_vals):
    """Sadece iki tarafta da DOLU olan ve BİRBİRİNDEN FARKLI olan değer çiftlerini döndürür.

    Bu liste, UI'da (result.html) <details> içinde "PDF Değeri ↔ Mevzuat Değeri"
    tablosunu doldurmada kullanılıyor.
    """
    pdf_vals = pdf_vals or []
    law_vals = law_vals or []
    n = max(len(pdf_vals), len(law_vals))
    pairs = []
    for i in range(n):
        a = "" if i >= len(pdf_vals) else str(pdf_vals[i]).strip()
        b = "" if i >= len(law_vals) else str(law_vals[i]).strip()
        if not a or not b:
            continue  # tek taraf boşsa atla
        if a != b:
            pairs.append((a, b))
    return pairs

# --- Basit JSON tabanlı mini KV-store (iş akışı verileri için)
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _load_worklog() -> Dict[str, Any]:
    if not WORKLOG_FILE.exists():
        return {"items": {}}
    try:
        return json.loads(WORKLOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"items": {}}

def _save_worklog(data: Dict[str, Any]) -> None:
    WORKLOG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _key(page: int, chunk_id: str) -> str:
    return f"{page}:{chunk_id}"


# ============================================================================
# 8.1) KISA RAPOR OLUŞTURUCU (Analyzer üretmemişse Excel'den derle)
# ============================================================================
def _derive_timestamp_from_name(p: Path) -> str:
    """
    analysis_YYYYMMDD_HHMM.xlsx -> 'YYYYMMDD_HHMM' ts'ini çıkar.
    Adreslenemezse 'now' döner (son çare).
    """
    m = re.search(r"analysis_(\d{8}_\d{4})", p.name)
    return m.group(1) if m else datetime.now().strftime("%Y%m%d_%H%M")

def _build_short_report_from_xlsx(xlsx_file: Path, out_dir: Path) -> Path:
    """
    ProblemDetails sheet'inden yalnızca "başlık + öneri" içeren kısa rapor (Markdown) üretir.
    - Her sayfa için kritik/orta/düşük sayıları + ilk öneri
    - Her bulgu için satır: [E-kodu] Tespit — Öneri
    - Dosya adı: analysis_{ts}_short.md  (ts, xlsx adına gömülü zaman etiketi)
    """
    xl = pd.ExcelFile(xlsx_file)
    sheets = [s.strip() for s in xl.sheet_names]

    # Sheet adlarını esnek eşleştirme
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    wanted_prob = {"problemdetails", "problems", "details"}
    prob_name = next((s for s in sheets if norm(s) in wanted_prob), None)
    if prob_name is None:
        # İlk sayfayı varsay (nadir durum)
        prob_name = sheets[0]

    df = pd.read_excel(xlsx_file, sheet_name=prob_name)

    # Gerekli kolonlar (yoksa tolere ederiz)
    cols = [c for c in [
        "page", "chunk_id",
        "explain_code", "explain_text", "action_hint",
        "severity", "status"
    ] if c in df.columns]
    if not cols:
        raise HTTPException(500, "Kısa rapor üretimi için beklenen kolonlar Excel'de yok.")

    # Sayfa bazlı grupla, özet + önerileri hazırla
    md: List[str] = []
    ts = _derive_timestamp_from_name(xlsx_file)
    md.append(f"# Kısa Rapor — {ts}\n")
    md.append(f"_Kaynak_: **{xlsx_file.name}**\n")

    for page, grp in df[cols].fillna("").groupby("page"):
        # küçük sayaçlar: kritik/orta/düşük
        krit, orta, dusuk = 0, 0, 0
        first_hint = ""
        for _, r in grp.iterrows():
            sev = str(r.get("severity","")).lower()
            if "kritik" in sev: krit += 1
            elif "orta" in sev: orta += 1
            else: dusuk += 1
            if not first_hint and r.get("action_hint"): first_hint = str(r["action_hint"])

        md.append(f"\n## Sayfa {int(page)} — Krit: {krit}  Orta: {orta}  Düşük: {dusuk}")
        if first_hint:
            md.append(f"> **Öneri:** {first_hint}")

        # satır satır: [E-kodu] Tespit — Öneri
        for _, r in grp.iterrows():
            code = (str(r.get("explain_code","")) or "E000").upper()
            text = str(r.get("explain_text","")).strip()
            hint = str(r.get("action_hint","")).strip()
            if not text and not hint:
                continue
            line = f"- **[{code}]** {text}"
            if hint:
                line += f" — _{hint}_"
            md.append(line)

    out = out_dir / f"analysis_{ts}_short.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out


# ============================================================================
# 9) Analyze → Result akışı
# ============================================================================
@app.post("/analyze")
def analyze(
    pdf: UploadFile = File(...),  # kullanıcıdan gelen PDF dosyası
    topk: int = Form(3),          # FAISS/BM25 sorguları için top-k (gap_checker)
    hi: float = Form(0.80),       # yüksek eşik (gap_checker)
    lo: float = Form(0.60),       # düşük eşik (gap_checker)
    law_version: Optional[str] = Form(None),  # (yeni) yürürlük/versiyon bilgisini analyzer'a iletmek için
):
    """
    Akış:
      1) PDF'i 'data/pdfs' altına kaydet
      2) FAISS indeksini geçici klasöre kopyala (dosya kilidi/izin sorunları yaşamamak için)
      3) pdf_gap_checker.py'yi çalıştır (CSV üretir)
      4) report_analyzer.py ile MD/XLSX/HTML (+DOCX) üret
      5) Eğer kısa rapor dosyası oluşmadıysa Excel'den kısa raporu burada oluştur
      6) /result'a yönlendir
    """
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF yükleyin.")

    # Dosyayı güvenli bir ada kaydet
    safe = pdf.filename.replace("/", "_").replace("\\", "_")
    saved = (PDF_DIR / safe).resolve()
    with open(saved, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    # FAISS indeks var mı?
    if not FAISS_INDEX.exists():
        raise HTTPException(400, f"FAISS index yok: {FAISS_INDEX}. Önce make_indices.py ile oluşturun.")

    # Unicode/yol sorunlarına karşı FAISS dosyalarını geçici bir klasöre kopyala
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_idx_"))
    tmp_index = tmp_dir / "index.faiss"
    shutil.copy2(FAISS_INDEX, tmp_index)

    # Eşlik eden "chunk map" dosyaları varsa onları da kopyala
    cand_maps = [
        FAISS_INDEX.with_name("index.chunks.jsonl"),
        FAISS_INDEX.with_name("index.chunks.json"),
        FAISS_INDEX.with_name("chunks.jsonl"),
        FAISS_INDEX.with_name("chunks.json"),
    ]
    for m in cand_maps:
        if m.exists():
            shutil.copy2(m, tmp_dir / m.name)

    # 1) GAP CHECKER
    py = sys.executable
    gap_script = str((ROOT / "pdf_gap_checker.py").resolve())
    _run(
        [py, gap_script, "--pdf", str(saved), "--index", str(tmp_index),
         "--topk", str(topk), "--hi", f"{hi:.2f}", "--lo", f"{lo:.2f}", "--out", str(REPORT_DIR)],
        cwd=ROOT,
    )

    # 2) REPORT ANALYZER
    #    - (yeni) law_version Form alanı doluysa veya settings.LAW_VERSION varsa CLI parametresi geç.
    csv_path = _latest(str(REPORT_DIR / "report_*.csv"))
    ana_script = str((ROOT / "report_analyzer.py").resolve())
    cmd = [py, ana_script, "--csv", csv_path, "--out", str(REPORT_DIR)]
    law_ver = law_version or getattr(settings, "LAW_VERSION", None)
    if law_ver:
        cmd += ["--law_version", str(law_ver)]
    _run(cmd, cwd=ROOT)

    # 3) Son üretilen dosyaları bul
    xlsx_path = _latest(str(REPORT_DIR / "analysis_*.xlsx"))
    md_path   = _latest(str(REPORT_DIR / "analysis_*.md"))
    html_path = _latest(str(REPORT_DIR / "analysis_*.html"))

    # 3.1) (yeni) Kısa rapor: Analyzer üretmemişse Excel'den derle
    short_name = None
    try:
        short_path = _latest(str(REPORT_DIR / "analysis_*_short.md"))
    except Exception:
        # Üretilmediyse hemen üret
        try:
            short_path = _build_short_report_from_xlsx(Path(xlsx_path), REPORT_DIR)
        except Exception as e:
            short_path = None
            print(f"[short] Kısa rapor üretilemedi: {e}")
    if short_path:
        short_name = Path(short_path).name

    # DOCX opsiyonel (docx üretimi başarısız olabilir; zorunlu değil)
    docx_name = None
    try:
        docx_path = _latest(str(REPORT_DIR / "analysis_*.docx"))
        docx_name = Path(docx_path).name
    except Exception:
        pass

    # 4) /result sayfasına yönlendir
    url = (
        f"/result?xlsx={Path(xlsx_path).name}"
        f"&md={Path(md_path).name}"
        f"&html={Path(html_path).name}"
    )
    if docx_name:
        url += f"&docx={docx_name}"
    if short_name:
        url += f"&short={short_name}"
    return RedirectResponse(url=url, status_code=303)


# ============================================================================
# 10) Result – Excel sheet adlarını otomatik bulur ve şablona veri hazırlar
# ============================================================================
@app.get("/result", response_class=HTMLResponse)
def result(
    request: Request,
    xlsx: str,
    md: str,
    html: str | None = None,
    docx: str | None = None,
    short: str | None = None,   # (yeni) kısa rapor dosya adı
):
    """Excel dosyasını okur, 'Overview' ve 'ProblemDetails' sheet'lerini tespit eder;
    sayfa-bazlı (per_page) veri yapısını kurar; result.html şablonuna gönderir."""
    xlsx_file = REPORT_DIR / xlsx
    if not xlsx_file.exists():
        raise HTTPException(404, f"Bulunamadı: {xlsx_file}")

    try:
        # 1) Excel'i aç ve sheet adlarını topla
        xl = pd.ExcelFile(xlsx_file)
        sheets = [s.strip() for s in xl.sheet_names]
        print("[result] xlsx:", xlsx_file, "sheets:", sheets)

        # Sheet adlarını esnek eşleştirme (büyük/küçük/boşluk/özel karakterden bağımsız)
        norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
        wanted_over = {"overview"}
        wanted_prob = {"problemdetails", "problems", "details"}

        over_name = next((s for s in sheets if norm(s) in wanted_over), None)
        prob_name = next((s for s in sheets if norm(s) in wanted_prob), None)

        # Bulunamazsa sırayla 1. ve 2. sheet'i varsay
        if over_name is None:
            over_name = sheets[0]
        if prob_name is None:
            prob_name = sheets[1] if len(sheets) > 1 else sheets[0]

        # 2) DataFrame'leri yükle
        df_over = pd.read_excel(xlsx_file, sheet_name=over_name)
        df_prob = pd.read_excel(xlsx_file, sheet_name=prob_name)

        # 3) Zorunlu kolon kontrolü (Overview)
        need_over = ["page","problems","aligned","total","problem_ratio","avg_sim"]

        # 4) ProblemDetails için zorunlu + opsiyonel kolonlar
        need_prob_base = [
            "page","chunk_id","status","severity","similarity","confidence",
            "diff_note","pdf_snippet","law_snippet"
        ]
        optional_prob: List[str] = []
        for opt in [
            "pdf_idx", "law_idx",
            "explanation",                       # eski açıklama alanı
            "explain_code", "explain_text", "action_hint",  # yeni alanlar
            "madde_pdf", "madde_law"
        ]:
            if opt in df_prob.columns:
                optional_prob.append(opt)
        need_prob = need_prob_base + optional_prob

        # 5) Eksik kolon hataları (sadece zorunlular)
        miss_over = [c for c in need_over if c not in df_over.columns]
        miss_prob = [c for c in need_prob_base if c not in df_prob.columns]
        if miss_over or miss_prob:
            raise HTTPException(
                500,
                detail={
                    "msg": "Excel beklenen kolonları içermiyor.",
                    "missing_overview_cols": miss_over,
                    "missing_problem_cols": miss_prob,
                    "sheets": sheets,
                },
            )

        # 6) Overview -> dict list
        overview = (
            df_over.sort_values(["problems", "problem_ratio", "page"], ascending=[False, False, True])
            .to_dict(orient="records")
        )

        # 7) ProblemDetails -> sayfa-bazlı grupla
        per_page: Dict[int, List[Dict[str, Any]]] = {}
        pages_order = [int(r["page"]) for r in overview]
        prob_groups = {int(p): g for p, g in df_prob.groupby("page")}
        subset_cols = [c for c in need_prob if c in df_prob.columns]

        for p in pages_order:
            g = prob_groups.get(p)
            rows: List[Dict[str, Any]] = []
            if g is None or g.empty:
                # Bu sayfa için hiç problem kaydı yoksa, UI'da yeşil kutu göstermek üzere placeholder satır ekle
                rows.append({
                    "chunk_id": "",
                    "status": "—",   # şablonda "Bu sayfada sorun bulunamadı" mesajını tetikliyor
                    "severity": "",
                    "similarity": "",
                    "confidence": "",
                    "diff_note": "",
                    "pdf_snippet": "— sorun yok —",
                    "law_snippet": "",
                    "pdf_values": [],
                    "law_values": [],
                    "diff_pairs": [],
                    # opsiyonel sütunlar boş geçilebilir
                    **({"pdf_idx": ""} if "pdf_idx" in subset_cols else {}),
                    **({"law_idx": ""} if "law_idx" in subset_cols else {}),
                    **({"explanation": ""} if "explanation" in subset_cols else {}),
                    **({"explain_code": ""} if "explain_code" in subset_cols else {}),
                    **({"explain_text": ""} if "explain_text" in subset_cols else {}),
                    **({"action_hint": ""} if "action_hint" in subset_cols else {}),
                    **({"madde_pdf": ""} if "madde_pdf" in subset_cols else {}),
                    **({"madde_law": ""} if "madde_law" in subset_cols else {}),
                })
            else:
                # İlgili sayfaya ait her problem satırını al, diff_pairs üret ve listeye ekle
                for _, r in g[subset_cols].fillna("").iterrows():
                    row = {k: r[k] for k in subset_cols}
                    pdf_vals, law_vals = _parse_diff(str(row.get("diff_note", "")))
                    row["pdf_values"] = pdf_vals
                    row["law_values"] = law_vals
                    row["diff_pairs"] = _mismatch_pairs(pdf_vals, law_vals)
                    rows.append(row)
            per_page[int(p)] = rows

        # 8) Şablona render et (Türkçe sözlükleri de geçiriyoruz! + kısa rapor linki)
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "overview": overview,
                "per_page": per_page,
                "xlsx_link": f"/reports/{xlsx}",
                "md_link": f"/reports/{md}",
                # Not: html_link'i 'son üretileni göster' endpoint'ine sabit veriyoruz,
                # çünkü HTML'i çoğunlukla "en son" görmek pratik.
                "html_link": f"/report/html" if html else None,
                "docx_link": (f"/reports/{docx}" if docx else None),
                "short_link": (f"/reports/{short}" if short else None),  # (yeni)
                # --- TR sözlükler UI'da kullanılacak ---
                "ecode_dict": E_CODE_DICT,
                "status_tr": STATUS_TR,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        # Geri kalan hataları tek bir 500 bloğunda açıkça göster
        raise HTTPException(status_code=500, detail={"error": str(e), "xlsx": str(xlsx_file)})


# ============================================================================
# 11) İndirme endpoint'leri (son üretileni ver)
# ============================================================================
@app.get("/download/{kind}")
def download(kind: str):
    """En son üretilen ilgili rapor türünü indir (md/xlsx/csv/html/docx/short)."""
    pat = {
        "latest-md":    str(REPORT_DIR / "analysis_*.md"),
        "latest-xlsx":  str(REPORT_DIR / "analysis_*.xlsx"),
        "latest-csv":   str(REPORT_DIR / "report_*.csv"),
        "latest-html":  str(REPORT_DIR / "analysis_*.html"),
        "latest-docx":  str(REPORT_DIR / "analysis_*.docx"),
        "latest-short": str(REPORT_DIR / "analysis_*_short.md"),  # (yeni) Kısa Rapor
    }.get(kind)
    if not pat:
        raise HTTPException(400, "Bilinmeyen tür")
    path = _latest(pat)
    return FileResponse(path, filename=Path(path).name)


# ============================================================================
# 12) HTML rapor görüntüleme (statik üretilen HTML'i aç)
# ============================================================================
@app.get("/report/html", response_class=HTMLResponse)
def report_html():
    """Son üretilen HTML raporunu (report_analyzer.py çıktısı) doğrudan göster."""
    try:
        path = _latest(str(REPORT_DIR / "analysis_*.html"))
    except Exception:
        raise HTTPException(404, "Henüz HTML rapor üretilmedi.")
    return HTMLResponse(Path(path).read_text(encoding="utf-8"))


# ============================================================================
# 13) İŞ AKIŞI UÇLARI (Atama / Durum / Not) — basit JSON saklama
# ============================================================================
@app.post("/work/assign")
def work_assign(req: AssignIn):
    """Bir problem satırına 'sorumlu' atamak için basit KV-store."""
    db = _load_worklog()
    k = _key(req.page, req.chunk_id)
    item = db["items"].get(k, {"page": req.page, "chunk_id": req.chunk_id})
    item["assignee"] = req.assignee
    item["updated_at"] = _now_iso()
    db["items"][k] = item
    _save_worklog(db)
    return {"ok": True, "item": item}

@app.post("/work/status")
def work_status(req: StatusIn):
    """Bir problem satırının çözüm durumunu güncelle (open/resolved/bypass vb.)."""
    db = _load_worklog()
    k = _key(req.page, req.chunk_id)
    item = db["items"].get(k, {"page": req.page, "chunk_id": req.chunk_id})
    item["status"] = req.status
    item["updated_at"] = _now_iso()
    db["items"][k] = item
    _save_worklog(db)
    return {"ok": True, "item": item}

@app.post("/work/note")
def work_note(req: NoteIn):
    """Bir problem satırına serbest metinli not ekle (append)."""
    db = _load_worklog()
    k = _key(req.page, req.chunk_id)
    item = db["items"].get(k, {"page": req.page, "chunk_id": req.chunk_id})
    notes = item.get("notes", [])
    notes.append({"text": req.note, "ts": _now_iso()})
    item["notes"] = notes
    item["updated_at"] = _now_iso()
    db["items"][k] = item
    _save_worklog(db)
    return {"ok": True, "item": item}

@app.get("/work/{page}/{chunk_id}")
def work_get(page: int, chunk_id: str):
    """Tek bir satırın iş akışı kaydını getir."""
    db = _load_worklog()
    k = _key(page, chunk_id)
    return {"item": db["items"].get(k)}


# ============================================================================
# 14) GERİ BİLDİRİM (👍/👎 + neden) — CSV'ye yaz
# ============================================================================
@app.post("/feedback")
def feedback(req: FeedbackIn):
    """
    Kullanıcı geri bildirimini aylık bir CSV'ye yazar:
      reports/feedback_YYYYMM.csv
    Kolonlar: ts,page,chunk_id,verdict,reason,code
    """
    yyyymm = datetime.utcnow().strftime("%Y%m")
    fpath = FEEDBACK_DIR / f"feedback_{yyyymm}.csv"
    exists = fpath.exists()
    with open(fpath, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts","page","chunk_id","verdict","reason","code"])
        w.writerow([_now_iso(), req.page, req.chunk_id, req.verdict, req.reason or "", req.code or ""])
    return {"ok": True, "file": str(fpath.name)}


# ============================================================================
# 15) FIX-IT TASLAK — E-koduna göre örnek düzeltme metni üret
# ============================================================================
_money_pat  = re.compile(r"(\d[\d\.\, ]*)(?:\s*(?:₺|TL|TRY))?", re.I)
_percent_pat= re.compile(r"%\s*([\d]+(?:[.,]\d+)?)")

def _extract_money(s: str) -> Optional[str]:
    if not s: return None
    m = _money_pat.search(s)
    return m.group(1).strip() if m else None

def _extract_percent(s: str) -> Optional[str]:
    if not s: return None
    m = _percent_pat.search(s)
    return m.group(1).replace(",", ".") if m else None

@app.post("/fixit/draft")
def fixit_draft(req: FixitIn):
    """
    Basit kural tabanlı taslak üretimi.
    Not: Bu metinler örnektir; nihai metin hukuk ekibi onayından geçmelidir.
    """
    code = (req.explain_code or "").upper().strip()
    law = (req.law_snippet or "") or ""
    pdf = (req.pdf_snippet or "") or ""

    draft = "Bu satır için genel bir düzeltme taslağı önerilemedi."
    if code == "E101":
        target = _extract_money(law) or _extract_money(req.diff_note or "") or "mevzuattaki tutar"
        draft = f"PDF’deki parasal değeri {target} olacak şekilde güncelleyin; para birimini ve ayraçları (binlik/ondalık) mevzuat formatına uyarlayın."
    elif code == "E102":
        p = _extract_percent(law) or _extract_percent(req.diff_note or "") or "mevzuattaki yüzde"
        draft = f"Metindeki oran(ları) {p}% olacak şekilde düzeltin; yüzde işareti ve ondalık biçimini tekilleştirin."
    elif code == "E103":
        draft = "Süre birimlerini (gün/ay/yıl) standardize edip mevzuattaki değerle uyumlayın. Gerekirse parantez içi açıklama ekleyin."
    elif code == "E104":
        draft = "Tarih(leri) mevzuattaki yürürlük/başlangıç-bitiş hükümlerine göre güncelleyin (gg.aa.yyyy biçimi önerilir)."
    elif code == "E105":
        draft = "Madde/bent referansını mevzuattaki doğru maddeye hizalayın; gerekirse metni ilgili maddeye taşıyın."
    elif code == "E106":
        draft = "Cümleyi mevzuatın yönüne göre yeniden yazın: 'yasak/uygulanmaz' veya 'izin/uygulanır' ifadesini açık seçik belirtin."
    elif code == "E111":
        draft = "Metni mevzuat terminolojisiyle güçlendirin; eş anlamlı ifadeleri mevzuat diline normalleştirin."
    elif code == "E112":
        draft = "Atıf yapılan mevzuat versiyonunu kontrol edip doğru yürürlük tarihli sürüme güncelleyin; raporda versiyon etiketini belirtin."
    elif code == "E113":
        draft = "PDF kaynağındaki OCR/format sorunlarını giderin; tabloları düzenli kolonlarla makine-okur formata çevirin."

    return {"ok": True, "draft": draft, "code": code}


# ============================================================================
# 16) SİMÜLASYON — Kullanıcının girdiği düzeltme ile basit yeniden değerlendirme
#     (TR/EN koruyucu katman ile)
# ============================================================================

def _norm_money(v: str) -> Optional[float]:
    """'10.000,25 TL' -> 10000.25"""
    if not v: return None
    v = v.replace(".", "").replace(" ", "").replace("₺","").replace("TL","").replace("TRY","")
    v = v.replace(",", ".")
    try:
        return float(v)
    except Exception:
        return None

def _norm_percent(v: str) -> Optional[float]:
    """'% 12,5' -> 12.5"""
    if not v: return None
    v = v.replace("%", "").replace(" ", "").replace(",", ".")
    try:
        return float(v)
    except Exception:
        return None

_date_pat = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})")

def _norm_date(v: str) -> Optional[str]:
    """'01.02.2024' -> '2024-02-01' (ISO)"""
    if not v: return None
    m = _date_pat.search(v.strip())
    if not m: return None
    d, mth, y = m.groups()
    y = ("20" + y) if len(y)==2 else y
    try:
        dt = datetime(int(y), int(mth), int(d))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

_dur_pat = re.compile(r"(\d+(?:[.,]\d+)?)\s*(gün|hafta|ay|yıl|saat|dk|dakika)", re.I)
_dur_fact = {"gün":1, "hafta":7, "ay":30, "yıl":365, "saat":1/24, "dk":1/(24*60), "dakika":1/(24*60)}

def _norm_duration(v: str) -> Optional[float]:
    """'3 ay' -> 90.0 (gün) — basit yaklaşık"""
    if not v: return None
    m = _dur_pat.search(v)
    if not m: return None
    num, unit = m.groups()
    num = float(num.replace(",", "."))
    unit = unit.lower()
    unit = "dk" if unit == "dakika" else unit
    fact = _dur_fact.get(unit)
    return num * fact if fact else None


# -------------------- (YENİ) TR/EN koruyucu katman yardımcıları --------------------

def _strip_diacritics(s: str) -> str:
    """Türkçe aksanları (ü,ş,ğ,ı,ö,ç) ASCII'ye indirger: 'yüzde'→'yuzde'."""
    if not isinstance(s, str):
        return ""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )

# Normalleştirilmiş anahtar -> kanonik tip (money/percent/date/duration/text)
TYPE_ALIASES: Dict[str, str] = {
    # EN doğrudan
    "money": "money", "percent": "percent", "date": "date",
    "duration": "duration", "text": "text",

    # TR karşılıklar (aksansızlaştırılmış halleri anahtar)
    "para": "money", "tl": "money", "try": "money", "lira": "money",
    "tutar": "money", "ucret": "money",

    "yuzde": "percent", "oran": "percent",

    "tarih": "date",

    "sure": "duration", "saat": "duration", "gun": "duration",
    "hafta": "duration", "ay": "duration", "yil": "duration",
    "dk": "duration", "dakika": "duration",

    "metin": "text", "yazi": "text",
}

def _canon_kind(user_type: Optional[str]) -> str:
    """
    Kullanıcı tipini (TR/EN, aksanlı/aksanız, kelime/simge) kanonik hale getir.
    Öncelik:
      1) '%' gibi açık ipuçları
      2) Sözlükte tam eşleşme (aksansız, boşluksuz, lower)
      3) İçerik tabanlı heuristik aramalar
      4) Varsayılan: 'text'
    """
    if not user_type:
        return "text"

    raw = str(user_type).strip()
    # 1) Bariz ipucu: yüzde simgesi
    if "%" in raw:
        return "percent"

    # 2) Normalize (aksanları ve boşlukları at)
    key = _strip_diacritics(raw).lower()
    key = re.sub(r"\s+", "", key)  # "yuz de" -> "yuzde"

    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]

    # 3) Heuristik/kısmi eşleşmeler
    if any(tok in key for tok in ("oran", "yuzde", "percent")):
        return "percent"
    if any(tok in key for tok in ("para", "tl", "try", "lira", "tutar", "ucret", "money")):
        return "money"
    if any(tok in key for tok in ("tarih", "date")):
        return "date"
    if any(tok in key for tok in ("sure", "hafta", "gun", "ay", "yil", "saat", "dk", "dakika", "duration")):
        return "duration"
    if any(tok in key for tok in ("metin", "yazi", "text")):
        return "text"

    # 4) Varsayılan
    return "text"


# -------------------- Simülasyon endpoint'i --------------------

@app.post("/simulate/recheck")
def simulate_recheck(req: SimIn):
    """
    Kullanıcının önerdiği düzeltmeyi kaba bir normalizasyonla mevzuat değeriyle karşılaştırır.
    Amaç: 'false positive' korkusunu azaltmak için hızlı bir ön kontrol.

    >>> TR/EN Koruyucu Katman:
        - `req.type` alanı TR/EN, aksanlı/aksanız veya kısmi terimlerle gelebilir.
          Örn: "Para", "yüzde", "oran", "Süre", "hafta", "metin", "%".
        - `_canon_kind()` bu girdiyi kanonik anahtara çevirir:
             money / percent / date / duration / text
    Not: Bu gerçek analiz motorunun yerini tutmaz; UI içi anlık geri bildirim sağlar.
    """
    # --- (YENİ) Kullanıcı tipini önce kanonik hale getiriyoruz
    kind = _canon_kind(req.type)

    # Kullanıcı değerleri (boşsa '' ver)
    pdf_v = req.pdf_value or ""
    law_v = req.law_value or ""

    match = False
    norm_pdf = norm_law = note = None

    try:
        if kind == "money":
            # '10.000,25 TL' -> 10000.25 (float)
            norm_pdf = _norm_money(pdf_v)
            norm_law = _norm_money(law_v)
            match = (norm_pdf is not None and norm_law is not None and abs(norm_pdf - norm_law) < 1e-6)
            note = "Parasal değerler normalize edilerek karşılaştırıldı."
        elif kind == "percent":
            # '% 12,5' -> 12.5 (float)
            norm_pdf = _norm_percent(pdf_v)
            norm_law = _norm_percent(law_v)
            match = (norm_pdf is not None and norm_law is not None and abs(norm_pdf - norm_law) < 1e-6)
            note = "Yüzde değerleri normalize edilerek karşılaştırıldı."
        elif kind == "date":
            # '01.02.2024' -> '2024-02-01' (ISO)
            norm_pdf = _norm_date(pdf_v)
            norm_law = _norm_date(law_v)
            match = (norm_pdf is not None and norm_law is not None and norm_pdf == norm_law)
            note = "Tarihler ISO (YYYY-MM-DD) formatına dönüştürülerek karşılaştırıldı."
        elif kind == "duration":
            # '3 ay' -> 90.0 gün eşdeğeri (float), basit approx.
            norm_pdf = _norm_duration(pdf_v)
            norm_law = _norm_duration(law_v)
            match = (norm_pdf is not None and norm_law is not None and abs(norm_pdf - norm_law) < 1e-6)
            note = "Süreler 'gün' eşdeğerine çevrilerek karşılaştırıldı."
        else:
            # Serbest metin — birebir eşitlik (case-insensitive, trim)
            norm_pdf = pdf_v.strip().lower()
            norm_law = law_v.strip().lower()
            match = (norm_pdf == norm_law)
            note = "Serbest metinde birebir eşitlik kontrolü yapıldı."
    except Exception as e:
        # Giriş kaynaklı beklenmedik hatalarda 400 döndür ve mesajı ilet
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

    return {
        "ok": True,
        "type": kind,            # Kanonik tip (money/percent/date/duration/text)
        "match": bool(match),
        "normalized_pdf": norm_pdf,
        "normalized_law": norm_law,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Geliştirme zamanı notu:
# Uvicorn ile çalıştırmak için:
#   .\.venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
# ---------------------------------------------------------------------------
