# -*- coding: utf-8 -*-
"""
nlp_helpers.py
- TR odaklı normalizasyon ve eşanlam genişletme
- Amaç: BM25 tarafında eşanlam destekli metinle arama, FAISS tarafında da
  sorgu normalize ederek "off-topic" hataları azaltmak.

Kullanım:
    from nlp_helpers import normalize_text, expand_query_terms

    q_norm = normalize_text(query)
    terms = expand_query_terms(q_norm)  # {'ruhsat','izin','onay', ...}

Not: Bu sözlük başlangıç setidir; domain'e göre genişletin.
"""

import re
import unicodedata
from typing import List, Set

# Basit TR lower + unicode normalizasyonu
def _simple_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Eşanlam kümeleri (kanonik → varyantlar)
SYN_SETS = [
    {"ruhsat", "izin", "onay", "lisans", "yetki"},
    {"bedel", "ücret", "tutar", "miktar", "meblağ"},
    {"fesih", "iptal", "sona erme", "sonlandırma"},
    {"başvuru", "müracaat", "talep"},
    {"yüklenici", "taşeron", "alt yüklenici"},
    {"vergi", "kdv", "katma değer vergisi"},
    {"süre", "termin", "vade"},
    {"sözleşme", "kontrat", "anlaşma"},
    {"ceza", "yaptırım", "müeyyide"},
]

CANON_TO_ALL = {}
TERM_TO_CANON = {}
for s in SYN_SETS:
    canon = sorted(s, key=len)[0]  # en kısa terimi kanonik seç
    CANON_TO_ALL[canon] = set(s)
    for t in s:
        TERM_TO_CANON[_simple_norm(t)] = canon

# Yazım/biçim normalize (örnekler)
REPLACERS = [
    (r"\b% ?(\d+)", r"yüzde \1"),  # "%20" → "yüzde 20"
    (r"₺", " tl "),                 # simge → kelime
    (r"\btry\b", " tl "),           # TRY → tl
    (r"\bmad\.?\b", " madde "),     # "mad." → "madde"
]

def normalize_text(s: str) -> str:
    s = _simple_norm(s)
    for pat, rep in REPLACERS:
        s = re.sub(pat, rep, s)
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zçğıöşü0-9\.]+", s, flags=re.I)

def expand_query_terms(s: str) -> Set[str]:
    """
    Sorgudaki her terim için: (1) kanonik, (2) eşanlamları set'e ekler.
    Çıkış, BM25 sorgu token’ları olarak kullanılabilir.
    """
    s = normalize_text(s)
    terms = set()
    for tok in tokenize(s):
        t = _simple_norm(tok)
        if t in TERM_TO_CANON:
            canon = TERM_TO_CANON[t]
            terms |= CANON_TO_ALL.get(canon, {t})
        else:
            terms.add(t)
    return terms
