# answerer.py
from __future__ import annotations
from typing import List, Dict, Any, Callable


class Answerer:
    """
    LLM'e prompt kurup cevabı döndüren yardımcı sınıf.
    generate_fn -> build_llm_generate_fn() tarafından sağlanan fonksiyon
    """
    def __init__(self, generate_fn: Callable[[str], str]):
        self.generate_fn = generate_fn

    # answerer.py  (Answerer._build_prompt ve answer)
    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        context_lines = []
        for c in contexts:
            file = c.get("file", "?")
            page = c.get("page", "?")
            snippet = (c.get("text", "") or "").strip().replace("\n", " ")
            if len(snippet) > 300:                # <-- PARÇA KISALT
                snippet = snippet[:297] + "..."
            context_lines.append(f"[{file} | sayfa {page}] {snippet}")

        joined = "\n".join(context_lines)

        prompt = f"""Aşağıda belgeden alınmış parçalar var. SADECE bu parçalara dayanarak cevap ver.Yanıtı tekrar etmeyecek şekilde yaz. Aynı kelimeyi üst üste kullanma.
Kaynakta yoksa "Belgede bulunamadı" de. Tahmin etme. Kısa ve net cevap yaz.

Bağlam:
{joined}

Soru: {query}

Cevap (Türkçe):
"""
        return prompt

    def answer(self, query: str, contexts: List[Dict[str, Any]], max_context: int = 5) -> str:
        use_ctx = contexts[:max_context]          # <-- 5’ten fazlasını alma
        prompt = self._build_prompt(query, use_ctx)
        return self.generate_fn(prompt)

