# smoke.py
# Basit duman testi: QAEngine'i ayağa kaldır, soru sor, yanıt ve kaynakları yazdır.
from __future__ import annotations
import sys, json
from pathlib import Path

def load_jsonl(p: Path):
    items = []
    if not p.exists():
        return items
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items

def print_contexts(ctxs):
    if not ctxs:
        print("  (bağlam yok)")
        return
    for i, c in enumerate(ctxs, 1):
        f = c.get("file", "?")
        p = c.get("page", "?")
        s = c.get("ce_score", c.get("score"))
        snippet = (c.get("text") or "").replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f"  {i}. {f} | sayfa {p} | skor={s}")
        print(f"     {snippet}")

def interactive(eng):
    print("[i] Interaktif duman testi. Çıkmak için: q / quit / exit")
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
        print_contexts(out.get("contexts", []))

def batch(eng):
    path = Path("tests/smoke_qa.jsonl")
    items = load_jsonl(path)
    if not items:
        print(f"[!] '{path}' bulunamadı veya boş. Interaktif moda geçebilirsiniz.")
        return
    ok = 0
    for i, it in enumerate(items, 1):
        q = it.get("q") or it.get("question") or ""
        if not q:
            continue
        print(f"\n[{i}/{len(items)}] Soru: {q}")
        out = eng.ask(q)
        print("Yanıt:", out["answer"])
        print("Kaynaklar:")
        print_contexts(out.get("contexts", []))
        ok += 1
    print(f"\n[✓] Tamamlanan test sayısı: {ok}")

def main():
    try:
        from boot import build_engine
        print("[i] Engine yükleniyor...")
        eng = build_engine()
        print("[✓] Engine hazır.")
    except Exception as e:
        print("[x] Engine başlatılamadı:", e)
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1].lower() in {"--batch", "-b"}:
        batch(eng)
    else:
        interactive(eng)

if __name__ == "__main__":
    main()
