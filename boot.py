# boot.py
import os
from typing import Callable, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

from adapters import BM25Adapter, FAISSAdapter, HybridRetriever
from reranker import MiniLMReranker
from answerer import Answerer
from config import settings
from persistence import reload_all, load_bm25  # kalıcı indeksler

load_dotenv()


# ---------- LLM generate() — Hugging Face ----------
def build_llm_generate_fn() -> Callable[[str], str]:
    """
    HF_MODE=local       -> transformers ile yerel inference (küçük instruct model önerilir)
    HF_MODE=inference   -> huggingface_hub Inference API (HUGGINGFACEHUB_API_TOKEN gerekir)
    HF_MODEL            -> model adı (örn: microsoft/Phi-3-mini-4k-instruct)
    HF_MAX_NEW_TOKENS   -> üretilecek maksimum token (örn: 180)
    """
    mode = (os.getenv("HF_MODE") or "local").lower()
    model_name = os.getenv("HF_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    max_new = int(os.getenv("HF_MAX_NEW_TOKENS", "180"))

    def _postprocess(text: str) -> str:
        # Küçük modellerin kalıntı başlıklarını ayıkla
        for sep in ["Kısa yanıt:", "Kısa Yanıt:", "Yanıt:", "Answer:", "Assistant:"]:
            if sep in text:
                return text.split(sep, 1)[-1].strip()
        return text.strip()

    if mode == "inference":
        # ---- Hugging Face Inference API (bulut) ----
        from huggingface_hub import InferenceClient

        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_MODE=inference ama HUGGINGFACEHUB_API_TOKEN yok.")

        client = InferenceClient(model=model_name, token=hf_token)

        def generate(prompt: str) -> str:
            out = client.text_generation(
                prompt=prompt,
                max_new_tokens=max_new,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.15,  # döngü kırıcı
                stream=False,
            )
            return _postprocess(out)

        return generate

    # ---- Yerel Transformers (CPU/GPU) ----
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        mdl.to("cpu")

    # pad token güvenliği
    pad_id = tok.pad_token_id or tok.eos_token_id
    if tok.pad_token_id is None and pad_id is not None:
        tok.pad_token_id = pad_id

    # Güvenli bağlam sınırı
    safe_ctx = min(getattr(tok, "model_max_length", 2048), 2048)

    def generate(prompt: str) -> str:
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,          # uzun promptu kes
            max_length=safe_ctx,      # modele sığdır
        )
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,          # deterministik
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.2,   # tekrarları kır
                no_repeat_ngram_size=4,   # 4-gram tekrarını yasakla
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        return _postprocess(text)

    return generate


# ---------- Engine kurulum ----------
def build_engine():
    # 1) Kalıcı indeksleri yükle
    print("[i] Kalıcı indeksler yükleniyor...")
    chunks, faiss_index, _ = reload_all()  # chunks.jsonl + index.faiss
    print(f"[✓] {len(chunks)} chunk yüklendi (chunks.jsonl).")

    # 2) BM25'i meta'sıyla birlikte yükle
    bm25_obj, bm25_meta = load_bm25(settings.BM25_INDEX)
    print(f"[✓] BM25 meta adedi: {len(bm25_meta)}")

    # 3) Adaptörler
    bm25_adp = BM25Adapter(bm25_obj, bm25_meta)
    embedder = SentenceTransformer(settings.EMB_MODEL, device="cpu")
    embedder.max_seq_length = 512
    faiss_adp = FAISSAdapter(faiss_index, embedder, chunks)

    # 4) Hybrid + Reranker + Answerer
    hybrid = HybridRetriever(bm25_adp, faiss_adp, k_rrf=settings.K_RRF)
    ce = CrossEncoder(settings.CROSS_ENCODER, device="cpu")
    reranker = MiniLMReranker(ce)
    answerer = Answerer(build_llm_generate_fn())

    # Dairesel bağımlılığı önlemek için burada import
    from pipeline import QAEngine
    print("[✓] Engine hazır.")
    return QAEngine(hybrid, reranker, answerer)


if __name__ == "__main__":
    eng = build_engine()
    print("[i] Duman testi: engine oluşturuldu.")
