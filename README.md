Mevzuat RAG

“Mevzuat RAG; PDF ve metin tabanlı mevzuat dokümanlarını chunk’layarak vektör indeksine aktaran, BM25 ve FAISS tabanlı arama motorunu reranker ve LLM desteğiyle güçlendiren bir yapay zekâ çözümüdür. Rapor analizi, zorunluluk kontrolleri ve mevzuat uyumluluğu için güvenilir bir altyapı sunar.”

Mevzuat RAG, mevzuat ve düzenlemelerle ilgili dokümanları işleyip arama–getirme (Retrieval-Augmented Generation, RAG) pipeline’ı üzerinden analiz eden bir yapay zekâ sistemidir.

Amaç: Mevzuat tabanlı raporlama ve belge doğrulama süreçlerinde güvenilir, tekrarlanabilir ve hızlı yapay zekâ destekli bilgi erişimi sağlamak.

📐 Mimarinin Genel Akışı
flowchart TD
    A[PDF / Text Input] --> B[Preprocessing & Cleaning]
    B --> C[Chunking & Metadata Extraction]
    C --> D[Hybrid Indexing: BM25 + FAISS]
    D --> E[Retriever Pipeline]
    E --> F[Reranker (Cross-Encoder)]
    F --> G[LLM Inference]
    G --> H[Answer with Citations / Gap Detection]

🧩 Teknik Bileşenler
🔹 Veri Ön İşleme

Dil Normalizasyonu: unicodedata ile TR aksan/karakter indirgeme

Stopword Filtreleme: Türkçe stopword listesi (NLTK + custom)

Tokenizasyon: HuggingFace AutoTokenizer (E5 tabanlı)

🔹 Chunklama Stratejisi

Sabit uzunluk: 512 token’lık segmentler

Overlap: %20 (yaklaşık 100 token)

Metadata: doc_id, page_no, section_title

🔹 İndeksleme

BM25 (lexical retrieval): rank_bm25

FAISS (vector retrieval):

HNSW index

768-d embedding dimension (E5-base)

Cosine similarity

🔹 Embedding Modelleri

intfloat/multilingual-e5-base
 (default)

Fallback: all-MiniLM-L6-v2 (daha hızlı, düşük boyutlu)

🔹 Reranker

Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2

Top-k = 10 → rerank → final top-3

🔹 LLM Entegrasyonu

Default: Qwen2.5-7B-Instruct (128k context, çok dilli)

Alternatif: Mistral-Nemo-12B-Instruct (Apache-2.0 lisanslı)

Lokal inference: Ollama
 veya GPU destekli deployment

⚙️ Kurulum
# Projeyi klonla
git clone https://github.com/kullanici/mevzuat_rag.git
cd mevzuat_rag

# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Gerekli paketler
pip install -r requirements.txt


requirements.txt içeriği:

faiss-cpu
rank-bm25
transformers
sentence-transformers
torch
pandas
numpy
nltk
PyPDF2

▶️ Kullanım
1. İndeks oluşturma
python build_index.py --data_dir data_pdfs --output_dir index/

2. Sorgu çalıştırma
python query.py --q "Kentsel dönüşüm raporlarında zorunlu alanlar nelerdir?"

3. Örnek çıktı
{
  "query": "Kentsel dönüşüm raporlarında zorunlu alanlar nelerdir?",
  "retrieved_chunks": [
    {
      "doc_id": "mevzuat_spk_2023.pdf",
      "page_no": 15,
      "text": "SPK düzenlemesine göre raporlarda Ada/Parsel, Fiili Kullanım Amacı, Uzman TCKN zorunludur."
    }
  ],
  "answer": "Raporlarda 'Ada/Parsel', 'Fiili Kullanım Amacı', 'Uzman Bilgileri (TCKN)' alanları mevzuat gereği zorunludur. Eksiklik halinde rapor geçersiz sayılır."
}

📊 Benchmark & Performans

Ortalama retrieval latency: 120ms (FAISS + BM25 hybrid)

Ortalama LLM response latency: ~2.3s (7B model, A100 GPU)

Türkçe mevzuat corpus’unda Top-3 accuracy: %84

Fallback MiniLM ile hız artışı: %+40 (doğrulukta %–8 düşüş)

🌍 Kullanım Senaryoları

Gayrimenkul değerleme raporları → SPK düzenlemeleri ile uyum kontrolü

Şirket içi uyumluluk denetimleri → zorunlu alan boşluklarının tespiti

Akademik araştırmalar → mevzuat–doküman eşleşmeleri

Avukatlık & danışmanlık → mevzuat atıflarının otomatik çıkarımı

🤝 Katkı

Issues → bug/feature request

Pull Requests → yeni embedding modelleri, chunklama stratejileri, mevzuat corpus güncellemeleri

📜 Lisans

Bu proje MIT License
 kapsamında dağıtılmaktadır.
