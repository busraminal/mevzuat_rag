# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# .env dosyasını OS ortamına yükle (pydantic-settings bunu da okuyabilir)
load_dotenv()

class Settings(BaseSettings):
    # pydantic-settings v2.x için önerilen config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATA_DIR: str = "./data"
    FAISS_INDEX: str = "./data/index.faiss"
    BM25_INDEX: str = "./data/bm25_index.pkl"
    META_PATH: str = "./data/meta.jsonl"

    EMB_MODEL: str = "intfloat/multilingual-e5-large"
    CROSS_ENCODER: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    TOP_K: int = 20
    RERANK_TOP: int = 8
    K_RRF: int = 60

    HOST: str = "0.0.0.0"
    PORT: int = 8000

# 🔴 ÖNEMLİ: app.py bu ismi import ediyor
settings = Settings()
