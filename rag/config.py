import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # Paths
    RAW_DIR: str = "data/raw"
    CHROMA_DIR: str = "data/chroma"   
    COLLECTION_NAME: str = "genpact_rag"

    # Chunking
    CHUNK_SIZE: int = 1100
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K: int = 6

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"

settings = Settings()
