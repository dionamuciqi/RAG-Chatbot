from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rag.config import settings

def get_db():
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )

    db = Chroma(
        collection_name=settings.COLLECTION_NAME,
        persist_directory=settings.CHROMA_DIR,
        embedding_function=embeddings
    )
    return db

def retrieve(query: str, k: int = None):
    if k is None:
        k = settings.TOP_K

    db = get_db()
    docs = db.similarity_search(query, k=k)
    return docs
