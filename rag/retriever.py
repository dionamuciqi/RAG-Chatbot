from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rag.config import settings

def get_db():
    """
    1) Krijon OpenAIEmbeddings -> për ta kthyer pyetjen në vektor
    2) Hap Chroma DB nga data/chroma
    3) Lidhet me collection: genpact_rag
    """
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
    """
    Merr pyetjen -> Chroma gjen top-k chunks më të ngjashme.
    Kthen listë dokumentesh (chunks) me metadata: source/page.
    """
    if k is None:
        k = settings.TOP_K

    db = get_db()
    docs = db.similarity_search(query, k=k)
    return docs
