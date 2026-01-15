import os
import uuid
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from rag.config import settings


def load_documents() -> Tuple[List, List[Tuple[str, str]]]:
    """
    Loads PDFs one-by-one so a single bad/encrypted PDF doesn't crash the pipeline.
    Returns:
      docs: list of LangChain Documents (per page)
      failed: list of (filename, error_message)
    """
    docs = []
    failed = []

    if not os.path.exists(settings.RAW_DIR):
        raise FileNotFoundError(f"RAW_DIR does not exist: {settings.RAW_DIR}")

    pdf_files = [f for f in os.listdir(settings.RAW_DIR) if f.lower().endswith(".pdf")]
    print(f"STEP A: Found {len(pdf_files)} PDF files in {settings.RAW_DIR}")

    for fname in pdf_files:
        path = os.path.join(settings.RAW_DIR, fname)
        try:
            loader = PyPDFLoader(path)
            file_docs = loader.load()  # one Document per page

            # Normalize metadata for citations
            for d in file_docs:
                src = d.metadata.get("source", "")  
                d.metadata["source"] = os.path.basename(src) or fname
                d.metadata["page"] = d.metadata.get("page", None)  
                d.metadata["file_path"] = path

            docs.extend(file_docs)

        except Exception as e:
            failed.append((fname, str(e)))

    ok_files = len({d.metadata.get("source") for d in docs})
    print(f"STEP B: Loaded OK files: {ok_files}")
    if failed:
        print(f"STEP B2: Failed PDFs: {len(failed)} (showing up to 10)")
        for f, err in failed[:10]:
            print(f"  - {f}: {err[:160]}")

    return docs, failed


def chunk_documents(docs: List) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for c in chunks:
        c.metadata["chunk_id"] = str(uuid.uuid4())

    return chunks


def build_index() -> Tuple[int, int]:
    print("STEP 0: build_index started")

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing. Put it in your .env file")

    print("STEP 1: API key found ")
    print("RAW_DIR      =", settings.RAW_DIR)
    print("CHROMA_DIR   =", settings.CHROMA_DIR)
    print("COLLECTION   =", settings.COLLECTION_NAME)
    print("CHUNK_SIZE   =", settings.CHUNK_SIZE)
    print("CHUNK_OVERLP =", settings.CHUNK_OVERLAP)
    print("EMBED_MODEL  =", settings.EMBEDDING_MODEL)

    docs, failed = load_documents()
    print(f"STEP 2: Loaded pages (documents) = {len(docs)}")

    if len(docs) == 0:
        print(" No documents loaded. Check that PDFs exist in data/raw.")
        return 0, 0

    chunks = chunk_documents(docs)
    print(f"STEP 3: Created chunks = {len(chunks)}")

    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    print("STEP 4: Embeddings initialized ")

    # Ensure persist dir exists
    os.makedirs(settings.CHROMA_DIR, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.COLLECTION_NAME,
        persist_directory=settings.CHROMA_DIR,
    )
    vectordb.persist()
    print("STEP 5: Chroma persisted ")

    return len(docs), len(chunks)


if __name__ == "__main__":
    pages, chunks = build_index()
    print(f"\nRESULT: Loaded pages = {pages}")
    print(f"RESULT: Created chunks = {chunks}")
    print(f"Done. Chroma is in: {settings.CHROMA_DIR}\n")
