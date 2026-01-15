from langchain_openai import ChatOpenAI

from rag.config import settings
from rag.retriever import retrieve
from rag.prompt import SYSTEM_PROMPT, USER_TEMPLATE


def build_context(docs):
    """
    Kthen chunks në një tekst të vetëm.
    Ne i shtojmë SOURCE dhe PAGE që modeli të ketë bazë për citations.
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] SOURCE: {src} | PAGE: {page}\n{d.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str):
    """
    1) Retrieval nga Chroma (Top-K)
    2) Ndërton context
    3) I dërgon OpenAI chat modelit pyetjen + context + rregullat (prompt.py)
    4) Kthen answer + citations (si listë për UI)
    """
    # 1) Retrieve top-k chunks
    docs = retrieve(question, k=settings.TOP_K)

    # Nëse s’ka asgjë, kthe menjëherë safe response
    if not docs:
        return "I don't know based on the provided documents.", []

    # 2) Build context text
    context = build_context(docs)

    # 3) Krijo LLM (OpenAI chat model)
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2
    )

    # 4) Ndërto prompt-in final për modelin
    user_prompt = USER_TEMPLATE.format(question=question, context=context)

    # 5) Thirr modelin
    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("user", user_prompt),
    ])

    # 6) Për UI: nxjerr citations nga docs (më “reliable” se sa të lexosh tekstin e LLM)
    citations = []
    for d in docs:
        citations.append({
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "?"),
            "snippet": d.page_content[:220].replace("\n", " ")
        })

    return response.content, citations
