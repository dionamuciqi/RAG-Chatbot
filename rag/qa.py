from langchain_openai import ChatOpenAI

from rag.config import settings
from rag.retriever import retrieve
from rag.prompt import SYSTEM_PROMPT, USER_TEMPLATE


def build_context(docs):
   
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] SOURCE: {src} | PAGE: {page}\n{d.page_content}")
    return "\n\n".join(parts)


def _clean_answer(text: str) -> str:
 
    if not text:
        return text

    markers = ["\nCitations:", "\nSources:", "\nReferences:"]
    for m in markers:
        if m in text:
            text = text.split(m)[0].strip()
    return text.strip()


def answer_question(question: str):
    # Retrieve top-k chunks
    docs = retrieve(question, k=settings.TOP_K)

    if not docs:
        return "I don't know based on the provided documents.", []

    #  Build context text
    context = build_context(docs)

    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2
    )

    user_prompt = USER_TEMPLATE.format(question=question, context=context)

    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("user", user_prompt),
    ])

    answer_text = _clean_answer(response.content)

    citations = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "source": src,
            "page": page,
            "snippet": d.page_content[:220].replace("\n", " ")
        })

    return answer_text, citations
