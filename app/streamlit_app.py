# app/streamlit_app.py
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from rag.qa import answer_question
from rag.retriever import get_db  

# ---------------- Page config ----------------
st.set_page_config(page_title="Finance Chatbot", layout="wide")
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()



# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def get_available_metadata():
    sources: set[str] = set()
    pages: set[int] = set()

    try:
        db = get_db()
        col = db._collection  
        data = col.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []

        for md in metadatas:
            if not md:
                continue

            src = md.get("source")
            if src:
                sources.add(str(src))

            pg = md.get("page")
            if pg is not None and pg != "?":
                try:
                    pages.add(int(pg))
                except Exception:
                    pass
    except Exception:
        pass

    sources_list = sorted(sources)
    pages_list = sorted(pages)
    pmin = min(pages_list) if pages_list else None
    pmax = max(pages_list) if pages_list else None
    return sources_list, pmin, pmax


def build_filters(selected_sources, page_from, page_to):
    clauses = []

    if selected_sources:
        if len(selected_sources) == 1:
            clauses.append({"source": str(selected_sources[0])})
        else:
            clauses.append({"source": {"$in": [str(s) for s in selected_sources]}})

    if page_from is not None:
        clauses.append({"page": {"$gte": int(page_from)}})
    if page_to is not None:
        clauses.append({"page": {"$lte": int(page_to)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("app/assets/finance_banner.png", width=100)
    st.markdown(
        "<h2 style='margin:8px 0 0 0; color:#EAF2FF;'>Finance Chatbot</h2>",
        unsafe_allow_html=True,
    )

    st.markdown("**Citations**")
    show_citations = st.toggle("Show citations", value=True)

    st.markdown("**Metadata filters**")
    sources_list, pmin, pmax = get_available_metadata()

    selected_sources = st.multiselect(
        "Source (PDF filename)",
        options=sources_list,
        default=[],
    )

    page_from = None
    page_to = None
    if pmin is not None and pmax is not None:
        col_pf, col_pt = st.columns(2)
        with col_pf:
            page_from = st.number_input(
                "Page from",
                min_value=int(pmin),
                max_value=int(pmax),
                value=int(pmin),
                step=1,
            )
        with col_pt:
            page_to = st.number_input(
                "Page to",
                min_value=int(pmin),
                max_value=int(pmax),
                value=int(pmax),
                step=1,
            )

    use_filters = st.toggle("Enable filters", value=False)

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Header ----------------
c1, c2 = st.columns([0.18, 0.82], vertical_alignment="center")
with c1:
    st.image("app/assets/finance_banner.png", width=160)
with c2:
    st.markdown(
        "<h1 style='margin:0; color:#EAF2FF; font-size:34px;'>Finance Chatbot</h1>",
        unsafe_allow_html=True,
    )

st.write("")  

# ---------------- Chat history ----------------
for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"<div class='bubble-user'>{m['content']}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(f"<div class='bubble-assistant'>{m['content']}</div>", unsafe_allow_html=True)

            if show_citations and m.get("citations"):
                seen = set()
                lines = []
                for c in m["citations"]:
                    key = (c.get("source"), c.get("page"))
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"- <code>{c.get('source')}</code> p.{c.get('page')}")
                cites_html = "<br/>".join(lines) if lines else "No citations."
                st.markdown(f"<div class='cites'><b>Sources</b><br/>{cites_html}</div>", unsafe_allow_html=True)

# ---------------- Input ----------------
question = st.chat_input("Ask anything")

if question:
    q = question.strip()
    st.session_state.messages.append({"role": "user", "content": q})

    # Short-term memory 
    history_msgs = st.session_state.messages[:-1]
    history_lines = []
    for msg in history_msgs[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")
    chat_history = "\n".join(history_lines)

    # Apply metadata filters 
    filters = build_filters(selected_sources, page_from, page_to) if use_filters else None

    with st.spinner("Retrieving relevant passages + generating answer..."):
        answer, citations = answer_question(q, chat_history=chat_history, filters=filters)

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
    st.rerun()