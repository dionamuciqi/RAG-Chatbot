import streamlit as st
from rag.qa import answer_question

# ---------- Page config ----------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    #MainMenu { visibility: hidden; }

    /* App background */
    .stApp {
        background: radial-gradient(circle at 15% 20%, rgba(56,189,248,0.18), transparent 35%),
                    radial-gradient(circle at 85% 10%, rgba(99,102,241,0.14), transparent 40%),
                    linear-gradient(180deg, #071521 0%, #06111a 50%, #050d14 100%);
        color: #e7eef7;
    }

    /* Layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 7rem;   
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,40,58,0.92), rgba(10,22,33,0.92));
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #e7eef7;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        border: 1px solid rgba(255,255,255,0.12);
        background: linear-gradient(90deg, rgba(34,211,238,0.85), rgba(59,130,246,0.85));
        color: #03101a;
        font-weight: 750;
        box-shadow: 0 10px 25px rgba(56,189,248,0.22);
    }

    /* Chat message container */
    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }

    /* Bubbles (same width, centered) */
    .bubble-user, .bubble-assistant {
        width: 100%;
        max-width: 880px;
        margin: 10px auto;
        border-radius: 16px;
        padding: 12px 14px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 22px rgba(0,0,0,0.25);
        line-height: 1.45;
        font-size: 16.5px;
        color: #EAF2FF;
    }
    .bubble-user {
        background: linear-gradient(135deg, rgba(34,211,238,0.22), rgba(59,130,246,0.16));
    }
    .bubble-assistant {
        background: rgba(255,255,255,0.085);
    }

    /* Citations */
    .cites {
        max-width: 880px;
        margin: 10px auto 0 auto;
        padding: 12px;
        border-radius: 14px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 13px;
        opacity: 0.95;
    }
    .cites code {
        background: rgba(0,0,0,0.25);
        padding: 2px 6px;
        border-radius: 8px;
    }

    /* Bright text */
    .stMarkdown, .stText, p, li { color: #EAF2FF !important; }

    /* Image */
    div[data-testid="stImage"] img {
        border-radius: 18px;
        padding: 10px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }

  
    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Floating chat input */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 22px;
        left: 0;
        right: 0;
        z-index: 999;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] > div {
        max-width: 980px;
        margin: 0 auto;
        padding: 0 18px;
        background: transparent !important;
        position: relative !important;
    }

    /* Input style */
    div[data-testid="stChatInput"] textarea {
        border-radius: 999px !important;
        background: #FFFFFF !important;
        border: 1px solid rgba(15,23,42,0.30) !important;
        color: #0B1F33 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        padding: 0.85rem 3.2rem 0.85rem 1.15rem !important; /* space for send icon */
        min-height: 54px !important;
        box-shadow: 0 14px 34px rgba(0,0,0,0.28);
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: rgba(11,31,51,0.55) !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        border: 1px solid rgba(56,189,248,0.95) !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.20), 0 14px 34px rgba(0,0,0,0.28) !important;
    }

    /* Center send icon */
    div[data-testid="stChatInput"] button {
        position: absolute !important;
        right: 26px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        height: 44px !important;
        width: 44px !important;
        border-radius: 999px !important;
        background: transparent !important;
        border: none !important;
    }

    div[data-testid="stChatInput"] button:hover {
        background: rgba(15,23,42,0.06) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.caption("Desktop UI • LangChain + Chroma + OpenAI")
    st.markdown("---")

    st.markdown("**Navigation**")
    st.markdown("- Chat")
    st.markdown("- History (soon)")
    st.markdown("- Dashboard (soon)")
    st.markdown("---")

    st.markdown("**Quick settings**")
    show_citations = st.toggle("Show citations", value=True)
    top_k = st.slider("Top-K", min_value=2, max_value=10, value=6, step=1)
    st.markdown("---")

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main ----------
c1, c2 = st.columns([0.18, 0.82], vertical_alignment="center")
with c1:
    st.image("app/assets/finance_banner.png", width=160)

with c2:
    st.markdown(
        """
        <div style="padding-top:2px;">
            <h1 style="margin:0; color:#EAF2FF; font-size:34px;">Finance Chatbot</h1>
            <h3 style="margin:6px 0 0 0; color:#38BDF8; font-weight:650; font-size:18px;">Development</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------- Render chat ----------
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

# ---------- Chat input ----------
question = st.chat_input("Ask anything")

if question:
    q = question.strip()
    st.session_state.messages.append({"role": "user", "content": q})

    with st.spinner("Retrieving relevant passages + generating answer..."):
        answer, citations = answer_question(q)

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
    st.rerun()
