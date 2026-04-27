"""
MindEase — Theme 1: Dark Royal Blue & Gold
Aesthetic: Regal, luxurious, authoritative — like a private counselor's office
"""

import streamlit as st
import os
from chatbot import initialize_llm, create_vector_db, load_vector_db, setup_qa_chain, build_memory
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="MindEase · Mental Health Chatbot",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400;1,600&family=Lato:wght@300;400;700&display=swap');

:root {
    --royal:       #0d1b4b;
    --royal-mid:   #1a2f7a;
    --royal-light: #263d9e;
    --gold:        #c9a84c;
    --gold-light:  #e8c96a;
    --gold-pale:   #f5e6bb;
    --ivory:       #faf8f2;
    --ivory-dark:  #f0ead6;
    --text-dark:   #0a1230;
    --text-mid:    #2a3d70;
    --shadow-gold: 0 4px 20px rgba(201,168,76,0.18);
    --shadow-royal:0 4px 24px rgba(13,27,75,0.25);
}

html, body, [class*="css"] { font-family: 'Lato', sans-serif; }

.stApp {
    background: var(--ivory);
    background-image:
        radial-gradient(ellipse 70% 50% at 0% 0%, rgba(26,47,122,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 100% 100%, rgba(201,168,76,0.09) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1535 0%, #0d1b4b 40%, #091228 100%) !important;
    border-right: 1px solid rgba(201,168,76,0.3) !important;
}
[data-testid="stSidebar"] * { color: #cdd5e8 !important; }
[data-testid="stSidebar"] h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important;
    font-style: italic !important;
    color: var(--gold-light) !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(201,168,76,0.7) !important;
    margin: 1.2rem 0 0.5rem !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(201,168,76,0.12) !important;
    color: var(--gold-pale) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    transition: all 0.25s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(201,168,76,0.25) !important;
    border-color: var(--gold) !important;
    color: var(--gold-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(201,168,76,0.2) !important;
}
[data-testid="stSidebar"] .stExpander {
    background: rgba(201,168,76,0.06) !important;
    border: 1px solid rgba(201,168,76,0.18) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stMetric {
    background: rgba(201,168,76,0.1) !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 8px !important;
    padding: 0.7rem 0.9rem !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--gold-light) !important;
    font-size: 1.5rem !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(201,168,76,0.2) !important; }

/* ── Main header ── */
.main-header {
    text-align: center;
    padding: 2.2rem 0 1.2rem;
}
.header-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: var(--royal);
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
}
.header-title span { color: var(--gold); font-style: italic; }
.header-divider {
    width: 80px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 0.9rem auto;
}
.header-sub {
    font-size: 0.88rem;
    font-weight: 300;
    color: var(--text-mid);
    letter-spacing: 0.06em;
}

/* ── Status badge ── */
.status-row { display: flex; justify-content: center; margin-bottom: 1.2rem; }
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-ready { background: #e6f0ff; color: #0d3080; border: 1px solid #b0c4e8; }
.badge-loading { background: #fff8e6; color: #7a5c00; border: 1px solid #e8d470; }
.badge-error { background: #fdecea; color: #8b2020; border: 1px solid #e8b0b0; }
.badge-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.badge-ready .badge-dot { animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Empty state ── */
.empty-wrap { text-align: center; padding: 2.5rem 1rem 1rem; }
.empty-icon {
    width: 76px; height: 76px;
    background: linear-gradient(135deg, var(--ivory-dark), var(--gold-pale));
    border: 2px solid rgba(201,168,76,0.35);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.9rem;
    margin: 0 auto 1rem;
    box-shadow: var(--shadow-gold);
}
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--royal);
    margin: 0 0 0.4rem;
}
.empty-sub { font-size: 0.86rem; font-weight: 300; color: var(--text-mid); max-width: 320px; margin: 0 auto; line-height: 1.65; }
.starters-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    text-align: center;
    margin: 2rem 0 0.8rem;
}
.divider-ornament { text-align: center; color: rgba(201,168,76,0.5); letter-spacing: 0.6em; font-size: 0.8rem; margin: 0.8rem 0 1.2rem; }

/* ── Starter buttons ── */
button[kind="secondary"] {
    background: white !important;
    border: 1.5px solid var(--ivory-dark) !important;
    border-radius: 12px !important;
    color: var(--text-dark) !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.83rem !important;
    padding: 0.78rem 1rem !important;
    text-align: left !important;
    box-shadow: var(--shadow-royal) !important;
    transition: all 0.25s !important;
}
button[kind="secondary"]:hover {
    background: var(--gold-pale) !important;
    border-color: var(--gold) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-gold) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    animation: fadeUp 0.3s ease both;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

/* USER MESSAGE - Force white text */
[data-testid="stChatMessage"][aria-label="user"] {
    color: #ffffff !important;
}

[data-testid="stChatMessage"][aria-label="user"] [data-testid="stMarkdownContainer"] {
    background: linear-gradient(135deg, var(--royal), var(--royal-mid)) !important;
    color: #ffffff !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 0.9rem 1.2rem !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    max-width: 74% !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    box-shadow: var(--shadow-royal) !important;
}

[data-testid="stChatMessage"][aria-label="user"] [data-testid="stMarkdownContainer"] * {
    color: #ffffff !important;
}

[data-testid="stChatMessage"][aria-label="user"] p {
    color: #ffffff !important;
}

[data-testid="stChatMessage"][aria-label="user"] span {
    color: #ffffff !important;
}

[data-testid="stChatMessage"][aria-label="user"] div {
    color: #ffffff !important;
}

/* ASSISTANT MESSAGE - Force dark text */
[data-testid="stChatMessage"][aria-label="assistant"] {
    color: #0a1230 !important;
}

[data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stMarkdownContainer"] {
    background: white !important;
    color: #0a1230 !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 0.9rem 1.2rem !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    max-width: 74% !important;
    border: 1.5px solid var(--ivory-dark) !important;
    border-left: 3px solid var(--gold) !important;
    box-shadow: var(--shadow-royal) !important;
}

[data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stMarkdownContainer"] * {
    color: #0a1230 !important;
}

[data-testid="stChatMessage"][aria-label="assistant"] p {
    color: #0a1230 !important;
}

[data-testid="stChatMessage"][aria-label="assistant"] span {
    color: #0a1230 !important;
}

[data-testid="stChatMessage"][aria-label="assistant"] div {
    color: #0a1230 !important;
}

/* Override any inherited colors */
[data-testid="stChatMessage"] * { color: inherit !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(250,248,242,0.97) !important;
    backdrop-filter: blur(20px) !important;
    border-top: 1px solid rgba(201,168,76,0.25) !important;
    padding: 1rem 2rem 1.2rem !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    border: 1.5px solid var(--ivory-dark) !important;
    background: white !important;
    color: var(--text-dark) !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--royal-light) !important;
    box-shadow: 0 0 0 3px rgba(26,47,122,0.1) !important;
}
[data-testid="stChatInput"] button { background: var(--royal) !important; border-radius: 10px !important; }
[data-testid="stChatInput"] button:hover { background: var(--royal-mid) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
defaults = {"messages": [], "qa_chain": None, "memory": [], "db_status": "not_loaded", "starter_clicked": None}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Auto-init ──
if st.session_state.qa_chain is None and st.session_state.db_status == "not_loaded":
    api_key = os.getenv("GROQ_API_KEY", "")
    csv_path = "Mental_Health_FAQ.csv"
    if api_key and os.path.exists(csv_path):
        st.session_state.db_status = "loading"
        try:
            llm = initialize_llm(api_key)
            db_path = "./chroma_db"
            vector_db = load_vector_db(db_path) if os.path.exists(db_path) else create_vector_db(csv_path, db_path)
            memory = build_memory()
            st.session_state.memory = memory
            st.session_state.qa_chain = setup_qa_chain(vector_db, llm, memory)
            st.session_state.db_status = "ready"
        except Exception as e:
            st.session_state.db_status = "error"
            st.session_state._init_error = str(e)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🔵 MindEase")
    st.caption("Your confidential companion")
    st.divider()
    if st.session_state.messages:
        st.markdown("### Session")
        c1, c2 = st.columns(2)
        with c1: st.metric("Messages", len(st.session_state.messages))
        with c2: st.metric("Memory", f"{len([m for m in st.session_state.memory if isinstance(m, HumanMessage)])}/10")
        st.markdown("### Actions")
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []; st.session_state.memory = []; st.rerun()
        st.divider()
    with st.expander("💡 Tips for Better Conversations"):
        st.markdown("- Be specific about what you're feeling\n- Mention context: work, sleep, events\n- Ask follow-ups if needed")
    with st.expander("🆘 Crisis Resources"):
        st.markdown("🇺🇸 **USA** — 988 | Text HOME → 741741\n\n🇬🇧 **UK** — 116 123\n\n🇮🇳 **India** — AASRA: 9820466626")
    with st.expander("ℹ️ About MindEase"):
        st.markdown("Powered by **LLaMA 3.3 70B** via Groq + RAG over a curated mental health knowledge base.\n\n⚠️ Not a substitute for professional care.")
    st.divider()
    s = st.session_state.db_status
    if s == "ready": st.success("✅ Knowledge base active")
    elif s == "loading": st.warning("⏳ Loading…")
    elif s == "error": st.error(f"❌ {getattr(st.session_state,'_init_error','error')}")
    else: st.info("⚙️ Set GROQ_API_KEY in .env")
    st.caption("🔒 Conversations stay on your device.")

# ── Main ──
# Sidebar toggle styling
st.markdown("""
<style>
/* Make sidebar toggle button visible and styled */
button[aria-label="Close sidebar"] {
    position: fixed;
    top: 1rem;
    left: 1rem;
    z-index: 999;
    background: linear-gradient(135deg, #263d9e, #0d1b4b) !important;
    border: 1.5px solid #c9a84c !important;
    color: #f5e6bb !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(13,27,75,0.25) !important;
}

button[aria-label="Close sidebar"]:hover {
    background: linear-gradient(135deg, #263d9e, #1a2f7a) !important;
    box-shadow: 0 6px 20px rgba(201,168,76,0.25) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

_, center, _ = st.columns([1, 5, 1])
with center:
    st.markdown("""
    <div class="main-header">
        <div class="header-eyebrow">Mental Health Support</div>
        <h1 class="header-title">Mind<span>Ease</span></h1>
        <div class="header-divider"></div>
        <p class="header-sub">A confidential space to find clarity and calm</p>
    </div>
    """, unsafe_allow_html=True)

    s = st.session_state.db_status
    bc = {"ready":"badge-ready","loading":"badge-loading","error":"badge-error"}.get(s,"badge-error")
    bt = {"ready":"Ready","loading":"Setting up…","error":"Unavailable","not_loaded":"Not configured"}.get(s,s)
    st.markdown(f'<div class="status-row"><span class="badge {bc}"><span class="badge-dot"></span>{bt}</span></div>', unsafe_allow_html=True)

    STARTERS = ["💭  I've been feeling really anxious lately","🏢  How can I manage stress at work?","🌀  I can't stop overthinking everything","🌙  Tips for better sleep and mental health?"]
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-icon">🕊️</div>
            <h3 class="empty-title">I'm here to listen</h3>
            <p class="empty-sub">Whatever's weighing on you, share it. This is your space.</p>
        </div>
        <div class="starters-label">Begin with a topic</div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, s in enumerate(STARTERS):
            with (c1 if i%2==0 else c2):
                if st.button(s, key=f"s_{i}", use_container_width=True):
                    st.session_state.starter_clicked = s.split("  ",1)[-1]; st.rerun()
        st.markdown('<div class="divider-ornament">— ✦ —</div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧠" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

def handle_query(user_input):
    user_input = user_input.strip()
    if not user_input: return
    st.session_state.messages.append({"role":"user","content":user_input})
    st.session_state.memory.append(HumanMessage(content=user_input))
    if st.session_state.qa_chain is None:
        reply = "⚠️ Knowledge base not loaded. Ensure GROQ_API_KEY is set in .env and restart."
    else:
        try:
            with st.spinner("Reflecting…"):
                reply = st.session_state.qa_chain(user_input, st.session_state.memory)
        except Exception as e:
            reply = f"Something went wrong — please try again. *(Error: {e})*"
    st.session_state.messages.append({"role":"assistant","content":reply})
    st.session_state.memory.append(AIMessage(content=reply))

if st.session_state.starter_clicked:
    handle_query(st.session_state.starter_clicked); st.session_state.starter_clicked = None; st.rerun()
if user_input := st.chat_input("What's on your mind…"):
    handle_query(user_input); st.rerun()