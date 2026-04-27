import streamlit as st
import os
import time
from chatbot import initialize_llm, get_embeddings, create_vector_db, load_vector_db, setup_qa_chain, build_memory
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindEase · Mental Health Chatbot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #f0f4f0 0%, #e8f0ed 50%, #f0ece8 100%);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2e22 0%, #0f1e17 100%) !important;
    border-right: 2px solid #2d4a35 !important;
    padding-top: 1rem !important;
}
[data-testid="stSidebar"] * { color: #c8d8c8 !important; }
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a8d8a8 !important;
    font-family: 'DM Serif Display', serif !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #2d5a35, #2d4a35) !important;
    color: #e8f5e8 !important;
    border: 1.5px solid #4a7a52 !important;
    border-radius: 10px !important;
    width: 100% !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #3d7a45, #3d6a3d) !important;
    border-color: #6a9a72 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(45,90,53,0.3) !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #2d4a35 !important;
    color: #e0ece0 !important;
    border: 1px solid #3d5a45 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #c8d8c8 !important;
}

/* ── Main title area ── */
.title-block {
    text-align: center;
    padding: 2rem 0 1rem;
}
.title-block h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #1a3a22;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.title-block p {
    color: #5a7a5a;
    font-size: 1rem;
    font-weight: 300;
}

/* ── Chat messages ── */
.chat-container {
    max-width: 780px;
    margin: 0 auto;
    padding: 1rem 0 6rem;
    background: linear-gradient(to bottom, rgba(240,244,240,0.5), rgba(232,240,237,0.3));
    border-radius: 12px;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 1.2rem 0;
    animation: fadeUp 0.3s ease;
    padding: 0 1rem;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 1.2rem 0;
    animation: fadeUp 0.3s ease;
    padding: 0 1rem;
}

.bubble-user {
    background: linear-gradient(135deg, #2d5a35, #1a4a22);
    color: #f0faf0;
    padding: 0.95rem 1.3rem;
    border-radius: 20px 20px 4px 20px;
    max-width: 72%;
    font-size: 0.95rem;
    line-height: 1.7;
    box-shadow: 0 2px 12px rgba(45,90,53,0.3);
    font-weight: 500;
}
.bubble-bot {
    background: #f8fdf8;
    color: #1a3a22;
    padding: 0.95rem 1.3rem;
    border-radius: 20px 20px 20px 4px;
    max-width: 72%;
    font-size: 0.95rem;
    line-height: 1.7;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    border: 1.5px solid #d4e8d4;
    font-weight: 500;
}

.avatar-bot {
    width: 34px; height: 34px;
    background: #c8e6c8;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    margin-right: 0.6rem;
    flex-shrink: 0;
    align-self: flex-end;
}

/* ── Input area ── */
.stChatInput {
    position: fixed;
    bottom: 0;
    background: rgba(240,244,240,0.95);
    backdrop-filter: blur(10px);
    padding: 1rem 2rem;
    border-top: 1px solid #d0ddd0;
    z-index: 100;
}
.stChatInput textarea {
    border-radius: 12px !important;
    border: 1.5px solid #c0d4c0 !important;
    background: white !important;
    color: #1a3a22 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stChatInput textarea:focus {
    border-color: #3d7a45 !important;
    box-shadow: 0 0 0 3px rgba(61,122,69,0.12) !important;
}

/* ── Status pills ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 1rem;
}
.status-ready { background: #d4edda; color: #1a5c2a; }
.status-loading { background: #fff3cd; color: #7a5c00; }
.status-error { background: #f8d7da; color: #721c24; }

/* ── Typing indicator ── */
.typing { display: flex; gap: 5px; padding: 4px 0; }
.typing span {
    width: 8px; height: 8px;
    background: #a0bea0;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-8px); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #7a9a7a;
}
.empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #3a6a3a;
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.9rem; max-width: 360px; margin: 0 auto; }

/* Starter prompts */
.starter-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    max-width: 500px;
    margin: 1.5rem auto 0;
}
.starter-btn {
    background: white;
    border: 1.5px solid #a8d8a8 !important;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: #1a3a22;
    cursor: pointer;
    text-align: left;
    transition: all 0.3s;
    font-weight: 500;
}
.starter-btn:hover {
    background: #e8f8e8;
    border-color: #2d5a35 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(45,90,53,0.2);
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "db_status" not in st.session_state:
    st.session_state.db_status = "not_loaded"
if "starter_clicked" not in st.session_state:
    st.session_state.starter_clicked = None


# ── Initialize knowledge base on startup ──────────────────────────────────────
if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
    api_key = os.getenv("GROQ_API_KEY", "")
    csv_path = "Mental_Health_FAQ.csv"
    
    if api_key and os.path.exists(csv_path):
        try:
            llm = initialize_llm(api_key)
            db_path = "./chroma_db"
            if os.path.exists(db_path):
                vector_db = load_vector_db(db_path)
            else:
                vector_db = create_vector_db(csv_path, db_path)
            memory = build_memory()
            st.session_state.memory = memory
            st.session_state.qa_chain = setup_qa_chain(vector_db, llm, memory)
            st.session_state.db_status = "ready"
        except Exception as e:
            st.session_state.db_status = "error"
            print(f"Error initializing knowledge base: {e}")
    else:
        st.session_state.db_status = "not_loaded"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 MindEase")
    st.markdown("---")

    if st.session_state.messages:
        st.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")
        mem = st.session_state.memory
        if mem:
            turns = len(mem) // 2
            st.markdown(f"**🧠 Memory:** {turns} / 10 turns")
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            # Reset memory so the new conversation starts fresh
            st.session_state.memory = []
            st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
This chatbot uses **RAG** (Retrieval-Augmented Generation):
- Searches a curated mental health FAQ
- Answers with context using LLaMA 3.3 70B via Groq
- **Not a substitute for professional help**
    """)
    st.markdown("---")
    st.caption("🔒 Your conversations are not stored.")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='title-block'>
    <h1>🌿 MindEase</h1>
    <p>A compassionate space to talk about mental health</p>
</div>
""", unsafe_allow_html=True)

# Status bar
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    status = st.session_state.db_status
    if status == "ready":
        st.markdown("<div class='status-pill status-ready'>● Ready</div>", unsafe_allow_html=True)
    elif status == "loading":
        st.markdown("<div class='status-pill status-loading'>● Setting up...</div>", unsafe_allow_html=True)
    elif status == "not_loaded":
        st.markdown("<div class='status-pill status-error'>● Not loaded — configure sidebar</div>", unsafe_allow_html=True)

# Starter prompts as buttons (shown when chat is empty)
STARTERS = [
    "I've been feeling really anxious lately",
    "How can I manage stress at work?",
    "I can't stop overthinking everything",
    "Tips for better sleep and mental health?",
]

if not st.session_state.messages:
    st.markdown("""
    <div class='empty-state'>
        <div class='icon'>🌱</div>
        <h3>Hi, I'm here to listen</h3>
        <p>Share what's on your mind, or try one of these to get started</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, starter in enumerate(STARTERS):
        with cols[i % 2]:
            if st.button(starter, key=f"starter_{i}", use_container_width=True):
                st.session_state.starter_clicked = starter
                st.rerun()

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='msg-user'>
            <div class='bubble-user'>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='msg-bot'>
            <div class='avatar-bot'>🌿</div>
            <div class='bubble-bot'>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Handle starter click ───────────────────────────────────────────────────────
def handle_query(user_input: str):
    from langchain_core.messages import HumanMessage, AIMessage
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Add to memory
    st.session_state.memory.append(HumanMessage(content=user_input))

    if st.session_state.qa_chain is None:
        reply = "⚠️ The knowledge base isn't loaded yet. Please enter your API key and click **Load / Build Knowledge Base** in the sidebar."
    else:
        try:
            with st.spinner(""):
                # Call the chain function with question and chat history
                reply = st.session_state.qa_chain(user_input, st.session_state.memory)
        except Exception as e:
            reply = f"Something went wrong: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Add to memory
    st.session_state.memory.append(AIMessage(content=reply))


if st.session_state.starter_clicked:
    handle_query(st.session_state.starter_clicked)
    st.session_state.starter_clicked = None
    st.rerun()

# Chat input
if user_input := st.chat_input("Share what's on your mind…"):
    handle_query(user_input)
    st.rerun()
