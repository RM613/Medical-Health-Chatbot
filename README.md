# 🌿 MindEase — Mental Health Chatbot

A RAG-powered mental health chatbot built with LangChain, Groq (LLaMA 3.3 70B), ChromaDB, and Streamlit.

## Setup

```bash
# 1. Clone / copy the project files
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your Groq API key (free at console.groq.com)

# 4. Run the app
streamlit run app.py
```

## First run

1. Open the app in your browser (default: http://localhost:8501)
2. In the **sidebar**, enter your Groq API key and the path to `Mental_Health_FAQ.csv`
3. Click **Load / Build Knowledge Base** — this embeds the CSV into ChromaDB (one-time)
4. Start chatting!

## Project structure

```
mental_health_chatbot/
├── app.py            # Streamlit UI
├── chatbot.py        # Backend: LLM, embeddings, RAG chain
├── requirements.txt
├── .env.example
└── Mental_Health_FAQ.csv   # your dataset (not included)
```

## Stack

| Component | Tool |
|-----------|------|
| LLM | LLaMA 3.3 70B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (local) |
| RAG framework | LangChain |
| UI | Streamlit |

## ⚠️ Disclaimer

This chatbot is for informational and supportive purposes only.
It is **not** a substitute for professional mental health care.
If you're in crisis, please contact a mental health professional or call a crisis helpline.
