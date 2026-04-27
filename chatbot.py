"""
chatbot.py — Backend logic for MindEase mental health chatbot
Uses modern LangChain retrieval chain with memory.
"""

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Final answer prompt with message history
QA_PROMPT_TEMPLATE = """You are MindEase, a compassionate mental health chatbot.
Use the context below to answer thoughtfully and empathetically.

Guidelines:
- Respond with warmth and genuine understanding
- Be non-judgmental; never dismiss or minimise feelings
- Reference earlier parts of the conversation naturally when relevant
- If someone seems in crisis, gently encourage professional help
- If the context doesn't cover the question, say so kindly

Context from knowledge base:
{context}

{chat_history}

Respond naturally to the user's question based on the context above."""


def initialize_llm(api_key: str) -> ChatGroq:
    return ChatGroq(
        temperature=0.2,
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
    )


def get_embeddings() -> HuggingFaceBgeEmbeddings:
    return HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)


def create_vector_db(csv_path: str, db_path: str) -> Chroma:
    loader = CSVLoader(csv_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    print(f"[chatbot] Created ChromaDB with {len(texts)} chunks at {db_path}")
    return vector_db


def load_vector_db(db_path: str) -> Chroma:
    print(f"[chatbot] Loading existing ChromaDB from {db_path}")
    return Chroma(persist_directory=db_path, embedding_function=get_embeddings())


def build_memory():
    """Initialize chat memory as an empty list."""
    return []


def setup_qa_chain(vector_db: Chroma, llm: ChatGroq, memory: list):
    """
    Build a modern retrieval chain with chat history support.
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Format chat history for the prompt
    def format_chat_history(chat_history):
        """Convert chat history list to formatted string."""
        if not chat_history:
            return ""
        
        formatted = "Chat history:\n"
        for msg in chat_history[-10:]:  # Keep last 10 messages (5 exchanges)
            if isinstance(msg, HumanMessage):
                formatted += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                formatted += f"MindEase: {msg.content}\n"
        return formatted
    
    # Create the QA prompt
    qa_prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history"],
    )
    
    # Build the chain
    def chain_func(question: str, chat_history: list = None):
        """Execute the retrieval chain."""
        if chat_history is None:
            chat_history = []
        
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format chat history
        formatted_history = format_chat_history(chat_history)
        
        # Create the prompt
        prompt_input = {
            "context": context,
            "chat_history": formatted_history,
            "question": question
        }
        
        # Generate response
        prompt_text = qa_prompt.format(**prompt_input)
        response = llm.invoke(prompt_text)
        
        # Extract text from response
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return answer
    
    return chain_func
