import streamlit as st
import os
import tempfile
import pdfplumber
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
# Set page configuration as the first Streamlit command
st.set_page_config(page_title="📄🔍 RAG Chatbot with Groq", layout="centered")

# Load Groq API key from Streamlit secrets
groq_api_key = st.secrets["groq"]["api_key"]
#groq_model = "llama3-8b-8192"
groq_model="llama-3.3-70b-versatile"
# LangChain LLM with Groq
llm = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model)

# Add debugging statement before loading the model
st.write("Loading SentenceTransformer model...")
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
stored_chunks = []
stored_metadata = []

st.title("🤖📚 PDF Chatbot using Groq + FAISS")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- PDF Chunking ----------------------
def load_and_chunk_pdfs(pdf_paths, chunk_size=500, chunk_overlap=50):
    documents = []

    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    metadata = {"page": i + 1, "source": os.path.basename(pdf_path)}
                    documents.append(Document(page_content=text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

# ------------------- Embedding & FAISS ----------------------
def embed_and_store(chunks):
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.encode(texts)
    index.add(np.array(embeddings))
    stored_chunks.extend(texts)
    stored_metadata.extend([doc.metadata for doc in chunks])

# ------------------- Retrieval ----------------------
def query_faiss(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [stored_chunks[i] for i in I[0]]

# ------------------- Generate Response from Groq ----------------------
def generate_answer(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = (
        "You are a helpful assistant. Answer the user's question based on the context below.\n\n"
        f"Context:\n{context}\n\nUser: {query}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

# ------------------- Streamlit Upload UI ----------------------
uploaded_files = st.file_uploader("📁 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            file_paths.append(tmp.name)

    chunks = load_and_chunk_pdfs(file_paths)
    embed_and_store(chunks)
    st.success(f"✅ {len(uploaded_files)} PDF(s) processed and embedded.")

# ------------------- Chat Interface ----------------------
with st.form("chat_form"):
    user_query = st.text_input("💬 Ask a question about the PDFs...")
    submitted = st.form_submit_button("Ask")

    if submitted and user_query:
        if len(stored_chunks) == 0:
            st.warning("Please upload and embed some PDFs first.")
        else:
            top_chunks = query_faiss(user_query)
            answer = generate_answer(user_query, top_chunks)

            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("GroqBot", answer))

# ------------------- Display Conversation ----------------------
st.divider()
for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}**: {msg}")
