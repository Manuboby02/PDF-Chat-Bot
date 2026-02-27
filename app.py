import streamlit as st
import tempfile
import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# CONFIG 
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please set GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

st.title("📄 PDF Chatbot with Gemini API")
st.markdown("Upload a PDF and ask questions based on its content.")

# LOAD EMBEDDING MODEL 
if "embed_model" not in st.session_state:
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.text_chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#FILE UPLOAD
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.index is None:

    with st.spinner("Processing PDF and building vector database..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Chunking
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        embeddings = st.session_state.embed_model.encode(chunks)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        st.session_state.index = index
        st.session_state.text_chunks = chunks

    st.success(f"✅ PDF processed successfully! {len(chunks)} chunks created.")

# CHAT SECTION
st.markdown("---")
st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question:")

if query and st.session_state.index is not None:

    with st.spinner("Generating answer..."):

        query_embedding = st.session_state.embed_model.encode([query])
        distances, indices = st.session_state.index.search(
            np.array(query_embedding), 3
        )

        context = ""
        for i in indices[0]:
            context += st.session_state.text_chunks[i] + "\n\n"

        # Add conversational memory
        memory_text = ""
        for q, a in st.session_state.chat_history:
            memory_text += f"User: {q}\nAssistant: {a}\n"

        prompt = f"""
        You are a helpful assistant.

        Previous Conversation:
        {memory_text}

        Context from PDF:
        {context}

        Current Question:
        {query}
        """

        try:
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error generating response: {e}"

    # Display answer
    st.markdown("### 🤖 Answer")
    st.write(answer)

    st.session_state.chat_history.append((query, answer))

# CHAT HISTORY DISPLAY 
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🕘 Conversation History")

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")