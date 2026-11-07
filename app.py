import os, pickle
import streamlit as st
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Research Paper AI Chatbot", page_icon="📚", layout="centered")
st.title("📚 Research Paper AI Chatbot")
st.write("Upload a research paper PDF, embed it locally, and chat with it intelligently!")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.sidebar.header("1. Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Research PDF", type=["pdf"])

os.makedirs("./vector_store_data", exist_ok=True)
embeddings_path = "./vector_store_data/embeddings.npy"
chunks_path = "./vector_store_data/chunks.pkl"

if uploaded_file is not None:
    pdf_path = os.path.join("./vector_store_data", uploaded_file.name)
    with open(pdf_path, "wb") as out:
        out.write(uploaded_file.getbuffer())
    if st.sidebar.button("Process & Embed PDF"):
        with st.spinner("Extracting text and generating local embeddings..."):
            doc = pymupdf.open(pdf_path)
            full_text = "".join([f"\n--- Page {i + 1} ---\n" + p.get_text("text") for i, p in enumerate(doc)])
            chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
            embeddings = model.encode(chunks, show_progress_bar=False)
            np.save(embeddings_path, embeddings)
            with open(chunks_path, "wb") as cp:
                pickle.dump(chunks, cp)
            st.sidebar.success(f"Successfully embedded {len(chunks)} chunks!")

st.header("🔍 Chat with Your Paper")
if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
    embeddings = np.load(embeddings_path)
    with open(chunks_path, "rb") as cp:
        chunks = pickle.load(cp)
        
    query = st.text_input("Ask a question about your research paper:")
    if query:
        with st.spinner("Searching context..."):
            q_emb = model.encode(query)
            sims = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
            best = np.argmax(sims)
            
            st.markdown(f"**Relevance Confidence:** `{sims[best]:.4f}`")
            
            context_excerpt = chunks[best]
            
            # Cleanly presented result section
            st.subheader("💡 Answer Summary:")
            st.write(context_excerpt)
            
            with st.expander("📄 View Raw Source Paragraph"):
                st.code(context_excerpt, language="text")
else:
    st.warning("⚠️ Please upload and process a PDF using the sidebar first to start chatting!")
