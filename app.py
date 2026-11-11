import os, pickle
import streamlit as st
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Research Paper AI Chatbot", page_icon="📚", layout="centered")
st.title("📚 Research Paper AI Chatbot")
st.write("Upload a research paper PDF, embed it locally, and search it precisely!")

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
        with st.spinner("Extracting text by paragraphs..."):
            doc = pymupdf.open(pdf_path)
            chunks = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
                for p in paragraphs:
                    chunks.append(f"[Page {page_num + 1}]\n{p}")
            
            if not chunks:
                chunks = ["Document text could not be segmented."]
                
            embeddings = model.encode(chunks, show_progress_bar=False)
            np.save(embeddings_path, embeddings)
            with open(chunks_path, "wb") as cp:
                pickle.dump(chunks, cp)
            st.sidebar.success(f"Successfully embedded {len(chunks)} distinct sections!")

st.header("🔍 Search Your Paper")
if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
    embeddings = np.load(embeddings_path)
    with open(chunks_path, "rb") as cp:
        chunks = pickle.load(cp)
        
    query = st.text_input("Ask a question about your research paper:")
    if query:
        with st.spinner("Searching top matches..."):
            q_emb = model.encode(query)
            
            # Safe cosine similarity calculation
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            q_emb_norm = q_emb / np.linalg.norm(q_emb)
            sims = np.dot(embeddings_norm, q_emb_norm)
            
            # Get top 3 matching indices
            top_indices = np.argsort(sims)[::-1][:3]
            
            st.subheader("💡 Top 3 Matching Excerpts:")
            for i, idx in enumerate(top_indices):
                st.markdown(f"**Match {i+1} (Confidence: `{sims[idx]:.4f}`)**")
                st.info(chunks[idx])
else:
    st.warning("⚠️ Please upload and process a PDF using the sidebar first!")
