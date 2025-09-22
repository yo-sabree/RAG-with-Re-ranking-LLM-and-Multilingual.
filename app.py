import sqlite3
import sys
from dotenv import load_dotenv
load_dotenv()
import os
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import fitz
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import google.genai as genai
from google.genai import types
from datetime import datetime
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD", "15"))
st.set_page_config(page_title="RAG Advanced", layout="wide")
st.title("RAG (Upload, Ask) — Advanced Retrieval")
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload up to 15 PDFs", type=["pdf"], accept_multiple_files=True)
with col2:
    build_index_btn = st.button("Build Index")
    clear_collection_btn = st.button("Clear Collection")
def extract_text_from_pdf_bytes(pdf_bytes, filename):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        pages.append({"text": text, "page": i+1, "source": filename})
    return pages

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = max(0, end - overlap)
        if end >= len(text):
            break
    return chunks

@st.cache_resource
def get_embedding_model(name):
    return SentenceTransformer(name)

@st.cache_resource
def get_reranker(name):
    return CrossEncoder(name)

@st.cache_resource
def get_chroma_client(path):
    return PersistentClient(path)

@st.cache_data
def build_tfidf(corpus):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    mat = vect.fit_transform(corpus)
    mat = normalize(mat, axis=1)
    return vect, mat

def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def call_gemini_stream(prompt, api_key, model):
    client = genai.Client(api_key=api_key)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024)
    result = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        text = getattr(chunk, "text", None)
        if text:
            result += text
        else:
            cand = getattr(chunk, "candidates", None)
            if cand and len(cand) > 0:
                c = cand[0]
                parts = getattr(c, "content", None)
                if parts and isinstance(parts, list) and len(parts) > 0:
                    part = parts[0]
                    ptext = getattr(part, "text", None)
                    if ptext:
                        result += ptext
    return result.strip()

client = get_chroma_client(CHROMA_DIR)

if clear_collection_btn:
    try:
        col = client.get_or_create_collection("rag_collection")
        all_ids = col.get(include=[])["ids"]
        if all_ids:
            col.delete(ids=all_ids)
        st.success("Cleared Chroma collection")
    except Exception as e:
        st.error(f"Failed to clear collection: {e}")

if build_index_btn and uploaded:
    if len(uploaded) > MAX_UPLOAD:
        st.error(f"Upload up to {MAX_UPLOAD} PDFs")
    else:
        embed_model_name = st.session_state.get("embed_model", EMBED_MODEL_NAME)
        rerank_model_name = st.session_state.get("rerank_model", RERANK_MODEL_NAME)
        embedder = get_embedding_model(embed_model_name)
        reranker = get_reranker(rerank_model_name)
        col = client.get_or_create_collection("rag_collection")
        all_chunks = []
        all_meta = []
        all_embeddings = []
        prog = st.progress(0)
        i = 0
        for f in uploaded:
            raw = f.read()
            pages = extract_text_from_pdf_bytes(raw, f.name)
            for p in pages:
                chunks = chunk_text(p["text"])
                for c in chunks:
                    meta = {"source": p["source"], "page": p["page"], "upload_date": datetime.utcnow().isoformat()}
                    all_chunks.append(c)
                    all_meta.append(meta)
            i += 1
            prog.progress(int((i/len(uploaded))*100))
        if len(all_chunks) == 0:
            st.error("No text extracted from uploaded PDFs")
        else:
            embeddings = embedder.encode(all_chunks, show_progress_bar=True)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            ids = [md5(all_meta[idx]["source"] + "|" + str(all_meta[idx]["page"]) + "|" + str(idx)) for idx in range(len(all_meta))]
            try:
                col.add(ids=ids, documents=all_chunks, metadatas=all_meta, embeddings=embeddings.tolist())
            except Exception:
                existing = client.get_or_create_collection("rag_collection")
                existing.add(ids=ids, documents=all_chunks, metadatas=all_meta, embeddings=embeddings.tolist())
            tfidf_vect, tfidf_mat = build_tfidf(all_chunks)
            st.session_state["tfidf_vect"] = tfidf_vect
            st.session_state["tfidf_mat"] = tfidf_mat
            st.session_state["chunks"] = all_chunks
            st.session_state["meta"] = all_meta
            st.session_state["embeddings"] = embeddings
            st.success(f"Indexed {len(all_chunks)} chunks")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

st.markdown("---")
sidebar = st.sidebar
sidebar.header("Retrieval & Visualization")
filename_filter = sidebar.multiselect("Filter filenames", options=sorted({m.get('source','') for m in st.session_state.get('meta',[]) if 'source' in m}), default=None)
page_min, page_max = sidebar.slider("Page range", min_value=1, max_value=500, value=(1,500))
date_min = sidebar.date_input("Uploaded after (optional)", value=None)
weights_label = sidebar.expander("Hybrid weights")
with weights_label:
    chroma_w = st.number_input("Chroma weight", min_value=0.0, max_value=1.0, value=0.4)
    tfidf_w = st.number_input("TF-IDF weight", min_value=0.0, max_value=1.0, value=0.2)
    rerank_w = st.number_input("Reranker weight", min_value=0.0, max_value=1.0, value=0.4)
norm_total = chroma_w + tfidf_w + rerank_w
if norm_total == 0:
    chroma_w, tfidf_w, rerank_w = 0.4, 0.2, 0.4
else:
    chroma_w, tfidf_w, rerank_w = chroma_w / norm_total, tfidf_w / norm_total, rerank_w / norm_total
visualize_btn = sidebar.button("Show Embedding Heatmap")
expansions_n = sidebar.number_input("Number of query expansions", min_value=0, max_value=5, value=2)

left, right = st.columns([2,3])
with left:
    query = st.text_area("Ask a question about uploaded documents", height=160)
    top_k = st.number_input("Top K documents to retrieve", min_value=1, max_value=50, value=6)
    search_btn = st.button("Search & Answer")
with right:
    st.markdown("### Answer")
    answer_box = st.empty()
    st.markdown("### Sources")
    source_box = st.empty()

if visualize_btn:
    emb = st.session_state.get('embeddings', None)
    metas = st.session_state.get('meta', None)
    if emb is None or metas is None:
        st.error('No embeddings found. Build the index first.')
    else:
        method = sidebar.selectbox('Dim reduction', ['PCA','TSNE'])
        if method == 'PCA':
            proj = PCA(n_components=2).fit_transform(emb)
        else:
            proj = TSNE(n_components=2, init='pca').fit_transform(emb)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(proj[:,0], proj[:,1])
        for i, m in enumerate(metas[:200]):
            ax.annotate(m.get('source','')+":"+str(m.get('page','')), (proj[i,0], proj[i,1]), fontsize=6)
        st.pyplot(fig)

if search_btn:
    if not query:
        st.error("Enter a question")
    else:
        try:
            col = client.get_or_create_collection("rag_collection")
        except Exception:
            st.error("No index found. Upload PDFs and build index first.")
            st.stop()
        embed_model = get_embedding_model(st.session_state.get("embed_model", EMBED_MODEL_NAME))
        reranker = get_reranker(st.session_state.get("rerank_model", RERANK_MODEL_NAME))
        expansions = [query]
        gemini_key = st.session_state.get("gemini_key", GEMINI_API_KEY)
        gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL)
        if expansions_n > 0 and gemini_key:
            prompt = f"Generate {expansions_n} concise query rewrites or expansions for: {query}"
            try:
                exp_text = call_gemini_stream(prompt, gemini_key, gemini_model)
                candidates = [e.strip() for e in exp_text.split('\n') if e.strip()]
                for c in candidates[:expansions_n]:
                    expansions.append(c)
            except Exception:
                expansions = [query]
        q_embs = embed_model.encode(expansions)
        q_embs = np.array(q_embs)
        q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-12)
        candidate_pool = max(80, top_k * 12)
        combined = {}
        try:
            for qe in q_embs:
                chroma_res = col.query(query_embeddings=[qe.tolist()], n_results=candidate_pool, include=['documents','metadatas','distances'])
                docs = chroma_res.get('documents',[[]])[0]
                metas = chroma_res.get('metadatas',[[]])[0]
                dists = chroma_res.get('distances',[[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    if filename_filter and meta.get('source','') not in filename_filter:
                        continue
                    if not (page_min <= meta.get('page',0) <= page_max):
                        continue
                    if date_min is not None and 'upload_date' in meta:
                        try:
                            ud = datetime.fromisoformat(meta['upload_date']).date()
                            if ud < date_min:
                                continue
                        except Exception:
                            pass
                    score = 1.0 - float(dist) if dist is not None else 0.0
                    key = md5(meta.get('source','')+str(meta.get('page',''))+doc)
                    if key not in combined:
                        combined[key] = {"text": doc, "meta": meta, "chroma_score": score}
                    else:
                        combined[key]["chroma_score"] = max(combined[key].get("chroma_score",0), score)
        except Exception:
            pass
        tfidf_vect = st.session_state.get("tfidf_vect", None)
        tfidf_mat = st.session_state.get("tfidf_mat", None)
        if tfidf_vect is not None and tfidf_mat is not None:
            q_vec = tfidf_vect.transform([query])
            q_vec = normalize(q_vec)
            sims = (tfidf_mat.dot(q_vec.T)).toarray().squeeze()
            chunks = st.session_state.get("chunks", [])
            meta = st.session_state.get("meta", [])
            for idx, sim in enumerate(sims):
                if filename_filter and meta[idx].get('source','') not in filename_filter:
                    continue
                if not (page_min <= meta[idx].get('page',0) <= page_max):
                    continue
                if date_min is not None:
                    try:
                        ud = datetime.fromisoformat(meta[idx].get('upload_date','')).date()
                        if ud < date_min:
                            continue
                    except Exception:
                        pass
                key = md5(meta[idx]['source']+str(meta[idx]['page'])+chunks[idx])
                if key not in combined:
                    combined[key] = {"text": chunks[idx], "meta": meta[idx], "tfidf_score": float(sim)}
                else:
                    combined[key]["tfidf_score"] = float(sim)
        candidate_docs = list(combined.values())
        if len(candidate_docs) == 0:
            st.warning("No candidate documents retrieved")
            st.stop()
        texts = [d["text"] for d in candidate_docs]
        pairs = [(query, t) for t in texts]
        try:
            scores = reranker.predict(pairs, convert_to_numpy=True)
        except Exception:
            scores = reranker.predict(pairs)
        for i, s in enumerate(scores):
            candidate_docs[i]["rerank_score"] = float(s)
        for d in candidate_docs:
            chroma_s = d.get("chroma_score",0.0)
            tfidf_s = d.get("tfidf_score",0.0)
            rerank_s = d.get("rerank_score",0.0)
            vals = np.array([chroma_s, tfidf_s, rerank_s], dtype=float)
            maxv = vals.max() if vals.max() != 0 else 1.0
            vals = vals / maxv
            hybrid = chroma_w*vals[0] + tfidf_w*vals[1] + rerank_w*vals[2]
            d["hybrid_score"] = float(hybrid)
        reranked = sorted(candidate_docs, key=lambda x: x["hybrid_score"], reverse=True)[:max(20, top_k*3)]
        final_top = reranked[:top_k]
        prompt_chunks = []
        for r in final_top:
            src = r["meta"].get("source","unknown")
            page = r["meta"].get("page", -1)
            snippet = r["text"].strip()
            prompt_chunks.append(f"[{src}|{page}]\n{snippet}")
        memory = st.session_state.chat_memory[-2:]
        conversation_context = ""
        for item in memory:
            conversation_context += f"USER: {item['query']}\nASSISTANT: {item['answer']}\n"
        full_prompt = (
            "You are an expert assistant. Use ONLY the information in the SOURCES.\n"
            "If sources lack info, reply EXACTLY with INSUFFICIENT_DOCS.\n"
            "Cite sources inline in [filename|page] format.\n"
            "Maintain context of last 2 conversation turns.\n\n"
            + conversation_context
            + f"USER QUESTION:\n{query}\n\nSOURCES:\n" + "\n\n".join(prompt_chunks) + "\n\nFINAL ANSWER:"
        )
        try:
            answer_text = call_gemini_stream(full_prompt, gemini_key, gemini_model)
        except Exception as e:
            answer_text = f"Generation error: {e}"
        if not answer_text:
            answer_text = "INSUFFICIENT_DOCS"
        st.session_state.chat_memory.append({"query": query, "answer": answer_text})
        if len(st.session_state.chat_memory) > 2:
            st.session_state.chat_memory = st.session_state.chat_memory[-2:]
        answer_box.success(answer_text)
        df = pd.DataFrame([{"source": r["meta"].get("source",""), "page": r["meta"].get("page",-1), "hybrid_score": round(r["hybrid_score"],4), "snippet": r["text"][:300]} for r in final_top])
        source_box.dataframe(df)
