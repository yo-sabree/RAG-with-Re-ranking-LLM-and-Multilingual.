import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

from dotenv import load_dotenv
load_dotenv()
import os
import io
import json
import time
import gc
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import fitz
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import google.genai as genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD", "15"))

st.set_page_config(page_title="RAG", layout="wide")
st.title("RAG (Upload, Ask.)")

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
        chunk = text[start:end].strip()
        chunks.append(chunk)
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
        prog = st.progress(0)
        i = 0
        for f in uploaded:
            raw = f.read()
            pages = extract_text_from_pdf_bytes(raw, f.name)
            for p in pages:
                chunks = chunk_text(p["text"])
                for c in chunks:
                    meta = {"source": p["source"], "page": p["page"]}
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
            st.success(f"Indexed {len(all_chunks)} chunks")

st.markdown("---")

left, right = st.columns([2,3])
with left:
    query = st.text_area("Ask a question about uploaded documents", height=160)
    top_k = st.number_input("Top K documents to retrieve", min_value=1, max_value=20, value=6)
    search_btn = st.button("Search & Answer")
with right:
    st.markdown("### Answer")
    answer_box = st.empty()
    st.markdown("### Sources")
    source_box = st.empty()

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
        q_emb = embed_model.encode([query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) or 1.0)
        candidate_pool = max(40, top_k * 8)
        try:
            chroma_res = col.query(query_embeddings=[q_emb.tolist()], n_results=candidate_pool, include=['documents','metadatas','distances'])
        except Exception:
            chroma_res = {"documents":[[]], "metadatas":[[]], "distances":[[]]}
        ch_docs = []
        if "documents" in chroma_res and len(chroma_res["documents"])>0:
            docs = chroma_res["documents"][0]
            metas = chroma_res["metadatas"][0]
            dists = chroma_res.get("distances",[[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                score = 1.0 - float(dist) if dist is not None else 0.0
                ch_docs.append({"text": doc, "meta": meta, "score": score})
        tfidf_vect = st.session_state.get("tfidf_vect", None)
        tfidf_mat = st.session_state.get("tfidf_mat", None)
        combined = {md5(x["meta"]["source"]+str(x["meta"]["page"])+x["text"]): x for x in ch_docs}
        if tfidf_vect is not None and tfidf_mat is not None:
            q_vec = tfidf_vect.transform([query])
            q_vec = normalize(q_vec)
            sims = (tfidf_mat.dot(q_vec.T)).toarray().squeeze()
            top_idx = sims.argsort()[-candidate_pool:][::-1]
            chunks = st.session_state.get("chunks", [])
            meta = st.session_state.get("meta", [])
            for idx in top_idx:
                if idx < len(chunks):
                    key = md5(meta[idx]["source"]+str(meta[idx]["page"])+chunks[idx])
                    if key not in combined:
                        combined[key] = {"text": chunks[idx], "meta": meta[idx], "score": float(sims[idx])}
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
        reranked = []
        for i, s in enumerate(scores):
            reranked.append({"text": texts[i], "meta": candidate_docs[i]["meta"], "score": float(s)})
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)[:max(10, top_k*2)]
        final_top = reranked[:top_k]
        prompt_chunks = []
        for r in final_top:
            src = r["meta"].get("source","unknown")
            page = r["meta"].get("page", -1)
            snippet = r["text"].strip()
            prompt_chunks.append(f"[{src}|{page}]\n{snippet}")
        instruction = (
            "You are an expert assistant. When answering the USER QUESTION, follow these rules:\n"
            "- Use ONLY the information in the SOURCES if the question requires knowledge from them.\n"
            "- If the SOURCES do not contain enough information to answer, reply EXACTLY with INSUFFICIENT_DOCS.\n"
            "- If the USER QUESTION is casual (e.g., greetings or small talk), respond naturally and conversationally (e.g., 'Hi, how are you?').\n"
            "- Cite sources inline in [filename|page] format whenever you use them.\n"
            "- Do not invent facts. Keep answers concise, direct, and accurate. End with a short sources list when sources are used."
        )
        full_prompt = instruction + "\n\nUSER QUESTION:\n" + query.strip() + "\n\nSOURCES:\n" + "\n\n".join(prompt_chunks) + "\n\nFINAL ANSWER:"
        gemini_key = st.session_state.get("gemini_key", GEMINI_API_KEY)
        gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL)
        try:
            answer_text = call_gemini_stream(full_prompt, gemini_key, gemini_model)
        except Exception as e:
            answer_text = f"Generation error: {e}"
        if not answer_text:
            answer_text = "INSUFFICIENT_DOCS"
        answer_box.success(answer_text)
        df = pd.DataFrame([{"source": r["meta"].get("source",""), "page": r["meta"].get("page",-1), "score": round(r["score"],4), "snippet": r["text"][:300]} for r in final_top])
        source_box.dataframe(df)
