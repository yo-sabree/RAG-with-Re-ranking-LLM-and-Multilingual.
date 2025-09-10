# RAG PDF Q&A App

A **Streamlit** app for **Retrieval-Augmented Generation (RAG)**. Upload PDFs, build a searchable index, and ask questions. Answers are generated via **Google Gemini API**, with sources cited.

---

## Features

- Upload multiple PDFs (up to 15)
- Extract and chunk text
- Store embeddings in **ChromaDB**
- Search & rerank with **TF-IDF** + **CrossEncoder**
- Generate answers using **Gemini API**
- Inline source citation `[filename|page]`
- Clear & rebuild the index anytime

---
