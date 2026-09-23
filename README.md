# 🎓 University Academic Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application designed to help university students ask questions about academic documents.

The application uses multiple academic PDF documents as its knowledge source.

The original PDF documents are processed offline in Google Colab. Their text is chunked, converted into embeddings, and stored in a FAISS vector database.

The Streamlit application uses the pre-built FAISS database instead of processing the original PDFs every time the application starts.

---

## 🚀 Features

- Multiple academic documents
- Pre-built RAG knowledge base
- FAISS vector database
- Semantic similarity search
- Sentence Transformer embeddings
- Groq LLM
- `openai/gpt-oss-120b`
- Source traceability
- File name and page number metadata
- Retrieved chunk display
- Similarity scores
- Streamlit user interface
- Streamlit Cloud deployment ready

---

## 🏗️ Architecture

```text
Academic PDFs
      ↓
Google Drive
      ↓
Google Colab
      ↓
Text Extraction
      ↓
Page-wise Chunking
      ↓
Sentence Transformer
      ↓
Embeddings
      ↓
FAISS
      ↓
Metadata
      ↓
GitHub
      ↓
Streamlit Cloud
      ↓
Student Question
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Source Metadata
      ↓
Groq
      ↓
openai/gpt-oss-120b
      ↓
Answer + Sources
