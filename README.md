
# University Academic RAG Database

This folder contains the pre-built vector database
for the University Student Academic Knowledge Assistant.

## Files

### university_faiss.index

FAISS vector index containing:

39 document chunk embeddings.

### university_metadata.pkl

Metadata and original text for every chunk.

Metadata includes:

- file name
- source
- page number
- chunk number
- total pages
- chunk ID
- document type
- original chunk text

### university_config.json

Configuration information including:

- embedding model
- embedding dimension
- chunk size
- chunk overlap
- vector database
- document count
- chunk count

## Embedding Model

sentence-transformers/all-MiniLM-L6-v2

## Documents

6

## Chunks

39
