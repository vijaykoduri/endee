# RAG Query System

A locally-run Retrieval-Augmented Generation (RAG) application that lets you upload documents and ask natural language questions against their content. Built with Flask, FAISS, Sentence Transformers, and a Flan-T5 language model.

---

## Project Overview

The RAG Query System allows users to:

- Upload one or more documents (PDF, DOCX, TXT)
- Ask multiple natural language questions in a single session
- Receive answers grounded in the content of the uploaded documents

Rather than relying on a general-purpose LLM with no document context, this system first retrieves the most relevant passage from your documents, then feeds it as context to the language model — resulting in more accurate, document-specific answers.

---

## System Design

The system is split into two main components: a Flask web application (`app.py`) and a RAG backend (`system.py`).

```
User Browser
     │
     ▼
┌─────────────────────────────┐
│        Flask App (app.py)   │
│                             │
│  ┌──────────┐  ┌─────────┐  │
│  │  /upload │  │ /query  │  │
│  └────┬─────┘  └────┬────┘  │
│       │              │      │
│  Text Extraction   Query    │
│  (PDF/DOCX/TXT)  Handling   │
└───────┬──────────────┬──────┘
        │              │
        ▼              ▼
┌───────────────────────────────┐
│         RAGSystem (system.py) │
│                               │
│  SentenceTransformer Encoder  │
│         (all-MiniLM-L6-v2)   │
│               │               │
│          FAISS Index          │
│    (IndexFlatIP, cosine sim)  │
└───────────────────────────────┘
        │
        ▼
┌────────────────────────────┐
│  Flan-T5 QA Pipeline       │
│  (google/flan-t5-base)     │
│  Generates final answer    │
└────────────────────────────┘
```

### Step-by-step Flow

**Document Ingestion**

1. User uploads files via the web UI.
2. `app.py` extracts text from each file using `pdfplumber` (PDF), `python-docx` (DOCX), or plain file reading (TXT).
3. Text is split into chunks — first by double newlines (paragraphs), then by sentence boundaries.
4. Chunks are passed to `RAGSystem.load_documents()`.

**Embedding & Indexing**

5. `RAGSystem` encodes all chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings).
6. Embeddings are L2-normalized and stored in a FAISS `IndexFlatIP` (inner product = cosine similarity after normalization).

**Query & Answer Generation**

7. The user submits one or more questions (one per line).
8. Each question is embedded and searched against the FAISS index (`top_k=1`).
9. The best-matching chunk is used as context in the prompt: `Context: {chunk}\nQuestion: {question}\nAnswer:`.
10. The Flan-T5 model generates an answer from this prompt.
11. Results are displayed in the UI showing the question, answer, and source context.

---

## Use of FAISS (FAISS / "Endee")

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is the vector similarity engine that powers the retrieval step of this RAG pipeline.

**Why FAISS?**

Traditional keyword search (like BM25) matches exact words. FAISS enables **semantic search** — finding passages that are conceptually similar to the question even when they use different words.

**How it's used here**

| Setting | Value |
|---|---|
| Index type | `IndexFlatIP` (exact inner product search) |
| Embedding dim | 384 (from MiniLM) |
| Normalization | L2-normalized before indexing, so inner product = cosine similarity |
| `top_k` | 1 — retrieves the single best-matching chunk per question |

**Key FAISS calls in `system.py`**

```python
# Build index
self.index = faiss.IndexFlatIP(self.embedding_dim)
self.index.add(embeddings.astype('float32'))

# Query
similarities, indices = self.index.search(query_embedding, top_k)
```

`IndexFlatIP` performs exhaustive (exact) search — suitable for small-to-medium document collections. For very large corpora, this can be swapped for an approximate index like `IndexIVFFlat` for faster retrieval at the cost of some accuracy.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- `pip`

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-query-system
```

### 2. Install Dependencies

```bash
pip install flask werkzeug pdfplumber python-docx faiss-cpu sentence-transformers transformers torch
```

> **Note:** If you have a CUDA-capable GPU and want faster inference, install `faiss-gpu` instead of `faiss-cpu` and use the appropriate PyTorch CUDA build.

### 3. Project Structure

```
rag-query-system/
├── app.py          # Flask application, file upload, text extraction, query handling
├── system.py       # RAGSystem class — embedding, FAISS indexing, retrieval
├── templates/
│   └── index.html  # Web UI
└── uploads/        # Created automatically on first upload
```

> **Important:** Move `index.html` into a `templates/` folder — Flask's `render_template` looks there by default.

```bash
mkdir templates
mv index.html templates/
```

### 4. Run the Application

```bash
python app.py
```

The app starts on `http://127.0.0.1:5000` by default.

### 5. Using the App

1. Open `http://127.0.0.1:5000` in your browser.
2. Upload one or more PDF, DOCX, or TXT files.
3. Once uploaded and processed, type your questions (one per line) in the text area.
4. Click **Submit** to receive answers with their source context.

---

## Known Issues & Limitations

- **Answer variable bug in `app.py`:** In the `/query` route, `answer = ". ".join(answer.split(". ")[:2])` references `answer` before it is assigned from the model output. Replace this with:
  ```python
  full_text = response[0]['generated_text']
  answer = full_text[len(prompt):].strip()  # Extract only the answer portion
  answer = ". ".join(answer.split(". ")[:2])
  ```
- **Flan-T5 model size:** `flan-t5-base` is lightweight but limited in reasoning depth. Consider upgrading to `flan-t5-large` or `flan-t5-xl` for better answers on complex questions.
- **No persistence:** The FAISS index is held in memory only. Restarting the server requires re-uploading documents.
- **Single-chunk retrieval:** Only `top_k=1` chunk is retrieved. Increasing this and concatenating chunks can improve answer quality for questions spanning multiple passages.

---

## Dependencies Summary

| Library | Purpose |
|---|---|
| `flask` | Web framework |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `sentence-transformers` | Text embedding (MiniLM) |
| `faiss-cpu` | Vector similarity search |
| `transformers` | Flan-T5 QA pipeline |
| `torch` | PyTorch backend for transformers |
