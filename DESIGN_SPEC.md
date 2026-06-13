# RAG Pipeline Design Spec — LangChain Core Chatbot

## Goal
Build a RAG pipeline that answers questions about LangChain's source code using `langchain-core` as the knowledge base.

## Tech Stack:
- Streamlit for UI
- LangChain for AI Orchestration
- UV for python package manager
- .env files for credentials, and include this in .gitignore

## Data Source
- Folder: `langchain/libs/core/langchain_core/` only
- Fetch strategy: Already present in the working folder. Using pre-cloned data.

## Pipeline

### 1. Parsing
- Loader: `GenericLoader` with `LanguageParser`
- Strategy: AST-based parsing using tree-sitter

### 2. Chunking
- Splitter: `RecursiveCharacterTextSplitter` with Python language
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Split priority: class → def → blank line → newline → character

### 3. Metadata
Attach to every chunk:
- `source` — file path
- `module` — e.g. runnables, prompts
- `filename` — e.g. base.py
- `package` — langchain-core

### 4. Embedding
- Dense: `text-embedding-3-small` (1536 dimensions) — OpenAI embedding model
- Sparse: `BM25Encoder` — classical keyword algorithm, fitted on all chunks, saved to disk

### 5. Storage
- Vector store: Pinecone serverless
- Index metric: `dotproduct` (required for hybrid search)
- Each record stores: dense vector + sparse vector + metadata

### 6. Retrieval
- Strategy: Hybrid search (dense + sparse)
- Retriever: `PineconeHybridSearchRetriever`
- top_k: 5
- alpha: 0.5 (equal weight dense and sparse)

### 7. Generation
- LLM: Claude (claude-sonnet-4-6)
- Prompt guidelines:
  - Answer strictly using only the retrieved RAG context — no outside knowledge
  - If the retrieved context does not contain enough information to answer the question, respond with: "The information needed to answer this question is not available in the provided context"
  - Do not infer, guess, or supplement with general knowledge
  - Do not make up class names, method names, or code behaviour

### 8. Evaluation
- Framework: RAGAS
- Metrics: faithfulness, answer relevancy, context precision, context recall
- Test set: 15-20 manually created question-answer pairs
- Target scores: 0.8+ across all metrics

## Tech Stack
| Component | Tool |
|---|---|
| Parsing | LangChain `GenericLoader` + `LanguageParser` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Dense embedding | OpenAI `text-embedding-3-small` |
| Sparse embedding | `BM25Encoder` (pinecone-text) |
| Vector store | Pinecone serverless |
| LLM | Anthropic Claude |
| Evaluation | RAGAS |

## Notebook Structure
Split across three notebooks by concern. Run indexing once, iterate freely on query and eval.

### `rag_pipeline.ipynb` — Indexing (run once)
| Module | Stage | Run |
|---|---|---|
| 1 | Load `.py` files with `GenericLoader` | Once |
| 2 | AST parse with `LanguageParser` — inspect sample doc | Once |
| 3 | Chunk with `RecursiveCharacterTextSplitter` — attach metadata | Once |
| 4 | Fit BM25 on all chunks + save to `bm25_params.json` | Once |
| 5 | Embed with `text-embedding-3-small` + upsert to Pinecone | Once (costs money) |

### `rag_query.ipynb` — Retrieval & Generation (tweak freely)
| Module | Stage | Run |
|---|---|---|
| 6 | Load BM25 from disk + connect Pinecone + set up retriever | Tweak freely |
| 7 | Set up RAG chain with prompt + Claude — test sample questions | Tweak freely |

### `rag_eval.ipynb` — Evaluation (run when ready)
| Module | Stage | Run |
|---|---|---|
| 8 | RAGAS evaluation — score faithfulness, relevancy, precision, recall | When ready |

### `app.py` — Streamlit Chatbot
Interactive chat UI. Loads BM25 + Pinecone + RAG chain on startup. Run with:
```
.venv/bin/streamlit run app.py
```

## Improvement Loop
Run RAGAS → identify weak metric → tune one parameter → re-evaluate:
- Low context precision → adjust alpha or chunk size
- Low context recall → increase top_k
- Low faithfulness → improve prompt
- Low answer relevancy → improve prompt or retrieval
