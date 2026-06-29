## Agentic AI Design

The chatbot is not a simple chain (retrieve → prompt → LLM). It uses a **ReAct agent** built with LangGraph, where the LLM decides when and how to call retrieval as a tool, reasons over the results, and then produces a grounded answer.

### Why Agentic over a Simple Chain

| Concern | Simple RAG chain | ReAct Agent |
|---|---|---|
| Retrieval decision | Always retrieves | LLM skips retrieval for conversational turns (e.g. greetings) |
| Multi-step reasoning | One pass | Can re-query or reason over tool output before answering |
| Extensibility | Hard-coded steps | New tools (e.g. code execution) added without restructuring |
| Transparency | Opaque | Tool calls and observations are explicit in the message trace |

### Agent Architecture

```
User message
    │
    ▼
┌─────────────────────────────────┐
│         LangGraph ReAct         │
│                                 │
│  ┌──────────┐   tool_calls?     │
│  │  Claude  │──────────────┐    │
│  │  (LLM)   │              ▼    │
│  │          │  ┌─────────────┐  │
│  │          │◄─│ search_     │  │
│  │          │  │ langchain_  │  │
│  └──────────┘  │ docs (tool) │  │
│       │        └─────┬───────┘  │
│       │              │          │
│       │        Pinecone hybrid  │
│       │        search (top_k=5) │
└───────┼──────────────┘──────────┘
        │
        ▼
   Final answer (streamed)
```

### Tool: `search_langchain_docs`

The single tool available to the agent. The LLM decides whether to call it and what query to use.

```
Input : query (str) — natural language question or keyword
Output: concatenated chunk text with source paths, or a no-results message
```

- Backed by `PineconeHybridSearchRetriever` (alpha=0.5, top_k=5)
- Returns up to 5 ranked chunks with their `[source]` file path prefix
- Errors inside the tool are caught and returned as a string so the agent can reason about the failure rather than crashing

### Agent Loop

Each user message triggers one or more iterations of the ReAct loop:

1. **Reason** — Claude reads the conversation history and system prompt
2. **Act** — If information is needed, emit a `tool_calls` request for `search_langchain_docs`
3. **Observe** — Tool result is appended to the message trace as a `ToolMessage`
4. **Respond** — Claude generates the final answer grounded in the retrieved context

For conversational turns (greetings, thanks) the agent skips steps 2–3 and responds directly.

### Behavioral Guardrails (System Prompt)

- Answer **only** from retrieved context — no outside knowledge
- If context is insufficient, respond with a fixed refusal string rather than hallucinating
- Never invent class names, method signatures, or code behaviour
- Skip retrieval for conversational messages that don't require documentation

### Multi-turn Conversation

Full message history (`HumanMessage` / `AIMessage`) is passed to the agent on every turn. This lets Claude refer back to earlier answers and maintain conversational context without requiring a separate memory module.

### Streaming

The agent streams with `stream_mode="messages"`, yielding `(chunk, metadata)` pairs. Only chunks where `langgraph_node == "agent"` are rendered:

- Chunks with `tool_call_chunks` → show a "Searching docs…" indicator
- Chunks with text content blocks → accumulate into the visible response

Anthropic returns content as typed blocks (`[{'type': 'text', 'text': '...'}]`) rather than a plain string; the app extracts text from each block before appending to the response buffer.

## Tech Stack
| Component | Tool |
|---|---|
| Parsing | LangChain `GenericLoader` + `LanguageParser` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Dense embedding | OpenAI `text-embedding-3-small` |
| Sparse embedding | `BM25Encoder` (pinecone-text) |
| Vector store | Pinecone serverless |
| Agent framework | LangGraph `create_react_agent` |
| LLM | Anthropic Claude (claude-sonnet-4-6) |
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
