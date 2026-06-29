import logging
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rag_app")

load_dotenv()

st.set_page_config(page_title="LangChain Core Assistant", page_icon="🔗")
st.title("🔗 LangChain Core Assistant")
st.caption("Ask anything about the LangChain Core library.")


@st.cache_resource
def load_agent():
    try:
        log.info("Loading BM25 encoder from data/bm25_params.json")
        bm25 = BM25Encoder()
        bm25.load("data/bm25_params.json")
        log.info("BM25 encoder loaded")
    except Exception as e:
        log.exception("Failed to load BM25 encoder")
        raise RuntimeError(f"BM25 load failed: {e}") from e

    try:
        log.info("Initialising OpenAI embeddings (text-embedding-3-small)")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
        log.info("OpenAI embeddings initialised")
    except KeyError:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment / .env file")
    except Exception as e:
        log.exception("Failed to initialise OpenAI embeddings")
        raise RuntimeError(f"OpenAI embeddings failed: {e}") from e

    try:
        log.info("Connecting to Pinecone index 'langchain-core-rag'")
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index("langchain-core-rag")
        stats = index.describe_index_stats()
        log.info("Pinecone index stats: %s", stats)
    except KeyError:
        raise RuntimeError("PINECONE_API_KEY is not set in the environment / .env file")
    except Exception as e:
        log.exception("Failed to connect to Pinecone index")
        raise RuntimeError(f"Pinecone connection failed: {e}") from e

    try:
        log.info("Building PineconeHybridSearchRetriever")
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25,
            index=index,
            top_k=5,
            alpha=0.5,
            text_key="text",
        )
        log.info("Retriever ready")
    except Exception as e:
        log.exception("Failed to build retriever")
        raise RuntimeError(f"Retriever init failed: {e}") from e

    @tool
    def search_langchain_docs(query: str) -> str:
        """Search LangChain Core documentation for information about classes, methods, or concepts."""
        log.info("search_langchain_docs called with query: %r", query)
        try:
            docs = retriever.invoke(query)
            log.info("Retrieved %d docs for query %r", len(docs), query)
            if not docs:
                log.warning("No docs returned for query: %r", query)
                return "No relevant documentation found."
            return "\n\n".join(
                f"[{d.metadata.get('source', '')}]\n{d.page_content}" for d in docs
            )
        except Exception as e:
            log.exception("search_langchain_docs failed for query %r", query)
            return f"[Error retrieving docs: {e}]"

    try:
        log.info("Initialising ChatAnthropic (claude-sonnet-4-6)")
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        log.info("LLM initialised")
    except KeyError:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment / .env file")
    except Exception as e:
        log.exception("Failed to initialise ChatAnthropic")
        raise RuntimeError(f"LLM init failed: {e}") from e

    system_prompt = """\
You are a code assistant for the LangChain Core library.
Use the search_langchain_docs tool to retrieve relevant documentation before answering questions.
Answer strictly using only the retrieved context — no outside knowledge.
If the context does not contain enough information, respond with:
"The information needed to answer this question is not available in the provided context."
Do not infer, guess, or make up class names, method names, or code behaviour.
For conversational messages (greetings, thanks, etc.) that don't require documentation, respond directly without using the tool.
"""

    agent = create_react_agent(llm, [search_langchain_docs], prompt=system_prompt)
    log.info("Agent created successfully")
    return agent


def build_messages():
    msgs = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs


if "messages" not in st.session_state:
    st.session_state.messages = []

# Eagerly load agent on startup so errors surface immediately in the sidebar.
try:
    agent = load_agent()
except Exception as boot_err:
    st.error(f"**Agent failed to load:** {boot_err}")
    log.exception("Agent failed to load at startup")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about LangChain Core..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        search_indicator = st.empty()
        response_placeholder = st.empty()
        full_response = ""
        chunk_count = 0

        try:
            log.info("Starting agent stream for question: %r", question)
            for chunk, metadata in agent.stream(
                {"messages": build_messages()},
                stream_mode="messages",
            ):
                chunk_count += 1
                node = metadata.get("langgraph_node", "")
                log.debug(
                    "Stream chunk #%d | node=%r | type=%s | content=%r",
                    chunk_count,
                    node,
                    type(chunk).__name__,
                    getattr(chunk, "content", "")[:120],
                )

                if node == "agent":
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        search_indicator.markdown("🔍 *Searching LangChain docs...*")
                    else:
                        # Anthropic returns content as a list of typed blocks;
                        # fall back to plain string for other providers.
                        if isinstance(chunk.content, list):
                            text = "".join(
                                block.get("text", "")
                                for block in chunk.content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        elif isinstance(chunk.content, str):
                            text = chunk.content
                        else:
                            text = ""

                        if text:
                            search_indicator.empty()
                            full_response += text
                            response_placeholder.markdown(full_response + "▌")

            log.info(
                "Stream complete. chunk_count=%d full_response_len=%d",
                chunk_count,
                len(full_response),
            )

            if not full_response:
                log.warning("full_response is empty after streaming (%d chunks seen)", chunk_count)
                response_placeholder.warning(
                    "No response was generated. Check the terminal logs for details."
                )
        except Exception as stream_err:
            log.exception("Error during agent stream")
            response_placeholder.error(f"**Stream error:** {stream_err}")
            full_response = f"[Error: {stream_err}]"

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
