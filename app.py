import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="LangChain Core Assistant", page_icon="🔗")
st.title("🔗 LangChain Core Assistant")
st.caption("Ask anything about the LangChain Core library.")


@st.cache_resource
def load_chain():
    bm25 = BM25Encoder()
    bm25.load("data/bm25_params.json")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("langchain-core-rag")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        top_k=5,
        alpha=0.5,
        text_key="text",
    )

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_template("""\
You are a code assistant for the LangChain library.
Answer strictly using only the retrieved context below — no outside knowledge.
If the context does not contain enough information, respond with:
"The information needed to answer this question is not available in the provided context."
Do not infer, guess, or make up class names, method names, or code behaviour.

Context:
{context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('source', '')}]\n{d.page_content}" for d in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about LangChain Core..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            chain = load_chain()
            answer = chain.invoke(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
