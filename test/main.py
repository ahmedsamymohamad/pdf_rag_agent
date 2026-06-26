import os
import streamlit as st

from langchain.tools import tool
from langchain.agents import create_agent

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore


# ---------------------------------
# Streamlit
# ---------------------------------

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

st.title("📄 PDF Chatbot")


# ---------------------------------
# Session State Initialization
# ---------------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "agent" not in st.session_state:
    st.session_state.agent = None


# ---------------------------------
# Cached Resources
# ---------------------------------

@st.cache_resource
def get_embedding_model():
    return OllamaEmbeddings(
        model="bge-m3:latest",
        dimensions=1024,
    )


@st.cache_resource
def get_llm():
    return ChatOllama(
        model="qcwind/qwen3-vl-4B-Q4-K-M",
    )


@st.cache_resource
def get_qdrant():
    return QdrantClient(path="./langchain_qdrant")


embedding_model = get_embedding_model()
llm = get_llm()
client = get_qdrant()


# ---------------------------------
# Upload PDF
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)


if uploaded_file is not None and not st.session_state.processed:

    with st.spinner("Processing PDF..."):

        os.makedirs("data", exist_ok=True)

        file_path = os.path.join(
            "data",
            uploaded_file.name,
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load PDF
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(docs)

        collection_name = "demo_collection"

        # Delete previous collection if it exists
        collections = [
            c.name
            for c in client.get_collections().collections
        ]

        if collection_name in collections:
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        # IMPORTANT
        vector_store.add_documents(chunks)

        st.session_state.vector_store = vector_store

        st.session_state.processed = True

        st.success("PDF processed successfully!")


# ---------------------------------
# Tool
# ---------------------------------

@tool
def search_documents(query: str) -> str:
    """
    Search relevant information inside the uploaded PDF.
    """

    if st.session_state.vector_store is None:
        return "No PDF uploaded."

    docs = vector_store.similarity_search(
        query,
        k=5,
    )

    if not docs:
        return "No relevant information found."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ---------------------------------
# Create Agent
# ---------------------------------

if st.session_state.processed and st.session_state.agent is None:

    st.session_state.agent = create_agent(
        model=llm,
        tools=[search_documents],
        system_prompt="""
You are a PDF assistant.

Whenever the user asks about the uploaded PDF,
ALWAYS use the search_documents tool first.

Only answer from the PDF.

If the information is not found in the PDF,
say:

"I couldn't find that information in the uploaded PDF."

Do not hallucinate.
""",
    )


# ---------------------------------
# Display Chat History
# ---------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------
# Chat
# ---------------------------------

if st.session_state.processed:

    prompt = st.chat_input("Ask something about the PDF...")

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = st.session_state.agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    }
                )

                answer = response["messages"][-1].content

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )