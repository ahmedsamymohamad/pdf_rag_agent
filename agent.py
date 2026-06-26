import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
# Fixed typo here: changed loangchain_ollama to langchain_ollama
from langchain_ollama import ChatOllama 
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


def upload_document(file):
    file_path = os.path.join(os.getcwd(), "data", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    return file_path
        
def load_document(file_path):
    loader = DoclingLoader(file_path=file_path)
    docs = loader.load()
    return docs

def chunk_text(docs, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)
    return docs

def create_embedding_model():
    embedding_model = OllamaEmbeddings(
        model="bge-m3:latest",
        dimensions=1024,
    )
    return embedding_model

def create_vector_store(chunks, embedding_model):
    client = QdrantClient(path="./langchain_qdrant")
    
    vector_size = len(embedding_model.embed_query("sample text"))

    if client.collection_exists("pdf_rag"):
        client.delete_collection("pdf_rag")
    client.create_collection(
        collection_name="pdf_rag",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="pdf_rag",
        embedding=embedding_model,
    )

    # get from environment url and api key for qdrant cloud
    
    # url = os.getenv("ENDPOINT")
    # api_key = os.getenv("API_KEY")
    # vector_store = QdrantVectorStore.from_documents(
    #     chunks,
    #     embedding=embedding_model,
    #     url=url,
    #     prefer_grpc=True,
    #     api_key=api_key,
    #     collection_name="my_documents",
    # )
    
    vector_store.add_documents(chunks)
    client.close()
    return vector_store

def process_pdf(file):
    print("Processing PDF...")
    file_path = upload_document(file)
    document = load_document(file_path)
    chunks = chunk_text(document)
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(chunks, embedding_model)
    print("Vector store created successfully!")
    return vector_store

@tool(response_format="content_and_artifact")
def search_query(query):
    """Search for relevant information in the uploaded PDF based on the user query."""
    embedding_model = OllamaEmbeddings(
        model="bge-m3:latest",
        dimensions=1024,
    )
    
    
    vector_store = QdrantVectorStore.from_existing_collection(
        path="./langchain_qdrant",
        collection_name="pdf_rag",
        embedding=embedding_model)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    print(f"Retrieved documents for query '{query}':\n{serialized}")
    return serialized, retrieved_docs





