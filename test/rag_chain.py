import os 
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    
    file_path = os.path.join(os.getcwd(), "data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
        
    loder = DoclingLoader(file_path=file_path)
    docs = loder.load()
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    embedding_model = OllamaEmbeddings(model="bge-m3:latest", dimensions=1024)
    
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    
    vector_store.add_documents(chunks)
    
    prompt = ChatPromptTemplate([
        ("system","""
                You are a PDF assistant.

                Whenever the user asks about the uploaded PDF,
                ALWAYS use the the context and use the citation.

                Only answer from the PDF.
                
                context: {context}

                If the information is not found in the PDF,
                say:

                "I couldn't find that information in the uploaded PDF."

                Do not hallucinate.
                """),
        ("user", "{input}")
    ])
    
    chat_model = ChatOllama(model="qwen3:0.6b", temperature=0.0)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough(),
    }
    | prompt
    | chat_model
)
    
    if query := st.chat_input("user"):
        results = chain.invoke(query)
        
        
        with st.chat_message("assistant"):
            st.markdown(results.content[-1])
            
        
    