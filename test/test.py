import os 
# import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
# st.title("PDF Chatbot")

# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


file_path = "./data/AI_Assignment_DSPy.pdf"

# with open(file_path, "wb") as f:
#     f.write(uploaded_file.getvalue())
    
loder = DoclingLoader(file_path=file_path)
docs = loder.load()

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

embedding_model = OllamaEmbeddings(model="bge-m3:latest", dimensions=1024)

# vector_store = InMemoryVectorStore(embedding=embedding_model)

client = QdrantClient(path="./langchain_qdrant")
collections = client.get_collections().collections
collection_names = [c.name for c in collections]

if "demo_collection" not in collection_names:
    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(
            size=1024,  # see note below
            distance=Distance.COSINE,
        ),
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embedding_model,
)

vector_store.add_documents(chunks)

prompt = ChatPromptTemplate([
    ("system","""
            You are a PDF assistant.

            Whenever the user asks about the uploaded PDF,
            ALWAYS use the the context and use the citation.

            Only answer from the PDF.
            
            Include the source in the citation.
            
            context: {context}

            If the information is not found in the PDF,
            say:

            "I couldn't find that information in the uploaded PDF."

            Do not hallucinate.
            """),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chat_model = ChatOllama(model="qwen3:0.6b", temperature=0.0)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['source']}]\n{d.page_content}"
        for d in docs
    )
    
chat_history = []

chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
        "chat_history": lambda x: chat_history,
    }
    | prompt
    | chat_model
)

# if query := st.chat_input("user"):
#     results = chain.invoke(query)
while True:
    query = input("\n\nEnter your query:\n")
    
    

    retrieved_docs = retriever.invoke(query)
    
    context = format_docs(retrieved_docs)
    
    messages = prompt.invoke({"context": context, "input": query, "chat_history": chat_history})
    
    citation = []
    response =[]
    if not query:
        break
    for i,chunk in enumerate(chat_model.stream(messages)):
        print(chunk.content, end="", flush=True)

    print(f"{retrieved_docs[0].metadata['source']}")
    for doc in retrieved_docs:
        print(
            f"Page {doc.metadata.get('page', 'Unknown')} "
        )
        print(doc)
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response))
    # print("assistant:", results.content)

    
    # with st.chat_message("assistant"):
    #     st.markdown(results.content[-1])
        
    
