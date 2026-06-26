import asyncio
from langchain_qdrant import QdrantVectorStore
import streamlit as st
from agent import process_pdf, search_query
from langchain_ollama import ChatOllama 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

st.title("PDF chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 1. Process PDF
if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    st.success("PDF processed successfully!")

# 2. Chat interface and model
if st.session_state.vector_store is not None:

    # Persist the core model model
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = ChatOllama(model="qwen3:0.6b", temperature=0.0)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render visible message history to UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about your PDF"):
        # Display and save user query
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # 1. Manually pull context from your agent file function
        context, retrieved_docs = search_query.func(query)
        
        with st.chat_message("assistant"):
            def generate():
                # 2. Build explicit Message Objects for Ollama
                # We build a strict prompt payload mapping the manual context cleanly
                system_instruction = SystemMessage(content=f"""You are a pdf chat assistant.
Your task is to answer the user's question using ONLY the provided context below.
Cite the source metadata when you use information from the context.
If the answer cannot be found in the context, answer that you don't know.

Context:
{context}""")

                # Construct chat payload history + the new user question
                formatted_messages = [system_instruction]
                for m in st.session_state.messages[:-1]: # Include older chat history
                    if m["role"] == "user":
                        formatted_messages.append(HumanMessage(content=m["content"]))
                    else:
                        formatted_messages.append(AIMessage(content=m["content"]))
                
                # Append current query
                formatted_messages.append(HumanMessage(content=query))

                # 3. Stream directly via the raw model wrapper
                for chunk in st.session_state.chat_model.stream(formatted_messages):
                    if chunk.content:
                        yield chunk.content
                
                
                for doc in retrieved_docs:
                    pages = sorted({
                    prov["page_no"]
                    for item in doc.metadata.get("dl_meta", {}).get("doc_items", [])
                    for prov in item.get("prov", [])
                    if "page_no" in prov
                        })
                    docs_pages = ",".join(map(str, pages))
                                            
                            
                    citation = f"Source: {doc.metadata.get('source', 'Unknown')}\n From pages: {docs_pages}\nContent: {doc.page_content[:200]}"
           
                yield f"\n\nCitations:\n{citation}"
            result = st.write_stream(generate())
            
        # Save assistant's answer to session state
        st.session_state.messages.append({"role": "assistant", "content": result})
        