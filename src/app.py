import os
import uuid
import streamlit as st 
from ingestion import collect_supported_files
from embeddings import load_model
from retrieval import search_multiple_sources
from generation import build_context, build_rag_prompt, ask_llm

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
@st.cache_resource
def get_embedding_model():
    return load_model()

def save_uploaded_files(uploaded_files, base_dir="uploaded_files"):
    session_dir = os.path.join(base_dir, str(uuid.uuid4()))
    os.makedirs(session_dir, exist_ok=True)
    
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(session_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    return saved_paths
            
st.title("RAG Search App")
st.write("This is the UI for your FAISS + Gemini RAG pipeline.")

if st.button("Clear chat history"):
    st.session_state.chat_history = []

query = st.text_input("Enter your question")
path = st.text_input("Enter a file or folder path", value="data")
top_k = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=5)
uploaded_files = st.file_uploader(
    "Or upload files",
    type=["csv", "xlsx", "pdf"],
    accept_multiple_files=True
)

if st.button("Run"):
    if uploaded_files:
        file_paths = save_uploaded_files(uploaded_files)
    else:
        file_paths = collect_supported_files(path)

    source_mode = "uploaded files" if uploaded_files else "file/folder path"

    if not file_paths:
        st.error("No supported files found.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        st.info(
            f"Source mode: {source_mode} | "
            f"Files found: {len(file_paths)}"
        )

        st.write("Using files:")
        for file_path in file_paths:
            st.write(f"- {file_path}")

        with st.spinner("Loading embedding model and retrieving results..."):
            embedding_model = get_embedding_model()
            results = search_multiple_sources(file_paths, query, embedding_model, top_k=top_k)
            context = build_context(results)
            prompt = build_rag_prompt(query, context)
            answer = ask_llm(prompt)

        st.success(f"Retrieved {len(results)} results.")
        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": answer
            }
        )
        
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})
            source = metadata.get("source")
            file_path = metadata.get("file_path")
            distance = item.get("distance")

            if source == "tabular":
                title = metadata.get("title")
                category = metadata.get("category")
                label = f"{i}. {title} | {category} | distance: {distance:.4f}"
            elif source == "pdf":
                page = metadata.get("page")
                chunk_id = metadata.get("chunk_id")
                label = f"{i}. PDF page {page} | chunk {chunk_id} | distance: {distance:.4f}"
            else:
                label = f"{i}. {file_path}"
                
            with st.expander(label):
                st.write(f"File: {file_path}")
                st.write(f"Metadata: {metadata}")
                st.write("Content:")
                st.write(item["unit"])

if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, item in enumerate(st.session_state.chat_history, start=1):
        with st.expander(f"Q{i}: {item['question']}"):
            st.write(item["answer"])
