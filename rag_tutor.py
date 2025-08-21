import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# API Key set करो
# -----------------------------
os.environ["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"] = "your_openai_api_key_here"

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="📘 AI Tutor", layout="wide")
st.title("📘 Interactive RAG-powered Tutor")

uploaded_file = st.file_uploader("📂 Upload your study material (PDF)", type=["pdf"])

if uploaded_file:
    # Step 1: PDF Load करो
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load()

    # Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Step 3: Vector DB बनाओ
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings)

    # Step 4: RAG pipeline
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Chat section
    st.subheader("💬 Ask your question:")
    query = st.text_input("Type your question here...")

    if query:
        response = qa_chain.run(query)
        st.success(response)
