import streamlit as st
from core.extractor import PdfPlumberExtractor
from core.splitter import RecursiveSplitter
from core.embedder import Embedder
from core.vector_store import FAISSVectorStore
from core.language_model import LanguageModel
from services.rag_chatbot import RAGChatbot
from utils.file_utils import save_uploaded_file
import logging

import os

# Check if the API key is set correctly
print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))


logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)


st.title("PDF-based RAG Chatbot")

# Initialize all components
extractor = PdfPlumberExtractor()
splitter = RecursiveSplitter()
embedder = Embedder()
vector_store = FAISSVectorStore()
language_model = LanguageModel()

rag_bot = RAGChatbot(extractor, splitter, embedder, vector_store, language_model)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save PDF to local temp folder
    pdf_path = save_uploaded_file(uploaded_file)

    # Ingest the PDF to build the knowledge base
    with st.spinner("Processing PDF..."):
        rag_bot.ingest_pdf(pdf_path)
    st.success("PDF ingested successfully!")

    # Ask user for questions
    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Getting answer..."):
            answer = rag_bot.answer_question(question)
        st.markdown("**Answer:**")
        st.write(answer)
