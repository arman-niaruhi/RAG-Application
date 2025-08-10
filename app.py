import streamlit as st
from core.extractor import PdfPlumberExtractor
from core.splitter import RecursiveSplitter
from core.embedder import Embedder
from core.vector_store import FAISSVectorStore
from core.language_model import LanguageModel
from services.rag_chatbot import RAGChatbot
from utils.file_utils import save_uploaded_file
import logging

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)

st.set_page_config(
    page_title="PDF RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1. Upload a PDF file  
        2. Wait for processing  
        3. Ask any question about the PDF content  
        
        Powered by Retrieval-Augmented Generation (RAG)
        """
    )
    st.markdown("---")
    st.write("Developed by Arman")

st.title("PDF-based RAG Chatbot")

@st.cache_resource(show_spinner=False)
def init_components():
    extractor = PdfPlumberExtractor()
    splitter = RecursiveSplitter()
    embedder = Embedder()
    vector_store = FAISSVectorStore()
    language_model = LanguageModel()
    return RAGChatbot(extractor, splitter, embedder, vector_store, language_model)

rag_bot = init_components()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], help="Upload your PDF to start.")

if uploaded_file:
    # Save the uploaded PDF path once
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = save_uploaded_file(uploaded_file)
        st.session_state.pdf_ingested = False  # Mark as not processed yet

    if not st.session_state.pdf_ingested:
        with st.spinner("Processing PDF and building knowledge base..."):
            rag_bot.ingest_pdf(st.session_state.pdf_path)
        st.session_state.pdf_ingested = True
        st.success("PDF ingested successfully!")

    question = st.text_input("Ask a question about the PDF", placeholder="Type your question here and press Enter")

    if question:
        print(question)
        with st.spinner("Generating answer..."):
            answer = rag_bot.answer_question(question)
        st.markdown("### Answer:")
        st.info(answer)

else:
    st.info("Please upload a PDF file to get started.")
    # Reset ingestion state if no file uploaded
    st.session_state.pdf_ingested = False
    if "pdf_path" in st.session_state:
        del st.session_state.pdf_path

st.markdown("---")
st.markdown(
    '<div style="text-align:center; font-size:0.8rem; color:gray;">'
    'Built with Streamlit | PDF RAG Assistant'
    '</div>',
    unsafe_allow_html=True,
)
