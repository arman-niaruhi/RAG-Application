# tests/test_rag_chatbot.py
import pytest
from unittest.mock import patch, MagicMock
from services.rag_chatbot import RAGChatbot

@patch("services.rag_chatbot.Embedder.embed_documents")
@patch("services.rag_chatbot.RecursiveSplitter.split_text")
@patch("services.rag_chatbot.TextExtractor.extract_text")
def test_ingest_pdf(mock_extract_text, mock_split_text, mock_embed_documents):
    # Setup mocks
    mock_extract_text.return_value = "This is some extracted text from pdf."
    mock_split_text.return_value = ["chunk1", "chunk2"]
    mock_embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

    rag_bot = RAGChatbot()
    pdf_path = "dummy.pdf"
    
    # Should not raise and process correctly
    rag_bot.ingest_pdf(pdf_path)

    mock_extract_text.assert_called_once_with(pdf_path)
    mock_split_text.assert_called_once_with("This is some extracted text from pdf.")
    mock_embed_documents.assert_called_once_with(["chunk1", "chunk2"])
