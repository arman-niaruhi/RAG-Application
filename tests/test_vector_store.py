import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from core.vector_store import FAISSVectorStore


# Mock document class with page_content attribute
class MockDocument:
    def __init__(self, content):
        self.page_content = content

def test_create_vector_store_success():
    # Prepare test data
    documents = [MockDocument("Doc 1"), MockDocument("Doc 2")]
    embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

    faiss_store = FAISSVectorStore()

    # Patch FAISS.from_embeddings to avoid real FAISS call
    with patch('langchain.vectorstores.FAISS.from_embeddings') as mock_from_embeddings:
        mock_from_embeddings.return_value = "mock_faiss_index"

        faiss_store.create_vector_store(embeddings, documents)

        # Check that from_embeddings called once with zipped pairs (list of tuples)
        expected_text_embeddings = list(zip([d.page_content for d in documents], embeddings))
        mock_from_embeddings.assert_called_once_with(expected_text_embeddings)

        # Check the vector_store attribute is set correctly
        assert faiss_store.vector_store == "mock_faiss_index"

def test_create_vector_store_length_mismatch():
    documents = [MockDocument("Doc 1")]
    embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]  # More embeddings than docs

    faiss_store = FAISSVectorStore()

    with pytest.raises(ValueError, match="Texts and embeddings length mismatch"):
        faiss_store.create_vector_store(embeddings, documents)

def test_similarity_search_success():
    query_embedding = np.array([0.5, 0.6])

    faiss_store = FAISSVectorStore()
    faiss_store.vector_store = MagicMock()
    faiss_store.vector_store.similarity_search_by_vector.return_value = ["doc1", "doc2"]

    results = faiss_store.similarity_search(query_embedding, k=2)

    faiss_store.vector_store.similarity_search_by_vector.assert_called_once_with(query_embedding, k=2)
    assert results == ["doc1", "doc2"]

def test_similarity_search_not_initialized():

    faiss_store = FAISSVectorStore()

    with pytest.raises(ValueError, match="Vector store not initialized"):
        faiss_store.similarity_search(np.array([0.1, 0.2]))
