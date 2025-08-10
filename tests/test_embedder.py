# tests/test_embedder.py
import pytest
from core.embedder import Embedder

def test_embed_documents():
    embedder = Embedder()
    documents = ["Hello world", "This is a test"]
    embeddings = embedder.embed_documents(documents)
    
    # Check output shape and type
    assert len(embeddings) == len(documents)
    assert all(len(vec) == 384 for vec in embeddings)  # 384 is embedding size of all-MiniLM-L6-v2
    assert all(isinstance(vec, float) or hasattr(vec, '__float__') for emb in embeddings for vec in emb)
