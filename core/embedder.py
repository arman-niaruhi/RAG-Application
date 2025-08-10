import logging
logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, no API key needed
    
    def embed_documents(self, documents):
        # documents is a list of texts (chunks)
        return self.model.encode(documents, show_progress_bar=True)


