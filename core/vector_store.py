from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings

import logging
import faiss
from langchain.vectorstores import FAISS

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self):
        logger.info("Initializing FAISS Vector Store")
        self.vector_store = None

    def create_vector_store(self, embeddings, documents):
        logger.info(f"Creating vector store with {len(documents)} documents")
        try:
            self.vector_store = FAISS.from_embeddings(documents, embeddings)
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def similarity_search(self, query_embedding, k=4):
        logger.debug(f"Performing similarity search with top {k} results")
        if not self.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search_by_vector(query_embedding, k=k)

