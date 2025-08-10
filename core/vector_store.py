import logging
from langchain_community.vectorstores import FAISS
from core.embedder import Embedder

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self):
        logger.info("Initializing FAISS Vector Store")
        self.vector_store = None
        self.embedding_function = Embedder

    def create_vector_store(self, embeddings, documents):
        """
        Create a FAISS vector store from precomputed embeddings and documents.
        """
        logger.info(f"Creating vector store with {len(documents)} documents")
        try:
            texts = [doc.page_content for doc in documents]

            if len(texts) != len(embeddings):
                logger.error(f"Length mismatch: {len(texts)} texts vs {len(embeddings)} embeddings")
                raise ValueError("Texts and embeddings length mismatch")

            text_embedding_pairs = list(zip(texts, embeddings))

            # Pass embedding function object as second argument!
            self.vector_store = FAISS.from_embeddings(text_embedding_pairs, self.embedding_function)

            logger.info("Vector store created successfully")

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def similarity_search(self, query_embedding, k=4):
        """
        Search for top-k similar documents using a query embedding vector.
        """
        logger.debug(f"Performing similarity search with top {k} results")

        if self.vector_store is None:
            logger.error("Vector store has not been initialized")
            raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search_by_vector(query_embedding, k=k)
