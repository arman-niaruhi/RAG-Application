import logging
from langchain_community.vectorstores import FAISS


logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self):
        # Initialize the vector store attribute as None
        logger.info("Initializing FAISS Vector Store")
        self.vector_store = None

    def create_vector_store(self, embeddings, documents):
        """
        Creates a FAISS vector store from the given documents and their embeddings.

        Parameters:
        - embeddings: List of vector embeddings (e.g., from OpenAI or Ollama embeddings)
        - documents: List of document objects, each with a 'page_content' attribute containing text
        """
        logger.info(f"Creating vector store with {len(documents)} documents")
        try:
            # Extract the text content
            texts = [doc.page_content for doc in documents]
            
            # Ensure inputs are lists and have matching lengths
            if not isinstance(texts, list):
                texts = list(texts)
            if not isinstance(embeddings, list):
                embeddings = list(embeddings)
            
            if len(texts) != len(embeddings):
                logger.error(f"Length mismatch: {len(texts)} texts, {len(embeddings)} embeddings")
                raise ValueError("Texts and embeddings length mismatch")
            
            # Debug log sample
            logger.debug(f"First text: {texts[0] if texts else 'No texts'}")
            logger.debug(f"First embedding shape: {embeddings[0].shape if embeddings else 'No embeddings'}")

            # Combine texts and embeddings into the required format
            text_embeddings = list(zip(texts, embeddings))
            
            # Create the FAISS vector store
            self.vector_store = FAISS.from_embeddings(text_embeddings)
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
        
    def similarity_search(self, query_embedding, k=4):
        """
        Perform a similarity search on the vector store.

        Parameters:
        - query_embedding: The embedding vector representing the query
        - k: Number of top similar documents to retrieve (default is 4)

        Returns:
        - List of top k similar documents from the vector store
        """
        logger.debug(f"Performing similarity search with top {k} results")

        if not self.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized")

        # Use FAISS's built-in similarity search by vector
        return self.vector_store.similarity_search_by_vector(query_embedding, k=k)
