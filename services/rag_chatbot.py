import logging
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, extractor, splitter, embedder, vector_store, language_model):
        logger.info("Initializing RAG Chatbot")
        self.extractor = extractor
        self.splitter = splitter
        self.embedder = embedder
        self.vector_store = vector_store
        self.language_model = language_model

    def ingest_pdf(self, pdf_path: str):
        logger.info(f"Ingesting PDF: {pdf_path}")
        text = self.extractor.extract_text(pdf_path)
        chunks = self.splitter.split_text(text)
        embeddings = self.embedder.embed_documents([chunk.page_content for chunk in chunks])
        self.vector_store.create_vector_store(embeddings, chunks)
        logger.info("PDF ingested and vector store ready")

    def retrieve_context(self, question: str, top_k: int = 5) -> str:
        """Retrieve relevant chunks from the vector store."""
        query_embedding = self.embedder.embed_documents([question])[0]
        docs = self.vector_store.similarity_search(query_embedding, k=top_k)
        context = "\n".join([doc.page_content for doc in docs])
        return context

    async def generate_answer_from_context(self, context, question):
        prompt = f"{context}\nQuestion: {question}"
        return await self.language_model.generate_answer(prompt)

    async def stream_answer_from_context(self, context, question):
        prompt = f"{context}\nQuestion: {question}"
        for chunk in self.language_model.stream_answer(prompt):
            yield chunk

    async def stream_question_answer(self, question: str):
        context = self.retrieve_context(question)
        async for chunk in self.stream_answer_from_context(context, question):
            yield chunk

