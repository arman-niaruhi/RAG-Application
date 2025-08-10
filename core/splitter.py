import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)
class TextSplitter(ABC):
    """
    Abstract base class for splitting large text into smaller chunks.
    """
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split the input text into a list of smaller text chunks.
        
        Args:
            text (str): Large text input.
        
        Returns:
            List[str]: List of text chunks.
        """
        pass


class RecursiveSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        logger.info(f"Initializing RecursiveSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str):
        logger.debug(f"Splitting text of length {len(text)}")
        docs = [Document(page_content=text)]
        chunks = self.splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
