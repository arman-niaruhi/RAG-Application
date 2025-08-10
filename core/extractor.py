import logging
from abc import ABC, abstractmethod
import pdfplumber
logger = logging.getLogger(__name__)

class PDFExtractor(ABC):
    """
    Abstract base class for PDF text extractors.
    Defines the interface for extracting text from a PDF file.
    """
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract all text content from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file.
        
        Returns:
            str: Extracted text.
        """
        pass

class PdfPlumberExtractor:
    def extract_text(self, pdf_path: str) -> str:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise
