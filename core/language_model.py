import logging
import requests

logger = logging.getLogger(__name__)

class LanguageModel:
    def __init__(self, model_name="llama3:latest"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate_answer(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")
