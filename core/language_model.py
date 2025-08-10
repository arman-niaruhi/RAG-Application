import requests
import json

class LanguageModel:
    def __init__(self, model_name="llama3:latest"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def stream_answer(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }
        try:
            with requests.post(self.api_url, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                        except Exception:
                            continue
        except Exception as e:
            raise RuntimeError(f"Ollama API streaming error: {e}")
