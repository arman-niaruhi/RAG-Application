import logging
import subprocess

logger = logging.getLogger(__name__)

class LanguageModel:
    def __init__(self, model_name="tinyllama:latest"):
        self.model_name = model_name

    def generate_answer(self, prompt):
        cmd = ["ollama", "run", self.model_name, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Ollama error: {result.stderr}")





