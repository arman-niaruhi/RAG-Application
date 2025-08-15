from openai import OpenAI


class LanguageModel:
    def __init__(self, model_name="tinyllama:latest"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required but unused
        )

    def stream_answer(self, prompt):
        """Query Ollama model with the given prompt"""
        try:
            response = self.ollama_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying Ollama: {str(e)}")
            raise
