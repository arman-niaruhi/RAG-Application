# tests/test_language_model.py
import pytest
from unittest.mock import patch, MagicMock
from core.language_model import LanguageModel

def test_generate_answer_success():
    model = LanguageModel(model_name="llama2")
    prompt = "What is AI?"

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "AI stands for Artificial Intelligence."
        mock_run.return_value = mock_result

        answer = model.generate_answer(prompt)
        assert "Artificial Intelligence" in answer

def test_generate_answer_fail():
    model = LanguageModel(model_name="llama2")
    prompt = "What is AI?"

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: model not found"
        mock_run.return_value = mock_result

        with pytest.raises(RuntimeError):
            model.generate_answer(prompt)
