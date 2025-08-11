import pytest
from unittest.mock import patch, MagicMock
from src.summarizer import Summarizer

@pytest.fixture
def sample_prompt():
    return "The patient was admitted with chest pain and shortness of breath."

@pytest.fixture
def sample_summary():
    return "Patient admitted with chest pain and shortness of breath."

def test_local_mode_summary(sample_prompt, sample_summary):
    summarizer = Summarizer(mode="local")
    summarizer.model = MagicMock()
    summarizer.model.predict.return_value = sample_summary

    result = summarizer.summarize(sample_prompt)

    assert isinstance(result, str)
    assert result.strip() != ""
    assert sample_summary in result

@patch("src.summarizer.genai.GenerativeModel")
def test_gemini_mode_summary(mock_gemini_model, sample_prompt, sample_summary):
    mock_instance = MagicMock()
    mock_instance.generate_content.return_value = MagicMock(
        text=sample_summary
    )
    mock_gemini_model.return_value = mock_instance

    summarizer = Summarizer(mode="gemini")
    result = summarizer.summarize(sample_prompt)

    assert isinstance(result, str)
    assert result.strip() != ""
    assert sample_summary in result

@patch("src.summarizer.genai.GenerativeModel")
def test_gemini_mode_failure(mock_gemini_model, sample_prompt):
    mock_instance = MagicMock()
    mock_instance.generate_content.side_effect = Exception("Gemini API Error")
    mock_gemini_model.return_value = mock_instance

    summarizer = Summarizer(mode="gemini")

    with pytest.raises(Exception) as exc_info:
        summarizer.summarize(sample_prompt)

    assert "Gemini API Error" in str(exc_info.value)
