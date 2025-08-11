# so the rest of the system can call one simple API irregardless of whether it’s running Google Gemini or local fine-tuned mode.

import os
from typing import Optional
from src.langchain_pipeline import LangChainSummarizer


class SummarizerService:
    """
    Abstraction layer for summarization so that
    the app, FastAPI routes, CLI tools, or tests
    can all use the same interface.
    """

    def __init__(self, mode: Optional[str] = None, model_name: Optional[str] = None):
        """
        Args:
            mode (str): "gemini" or "local". Defaults to env MODE or 'local'.
            model_name (str): Local model path or HF Hub ID if in local mode.
        """
        self.mode = (mode or os.getenv("MODE", "local")).lower()
        self.model_name = model_name or os.getenv("LOCAL_MODEL_PATH", "distilgpt2")
        self.pipeline = LangChainSummarizer(mode=self.mode, model_name=self.model_name)

    def summarize(self, text: str) -> str:
        """Summarize the given text."""
        if not text or not text.strip():
            return "Error: Empty input text."
        return self.pipeline.run(text)

    def batch_summarize(self, texts: list[str]) -> list[str]:
        """Summarize multiple texts in a batch."""
        results = []
        for txt in texts:
            results.append(self.summarize(txt))
        return results


if __name__ == "__main__":
    # Quick manual test
    mode = os.getenv("MODE", "local")
    service = SummarizerService(mode=mode)
    sample_note = "Patient is a 55-year-old male with hypertension, presenting with dizziness."
    print(f"Mode: {mode}")
    print("Summary:", service.summarize(sample_note))
