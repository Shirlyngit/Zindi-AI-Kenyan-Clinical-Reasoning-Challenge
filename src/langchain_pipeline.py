# src/langchain_pipeline.py

import os
from typing import Optional

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Gemini imports
import google.generativeai as genai

# Local model imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LangChainSummarizer:
    """
    A wrapper that unifies Google Gemini API and local fine-tuned model
    into a LangChain-based summarization pipeline.
    """

    def __init__(self, mode: str = "gemini", model_name: Optional[str] = None):
        """
        Args:
            mode (str): "gemini" or "local".
            model_name (str): Local model name/path (HF Hub or local dir).
        """
        self.mode = mode.lower()
        self.model_name = model_name
        self.llm = None

        if self.mode == "gemini":
            self._setup_gemini()
        elif self.mode == "local":
            self._setup_local()
        else:
            raise ValueError("Mode must be 'gemini' or 'local'.")

        self.prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template=(
                "You are a clinical summarization assistant. "
                "Given the clinician's note, produce a concise, clear, and accurate summary.\n\n"
                "Clinician's Note:\n{prompt}\n\nSummary:"
            )
        )

    def _setup_gemini(self):
        """Configure Google Gemini API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in environment variables.")
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel("gemini-pro")

    def _setup_local(self):
        """Load local fine-tuned model."""
        model_path = self.model_name or os.getenv("LOCAL_MODEL_PATH", "distilgpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.3,
            do_sample=False
        )

    def run(self, input_text: str) -> str:
        """Generate a summary from the given input text."""
        # Format prompt
        final_prompt = self.prompt_template.format(prompt=input_text)

        if self.mode == "gemini":
            response = self.llm.generate_content(final_prompt)
            return response.text.strip()

        elif self.mode == "local":
            outputs = self.pipeline(final_prompt, num_return_sequences=1)
            return outputs[0]["generated_text"].replace(final_prompt, "").strip()

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


if __name__ == "__main__":
    # Quick debug test
    summarizer = LangChainSummarizer(mode=os.getenv("MODE", "local"))
    sample_note = "Patient reports persistent cough for 2 weeks, denies fever or chest pain."
    print(summarizer.run(sample_note))
