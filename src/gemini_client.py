import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import google.generativeai as genai
except Exception:
    genai = None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

if genai and API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        logger.warning("Could not configure google.generativeai: %s", e)

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or passed to GeminiClient")
        if genai is None:
            raise RuntimeError("google.generativeai package not installed. Install it or adapt this wrapper.")
        self.model_name = model_name or GEMINI_MODEL

    def call_generate(self, prompt: str, max_output_tokens: int = 512, temperature: float = 0.0) -> str:
        """Call Gemini to generate text. Adapt if SDK differs in your environment."""
        try:
            try:
                resp = genai.generate(model=self.model_name, prompt=prompt, max_output_tokens=max_output_tokens, temperature=temperature)
                if hasattr(resp, "text"):
                    return resp.text
                if hasattr(resp, "output"):
                    pieces = []
                    for o in getattr(resp, "output"):
                        if isinstance(o, dict) and "content" in o:
                            for c in o["content"]:
                                pieces.append(c.get("text", ""))
                        elif hasattr(o, "text"):
                            pieces.append(o.text)
                    return "\n".join(pieces)
                return str(resp)
            except Exception:
                resp = genai.responses.create(model=self.model_name, input=prompt)
                if hasattr(resp, "output"):
                    return getattr(resp, "output")[0].get("content")[0].get("text", "")
                if hasattr(resp, "answer"):
                    return resp.answer
                return str(resp)
        except Exception as e:
            logger.exception("Gemini generate failed: %s", e)
            raise

    def attempt_finetune(self, training_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Attempt to call a fine-tune endpoint on Gemini. Many accounts do not have this.
        Adapt to your account's SDK if available.
        """
        if not hasattr(genai, "fine_tunes") and not hasattr(genai, "fine_tune"):
            raise RuntimeError("Your google.generativeai client does not expose a fine-tune API. Use the RAG fallback.")
        try:
            if hasattr(genai, "fine_tunes"):
                job = genai.fine_tunes.create(training_file=training_file_path, model=self.model_name, **kwargs)
            else:
                job = genai.fine_tune.create(training_file=training_file_path, model=self.model_name, **kwargs)
            return {"job": job}
        except Exception as e:
            logger.exception("Fine-tune attempt failed: %s", e)
            raise
