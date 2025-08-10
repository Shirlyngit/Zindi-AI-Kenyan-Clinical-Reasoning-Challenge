import os
import google.generativeai as genai

class GeminiClient:
def init(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
self.api_key = api_key or os.getenv("GEMINI_API_KEY")
if not self.api_key:
raise ValueError("Gemini API key not found. Set GEMINI_API_KEY env variable or pass api_key explicitly.")
genai.configure(api_key=self.api_key)
self.model = genai.GenerativeModel(model_name)

def generate_summary(self, clinical_prompt: str) -> str:
    try:
        response = self.model.generate_content(clinical_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

if __name__ == "__main__":
  client = GeminiClient()
  prompt = "Patient presents with fever, cough, and sore throat."
  summary = client.generate_summary(prompt)
  print("Generated Medical Summary:\n", summary)
