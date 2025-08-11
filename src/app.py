import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.summarizer import SummarizerService

# Read defaults from environment
MODE = os.getenv("MODE", "local")
MODEL_NAME = os.getenv("LOCAL_MODEL_PATH", "distilgpt2")

# Initialize the service once at startup
summarizer_service = SummarizerService(mode=MODE, model_name=MODEL_NAME)

app = FastAPI(
    title="Medical Summarization API",
    description="REST API for generating medical summaries using Gemini API or a locally fine-tuned model.",
    version="1.0.0",
)


class SummarizeRequest(BaseModel):
    text: str


class BatchSummarizeRequest(BaseModel):
    texts: List[str]


class SummarizeResponse(BaseModel):
    summary: str


class BatchSummarizeResponse(BaseModel):
    summaries: List[str]


@app.get("/")
def root():
    return {"message": "Medical Summarization API is running", "mode": MODE}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_text(request: SummarizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    summary = summarizer_service.summarize(request.text)
    return SummarizeResponse(summary=summary)


@app.post("/summarize/batch", response_model=BatchSummarizeResponse)
def summarize_batch(request: BatchSummarizeRequest):
    if not request.texts or not all(t.strip() for t in request.texts):
        raise HTTPException(status_code=400, detail="All texts must be non-empty")
    summaries = summarizer_service.batch_summarize(request.texts)
    return BatchSummarizeResponse(summaries=summaries)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
