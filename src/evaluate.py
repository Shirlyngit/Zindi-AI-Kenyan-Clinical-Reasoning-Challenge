# src/evaluate.py

import os
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from rouge_score import rouge_scorer
from src.summarizer import SummarizerService
from src.data_loader import load_csv_data

def evaluate_model(
    csv_path: str,
    mode: str = "local",
    model_name: str = "distilgpt2",
    sample_size: int = 50
):
    """
    Evaluate model-generated summaries against clinician-written summaries.

    Args:
        csv_path (str): Path to dataset CSV
        mode (str): "local" or "gemini"
        model_name (str): Hugging Face model name or Gemini model ID
        sample_size (int): Number of samples to evaluate
    """
    # Load dataset
    df = load_csv_data(csv_path)

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    # Initialize summarizer
    summarizer = SummarizerService(mode=mode, model_name=model_name)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = []

    print(f"\nEvaluating {sample_size} samples in {mode.upper()} mode...\n")

    for i, row in df.iterrows():
        prompt = row["Prompt"]
        gold_summary = row["Clinician"]

        generated_summary = summarizer.summarize(prompt)

        # Compute ROUGE scores
        rouge_scores = rouge.score(gold_summary, generated_summary)

        results.append({
            "prompt": prompt,
            "gold_summary": gold_summary,
            "generated_summary": generated_summary,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure
        })

        print(f"Sample {len(results)}/{sample_size} — ROUGE-L: {rouge_scores['rougeL'].fmeasure:.3f}")

    # Save results to CSV
    out_path = "evaluation_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n✅ Evaluation complete. Results saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate medical summarization model.")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--mode", choices=["local", "gemini"], default="local")
    parser.add_argument("--model", default=os.getenv("LOCAL_MODEL_PATH", "distilgpt2"))
    parser.add_argument("--samples", type=int, default=50)

    args = parser.parse_args()

    evaluate_model(
        csv_path=args.csv,
        mode=args.mode,
        model_name=args.model,
        sample_size=args.samples
    )
