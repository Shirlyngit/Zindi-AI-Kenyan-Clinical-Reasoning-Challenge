import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import google.generativeai as genai

from data_loader import load_clinical_data

def fine_tune_local(data_path, model_dir="models/t5_finetuned"):
    print("[INFO] Starting local fine-tuning with Hugging Face...")
    
    # Load data
    df = load_clinical_data(data_path)

    # Hugging Face dataset
    dataset = Dataset.from_pandas(df.rename(columns={"Prompt": "input_text", "Clinician": "target_text"}))

    # Tokenizer & model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def preprocess(batch):
        inputs = ["summarize: " + text for text in batch["input_text"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        labels = tokenizer(batch["target_text"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

    # Training setup
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="no",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"[INFO] Local fine-tuned model saved to {model_dir}")

def fine_tune_gemini(data_path):
    print("[INFO] Simulating Gemini fine-tuning (prompt adaptation)...")

    # Configure API
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Load CSV
    df = load_clinical_data(data_path)

    # Loop through small subset for testing
    for i, row in df.head(5).iterrows():
        prompt = row["Prompt"]
        target = row["Clinician"]

        response = model.generate_content(f"Given the clinical note:\n{prompt}\nGenerate a medical summary similar to:\n{target}")
        print(f"[SAMPLE {i}]")
        print("Generated:", response.text.strip())
        print("Target:", target)
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune summarization model.")
    parser.add_argument("--mode", choices=["local", "gemini"], required=True, help="Fine-tuning mode.")
    parser.add_argument("--data", default="data/clinical_prompt.csv", help="Path to dataset.")
    args = parser.parse_args()

    if args.mode == "local":
        fine_tune_local(args.data)
    elif args.mode == "gemini":
        fine_tune_gemini(args.data)
