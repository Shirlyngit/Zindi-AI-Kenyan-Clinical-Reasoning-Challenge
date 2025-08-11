import json
from pathlib import Path
from data_loader import ClinicalDataLoader

class FinetuneDatasetPreparer:
    """
    Converts the clinical prompt dataset into a JSONL format
    compatible with Gemini fine-tuning.
    """

    def __init__(self, input_csv: str, output_dir: str = "data/processed"):
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        loader = ClinicalDataLoader(self.input_csv)
        df = loader.load_data()

        # Create training JSONL
        train_path = self.output_dir / "train.jsonl"
        with open(train_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                record = {
                    "input_text": row["prompt"],
                    "output_text": row["gold_summary"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ Training dataset saved at: {train_path}")

if __name__ == "__main__":
    preparer = FinetuneDatasetPreparer("data/clinical_prompts.csv")
    preparer.prepare()
