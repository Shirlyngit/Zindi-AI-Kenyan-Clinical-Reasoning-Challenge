# src/data_loader.py

import pandas as pd
from pathlib import Path

class ClinicalDataLoader:
    """
    Loads and preprocesses clinical prompt dataset.
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path("clinical_prompts.csv")

    def load_data(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Ensure expected columns exist
        expected_cols = {"Prompt", "Clinician"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(
                f"CSV missing required columns. Found {df.columns}, expected {expected_cols}"
            )

        # Rename for consistency inside code
        df = df.rename(columns={"Prompt": "prompt", "Clinician": "gold_summary"})

        # Drop rows with missing values
        df = df.dropna(subset=["prompt", "gold_summary"])

        return df

if __name__ == "__main__":
    loader = ClinicalDataLoader("data/clinical_prompts.csv")
    data = loader.load_data()
    print(data.head())
