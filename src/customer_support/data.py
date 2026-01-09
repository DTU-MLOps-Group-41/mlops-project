from pathlib import Path
import os
import typer
import pandas as pd
import kagglehub
from transformers import DistilBertTokenizer
from datasets import Dataset

# Define consistent paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/preprocessed")


class TicketData:
    """
    Handles downloading and loading raw data.
    """

    def __init__(self, dataset_type: str = "small", model_name: str = "distilbert-base-multilingual-cased") -> None:
        self.dataset_type = dataset_type

        # 1. Download Data (Cached by KaggleHub)
        print(f"[{dataset_type}] Checking dataset via KaggleHub...")
        self.data_path = kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets")

        # 2. Load the DataFrame
        self.data = self.get_dataset_file()

        # 3. Initialize Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def get_dataset_file(self) -> pd.DataFrame:
        """Load the specific CSV file based on dataset_type."""
        files = {
            "full": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
            "medium": "dataset-tickets-multi-lang-4-20k.csv",
            "small": "dataset-tickets-multi-lang3-4k.csv",
        }

        if self.dataset_type not in files:
            raise ValueError(f"dataset_type must be one of {list(files.keys())}")

        target_filename = files[self.dataset_type]
        full_path = os.path.join(self.data_path, target_filename)

        print(f"[{self.dataset_type}] Loading CSV from: {target_filename}")
        return pd.read_csv(full_path)


def process_single_dataset(dataset_type: str) -> None:
    print(f"\n=== Processing: {dataset_type.upper()} ===")

    loader = TicketData(dataset_type=dataset_type)
    df = loader.data
    tokenizer = loader.tokenizer

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_save_path = RAW_DATA_DIR / f"{dataset_type}.csv"
    df.to_csv(raw_save_path, index=False)
    print(f"Saved RAW data to: {raw_save_path}")

    # Select ticket body and priority (for now, we can add more later)
    data = df[["body", "priority"]].copy()

    data = data.dropna(subset=["body", "priority"])  # Drop missing text/labels
    data["body"] = data["body"].astype(str)  # Force string format

    label_map = {"low": 0, "medium": 1, "high": 2}
    data["labels"] = data["priority"].str.lower().map(label_map)

    # Remove rows where label mapping failed
    data = data.dropna(subset=["labels"])
    data["labels"] = data["labels"].astype(int)

    print("Tokenizing...")
    hf_dataset = Dataset.from_pandas(data[["body", "labels"]])

    def tokenize_function(examples):
        return tokenizer(examples["body"], truncation=True, padding=False)

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    cols_to_remove = ["body"]
    if "__index_level_0__" in tokenized_dataset.column_names:
        cols_to_remove.append("__index_level_0__")

    tokenized_dataset = tokenized_dataset.remove_columns(cols_to_remove)
    tokenized_dataset.set_format("torch")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_save_path = PROCESSED_DATA_DIR / dataset_type

    tokenized_dataset.save_to_disk(processed_save_path)
    print(f"Saved PROCESSED dataset to: {processed_save_path}")
    print(f"Final Rows: {len(tokenized_dataset)}")


def main() -> None:
    datasets_to_process = ["small", "medium", "full"]
    for dtype in datasets_to_process:
        try:
            process_single_dataset(dtype)
        except Exception as e:
            print(f"Error processing {dtype}: {e}")


if __name__ == "__main__":
    typer.run(main)
