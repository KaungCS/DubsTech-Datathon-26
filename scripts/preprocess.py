import re
from pathlib import Path

import pandas as pd


def is_valid_id(id_val):
    return bool(re.match(r"^\d+_\d+$", str(id_val)))


def preprocess_data(input_path=None, output_path=None):
    script_dir = Path(__file__).resolve().parent
    if input_path is None:
        input_path = script_dir / ".." / "data" / "raw" / "Access_to_Tech_Dataset.csv"
    if output_path is None:
        output_path = script_dir / ".." / "data" / "processed" / "cleaned_dataset.csv"

    df = pd.read_csv(input_path)
    clean_df = df[df["id"].apply(is_valid_id)]
    clean_df = clean_df[clean_df["scrape_status"] == "scraped"]
    clean_df.to_csv(output_path, index=False)

    print(f"Removed {len(df) - len(clean_df)} junk rows.")
    print(len(clean_df))
    return output_path


if __name__ == "__main__":
    preprocess_data()