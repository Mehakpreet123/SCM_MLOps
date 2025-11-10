import argparse
import os
import pandas as pd

def ingest_data(input_path: str, output_dir: str):
    """Reads the raw data and saves it into standardized format."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    output_path = os.path.join(output_dir, "raw_data.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Data ingested and saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="data/raw/input.csv")
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()
    ingest_data(args.in_path, args.out_dir)
