import argparse
import pandas as pd
import os
from features import create_lags

def preprocess(input_dir: str, output_file: str):
    """Aggregate monthly purchases and create lag features."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.read_csv(os.path.join(input_dir, "raw_data.csv"))

    # Aggregate and sort
    monthly_data = (
        df.groupby(['Country', 'State', 'products', 'year', 'month'])['Total_Purchases']
        .sum()
        .reset_index()
        .sort_values(by=['Country', 'State', 'products', 'year', 'month'])
    )


    monthly_data.to_csv(output_file, index=False)
    print(f"✅ Processed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/raw")
    parser.add_argument("--out_file", default="data/processed/monthly_data.csv")
    args = parser.parse_args()
    preprocess(args.in_dir, args.out_file)
