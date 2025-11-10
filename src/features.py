# src/create_lags.py
import argparse
import pandas as pd
import os

def create_lags(df, lags):
    """Create lag features for Total_Purchases column grouped by product hierarchy."""
    df = df.sort_values(['Country', 'State', 'products', 'year', 'month'])
    
    # Optional rolling mean
    df['rolling_mean_3'] = (
        df.groupby(['Country', 'State', 'products'])['Total_Purchases']
        .transform(lambda x: x.shift(1).rolling(3).mean())
    )
    
    # Create lag features
    for lag in lags:
        df[f'lag_{lag}'] = df['Total_Purchases'].shift(lag)
    
    # Drop rows with NaN values introduced by lags
    df = df.dropna(subset=[f'lag_{l}' for l in lags])
    return df

def main(input_file, output_file, lags):
    """Load CSV, create lag features, and save to output CSV."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df = pd.read_csv(input_file)
    df_lags = create_lags(df, lags)
    df_lags.to_csv(output_file, index=False)
    
    print(f"✅ Lag features created and saved to {output_file}")
    print(f"Data shape: {df_lags.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create lag features for Total_Purchases")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save CSV with lag features")
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3], help="List of lag periods to create")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.lags)
