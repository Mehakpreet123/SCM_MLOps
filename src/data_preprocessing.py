import argparse
import pandas as pd
import os
from features import create_lags

def preprocess(input_dir: str, output_file: str):
    """Aggregate monthly purchases and create lag features."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Automatically pick the first CSV file in the folder
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    data_file = os.path.join(input_dir, csv_files[0])

    df = pd.read_csv(data_file)
    df.drop(columns=['Customer_ID', 'Name', 'Email', 'Phone', 'Address',
    'Zipcode','Age', 'Gender', 'Income','Year', 'Month','Time'],inplace=True)

    df['Date']=pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)

    df = df.dropna(subset=['Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date').reset_index(drop=True)
    # 1-03-2023 to 29-02-2024
    df['year']=df['Date'].dt.year
    df['month']=df['Date'].dt.month
    df['day']=df['Date'].dt.day


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
