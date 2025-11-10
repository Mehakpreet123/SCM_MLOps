import pandas as pd

def create_lags(df, lags):
    """Create lag features for Total_Purchases column grouped by product hierarchy."""
    df = df.sort_values(['Country', 'State', 'products', 'year','month'])
    df['rolling_mean_3'] = (
    df.groupby(['Country', 'State', 'products','year','month'])['Total_Purchases']
    .transform(lambda x: x.shift(1).rolling(3).mean())
)
    for lag in lags:
        df[f'lag_{lag}'] = df['Total_Purchases'].shift(lag)
    df = df.dropna(subset=[f'lag_{l}' for l in lags])
    return df
