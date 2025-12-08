import pandas as pd
import glob
import os
import re

def main():
    print("Initializing Pandas ETL (Spark Fallback)...")
    
    # 1. Ingest Technicals
    path = "dataset/technicals_csv/*.csv"
    files = glob.glob(path)
    print(f"Found {len(files)} technical files.")
    
    dfs = []
    for f in files:
        try:
            # Extract Ticker
            ticker = re.search(r"\\([^\\]+)_technicals\.csv$", f).group(1) 
            df = pd.read_csv(f)
            df['Ticker'] = ticker
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        print("No data loaded.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # 2. Feature Engineering
    # Rename 'Price' to 'Date' if it exists (Common in Yahoo finance downloads)
    if 'Price' in full_df.columns:
        full_df.rename(columns={'Price': 'Date'}, inplace=True)

    # Ensure numeric
    cols = ["Close", "Open", "High", "Low", "50_SMA", "200_SMA"]
    for c in cols:
        if c in full_df.columns:
            full_df[c] = pd.to_numeric(full_df[c], errors='coerce')

    # Sort for Window ops
    full_df['Date'] = pd.to_datetime(full_df['Date'], errors='coerce')
    full_df = full_df.dropna(subset=['Date']) # Drop rows where Date parse failed (header garbage)
    full_df = full_df.sort_values(['Ticker', 'Date'])
    
    # Target: Next Close > Close
    full_df['NextClose'] = full_df.groupby('Ticker')['Close'].shift(-1)
    full_df['Target'] = (full_df['NextClose'] > full_df['Close']).astype(int)
    
    # Bullish MA
    if "50_SMA" in full_df.columns and "200_SMA" in full_df.columns:
        full_df['Bullish_MA'] = (full_df['50_SMA'] > full_df['200_SMA']).astype(int)
    else:
        full_df['Bullish_MA'] = 0

    # Risk Score (Volatility -> (High-Low)/Open) * 100
    full_df['Volatility'] = (full_df['High'] - full_df['Low']) / full_df['Open']
    full_df['Risk_Score'] = full_df['Volatility'] * 100

    # 3. Save
    output_dir = "dataset/spark_processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_data.parquet")
    
    # Drop NaNs for ML
    final_df = full_df.dropna(subset=['Target', 'Risk_Score', 'Bullish_MA'])
    
    print(f"Processed {len(final_df)} rows.")
    print(final_df[['Ticker', 'Date', 'Target', 'Risk_Score']].head())
    
    final_df.to_parquet(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
