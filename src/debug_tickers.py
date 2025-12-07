import pandas as pd

def check_tickers():
    try:
        fund_df = pd.read_parquet("database/fundamentals.parquet")
        tech_df = pd.read_parquet("database/technicals.parquet")
        
        fund_tickers = sorted(fund_df['Ticker'].unique())
        tech_tickers = sorted(tech_df['Ticker'].unique())
        
        print(f"Fundamental Tickers ({len(fund_tickers)}):")
        print(fund_tickers[:10])
        print("...")
        
        print(f"\nTechnical Tickers ({len(tech_tickers)}):")
        print(tech_tickers[:10])
        print("...")
        
        common = set(fund_tickers).intersection(set(tech_tickers))
        print(f"\nCommon Tickers: {len(common)}")
        print(list(common)[:10])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_tickers()
