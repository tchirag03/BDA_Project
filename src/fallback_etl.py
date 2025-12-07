import pandas as pd
import numpy as np
import os
import glob
import difflib

def process_fundamentals(input_dir, output_path):
    print("Processing Fundamentals...")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Load Technical Tickers for matching
    tech_files = glob.glob(os.path.join(input_dir.replace("fundamentals", "technicals"), "*.csv"))
    tech_tickers = [os.path.basename(f).replace("_technicals.csv", "") for f in tech_files]
    
    # Simple normalization helper
    def normalize(s):
        return s.lower().replace(" ", "").replace(".", "").replace("&", "").replace("limited", "").replace("ltd", "")

    # Create mapping
    ticker_map = {}
    # 1. Direct match (normalized)
    for t in tech_tickers:
        ticker_map[normalize(t)] = t
        
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            
            # Get original company name from filename
            original_name = os.path.basename(filename).replace(".csv", "")
            
            # Attempt to map to Technical Ticker
            norm_name = normalize(original_name)
            
            # Heuristic matching
            matched_ticker = None
            
            # 1. Exact normalized match
            if norm_name in ticker_map:
                matched_ticker = ticker_map[norm_name]
            
            # Explicit Mapping for difficult cases
            explicit_map = {
                'ultratechcem': 'ULTRACEMCO',
                'varunbeverages': 'VBL',
                'unitedspirits': 'UNITDSPR',
                'vedanta': 'VEDL',
                'zyduslifesci': 'ZYDUSLIFE',
                'siemensenerind': 'SIEMENS', # Maybe? Or distinct.
                'jindalsteel': 'JINDALSTEL',
                'mahindra&mahindra': 'M&M',
                'm&m': 'M&M',
                'kotakmahbank': 'KOTAKBANK',
                'powergridcorp': 'POWERGRID',
                'bajajauto': 'BAJAJ-AUTO',
                'heromotocorp': 'HEROMOTOCO',
                'hindunilever': 'HINDUNILVR',
                'britanniainds': 'BRITANNIA',
                'godrejconsumer': 'GODREJCP',
                'shreecement': 'SHREECEM',
                'srft': 'SRF',
                'samvardhana': 'MOTHERSON',
                'motherson': 'MOTHERSON',
                'maxhealthcare': 'MAXHEALTH',
                'icicilombard': 'ICICIGI',
                'sbilifeinsur': 'SBILIFE',
                'hdfclifeinsur': 'HDFCLIFE',
                'avenuesuper': 'DMART',
                'sbi': 'SBIN',
                'infoedg(india)': 'NAUKRI',
                'bajajhousing': 'BAJAJHFL'
            }
            if not matched_ticker:
                if norm_name in explicit_map:
                    matched_ticker = explicit_map[norm_name]
            
            # 2. Starts with match (e.g. Adani Enterp -> ADANIENT)
            
            # 2. Starts with match (e.g. Adani Enterp -> ADANIENT)
            if not matched_ticker:
                for t in tech_tickers:
                    if normalize(t).startswith(norm_name) or norm_name.startswith(normalize(t)):
                        matched_ticker = t
                        break
            
            # 3. Contains match (e.g. Bajaj Finance -> BAJFINANCE)
            if not matched_ticker:
                 for t in tech_tickers:
                    if normalize(t) in norm_name or norm_name in normalize(t):
                        matched_ticker = t
                        break
            
            # 4. Fuzzy Match (difflib)
            if not matched_ticker:
                # Get close matches
                # We match normalized names
                normalized_tickers = [normalize(t) for t in tech_tickers]
                matches = difflib.get_close_matches(norm_name, normalized_tickers, n=1, cutoff=0.6)
                if matches:
                    # Find original ticker corresponding to matched normalized
                    best_match_norm = matches[0]
                    # Reverse lookup (inefficient but safe)
                    for t in tech_tickers:
                        if normalize(t) == best_match_norm:
                            matched_ticker = t
                            # print(f"Fuzzy Match: {original_name} -> {matched_ticker}")
                            break
            
            # Fallback: Use original if no match (will result in missing data in dashboard but preserves data)
            final_ticker = matched_ticker if matched_ticker else original_name
            
            # Normalize columns
            # Expected columns in cleaned CSV might vary slightly, so we map them.
            # Based on inspection: 'Company', 'Period', 'Net profit', 'Sales', etc.
            
            # Map known columns to standard names
            col_map = {
                'Company': 'Ticker_Old', # Rename old company col to avoid conflict
                'Net profit': 'NetProfit',
                'Operating Profit': 'OperatingProfit',
                'Equity Capital': 'EquityShareCapital',
                'Total Liabilities': 'Debt',
                'Total Assets': 'NetWorth', # Approximation
                'Sales': 'Sales',
                'Reserves': 'Reserves',
                'Borrowings': 'Borrowings',
                'Other Income': 'EBIT', # Approximation if EBIT missing, or calculate
                'EPS': 'EPS',
                'Book Value': 'BVPS',
                'ROE': 'ROE',
                'ROCE': 'ROCE',
                'No. of Equity Shares': 'Shares'
            }
            
            df = df.rename(columns=col_map)
            
            # Set Ticker
            df['Ticker'] = final_ticker
            
            # Create Date
            if 'Date' in df.columns:
                 # Check if date is in Mar-YY format
                 try:
                     df['Date'] = pd.to_datetime(df['Date'], errors='raise')
                 except:
                     try:
                         # Try Mar-16 format (%b-%y)
                         df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
                     except:
                         # Fallback to coaxing
                         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            elif 'Period' in df.columns:
                # Create Date from Period
                # Assuming Period 1 = 2024, 2 = 2023, etc.
                # We'll create a dummy date: YYYY-03-31
                current_year = 2025
                df['Year'] = current_year - df['Period']
                df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-03-31')
            else:
                # Fallback if Period is missing
                df['Date'] = pd.to_datetime('2024-03-31')

            # --- Derived Metrics Calculation ---
            
            # Fill Debt with Borrowings if Debt is missing
            if 'Debt' not in df.columns or df['Debt'].isnull().all():
                if 'Borrowings' in df.columns:
                    df['Debt'] = df['Borrowings']
            
            # fill 0 for calculation safety
            # Helper to safely get numeric series
            def get_numeric(col_name):
                if col_name in df.columns:
                    return pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                return 0.0

            df['EquityShareCapital'] = get_numeric('EquityShareCapital')
            df['Reserves'] = get_numeric('Reserves')
            df['Borrowings'] = get_numeric('Borrowings')
            df['Debt'] = get_numeric('Debt')
            df['NetProfit'] = get_numeric('NetProfit')
            df['OperatingProfit'] = get_numeric('OperatingProfit')
            
            pbt = get_numeric('Profit before tax')
            interest = get_numeric('Interest')

            # NetWorth = Equity + Reserves
            if 'NetWorth' not in df.columns or df['NetWorth'].isnull().all():
                df['NetWorth'] = df['EquityShareCapital'] + df['Reserves']
            
            # EBIT = PBT + Interest
            if 'EBIT' not in df.columns or df['EBIT'].isnull().all():
                df['EBIT'] = pbt + interest

            # Shares
            shares = get_numeric('Shares')
            # Avoid div by zero
            if isinstance(shares, pd.Series):
                 shares = shares.replace(0, np.nan)
            elif shares == 0:
                 shares = np.nan
            
            # EPS = (NetProfit * 1Cr) / Shares
            # NetProfit is in Crores, Shares is absolute.
            if 'EPS' not in df.columns or df['EPS'].isnull().all():
                df['EPS'] = (df['NetProfit'] * 10000000) / shares
                
            # BVPS = (NetWorth * 1Cr) / Shares
            if 'BVPS' not in df.columns or df['BVPS'].isnull().all():
                df['BVPS'] = (df['NetWorth'] * 10000000) / shares
            
            # ROE = (NetProfit / NetWorth) * 100
            nw = df['NetWorth'].replace(0, np.nan)
            if 'ROE' not in df.columns or df['ROE'].isnull().all():
                df['ROE'] = (df['NetProfit'] / nw) * 100
                
            # ROCE = (EBIT / (NetWorth + Debt)) * 100
            capital_employed = (df['NetWorth'] + df['Debt']).replace(0, np.nan)
            if 'ROCE' not in df.columns or df['ROCE'].isnull().all():
                df['ROCE'] = (df['EBIT'] / capital_employed) * 100

            # Select and fill missing standard columns
            required_cols = ["Ticker", "Date", "NetProfit", "Sales", "OperatingProfit", "EquityShareCapital", "Reserves", "Borrowings", "Debt", "NetWorth", "EBIT", "ROE", "ROCE", "EPS", "BVPS"]
            
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            dfs.append(df[required_cols])
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        # Ensure numeric types
        numeric_cols = ["NetProfit", "Sales", "OperatingProfit", "EquityShareCapital", "Reserves", "Borrowings", "Debt", "NetWorth", "EBIT", "ROE", "ROCE", "EPS", "BVPS"]
        for col in numeric_cols:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            
        final_df.to_parquet(output_path, index=False)
        print(f"Fundamentals saved to {output_path}")
        return final_df
    else:
        print("No fundamental data found.")
        return pd.DataFrame()

def process_technicals(input_dir, output_path):
    print("Processing Technicals...")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            
            # Rename Company to Ticker if needed
            if 'Company' in df.columns:
                df = df.rename(columns={'Company': 'Ticker'})
            
            # Ensure Date
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Calculate Daily Return
            df['DailyReturn'] = df['Close'].pct_change()
            
            # Calculate SMA200 if missing
            if 'SMA200' not in df.columns or df['SMA200'].isnull().all():
                df['SMA200'] = df['Close'].rolling(window=200).mean()
            
            # Ensure other SMAs
            if '20_SMA' in df.columns:
                df = df.rename(columns={'20_SMA': 'SMA20'})
            if '50_SMA' in df.columns:
                df = df.rename(columns={'50_SMA': 'SMA50'})
                
            if 'SMA20' not in df.columns:
                df['SMA20'] = df['Close'].rolling(window=20).mean()
            if 'SMA50' not in df.columns:
                df['SMA50'] = df['Close'].rolling(window=50).mean()

            # Add placeholder columns for ML
            df['Label'] = None
            df['Technical_ML_Score'] = None
            df['TrendScore'] = None
            df['TechnicalScore'] = None

            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_parquet(output_path, index=False)
        print(f"Technicals saved to {output_path}")
        return final_df
    else:
        print("No technical data found.")
        return pd.DataFrame()

def process_ratings(fund_df, tech_df, output_path):
    print("Calculating Ratings...")
    
    if fund_df.empty or tech_df.empty:
        print("Cannot calculate ratings: Missing data.")
        return

    # Get latest data for each ticker
    fund_latest = fund_df.sort_values('Date').groupby('Ticker').tail(1)
    tech_latest = tech_df.sort_values('Date').groupby('Ticker').tail(1)
    
    # Merge
    df = pd.merge(fund_latest, tech_latest, on='Ticker', suffixes=('_fund', '_tech'))
    
    # --- Scoring Logic (0-5 Scale) ---
    
    # 1. Balance Sheet Score (Debt/NetWorth)
    # Handle missing values
    df['Debt'] = df['Debt'].fillna(0)
    df['NetWorth'] = df['NetWorth'].replace(0, np.nan) # Avoid div by zero
    
    df['DE_Ratio'] = df['Debt'] / df['NetWorth']
    
    # Base Score 2.0
    df['BalanceSheetScore'] = 2.0
    
    # Additive bonuses to reach max 5
    # If DE < 2.0: +1.0 (Total 3.0)
    # If DE < 1.0: +1.0 (Total 4.0)
    # If DE < 0.5: +1.0 (Total 5.0)
    
    df['BalanceSheetScore'] += np.where(df['DE_Ratio'] < 2.0, 1.0, 0)
    df['BalanceSheetScore'] += np.where(df['DE_Ratio'] < 1.0, 1.0, 0)
    df['BalanceSheetScore'] += np.where(df['DE_Ratio'] < 0.5, 1.0, 0)
    
    # Clip to max 5
    df['BalanceSheetScore'] = df['BalanceSheetScore'].clip(upper=5.0)
    
    # 2. P&L Score (ROE)
    df['ROE'] = df['ROE'].fillna(0)
    
    # Map ROE to 0-5
    # > 25% -> 5
    # > 20% -> 4
    # > 15% -> 3
    # > 10% -> 2
    # > 5% -> 1
    # Else 0
    
    conditions_pl = [
        (df['ROE'] > 25),
        (df['ROE'] > 20),
        (df['ROE'] > 15),
        (df['ROE'] > 10),
        (df['ROE'] > 5)
    ]
    choices_pl = [5.0, 4.0, 3.0, 2.0, 1.0]
    df['PLScore'] = np.select(conditions_pl, choices_pl, default=0.0)
    
    # 3. Valuation Score (Placeholder)
    # Set to Neutral (2.5) for now as we don't have PE logic fully here yet
    # Or assuming user wants 5.0 based on request "higest will be 5"? 
    # Let's use 2.5 (Neutral) to avoid skewing high.
    df['ValuationScore'] = 2.5
    
    # 4. Technical Score
    # Price > SMA200 (+2), Price > SMA50 (+2), SMA50 > SMA200 (+1) -> Max 5
    df['Tech_Cond1'] = np.where(df['Close'] > df['SMA200'], 2.0, 0)
    df['Tech_Cond2'] = np.where(df['Close'] > df['SMA50'], 2.0, 0)
    df['Tech_Cond3'] = np.where(df['SMA50'] > df['SMA200'], 1.0, 0)
    
    df['TechnicalScore'] = df['Tech_Cond1'] + df['Tech_Cond2'] + df['Tech_Cond3']
    
    # 5. Overall Score (Weighted)
    # Fundamentals & Valuation: 30% each
    # Technicals: 10%
    # Weights: BS(0.3) + PL(0.3) + Val(0.3) + Tech(0.1) = 1.0
    
    df['OverallScore'] = (
        df['BalanceSheetScore'] * 0.2 + 
        df['PLScore'] * 0.4 + 
        df['ValuationScore'] * 0.4 + 
        df['TechnicalScore'] * 0.2
    )
    
    # Select columns
    ratings_df = df[['Ticker', 'Date_tech', 'BalanceSheetScore', 'PLScore', 'ValuationScore', 'TechnicalScore', 'OverallScore']]
    ratings_df = ratings_df.rename(columns={'Date_tech': 'Date'})
    
    ratings_df.to_parquet(output_path, index=False)
    print(f"Ratings saved to {output_path}")

def main():
    # Paths
    fund_input = "dataset/fundamentals_cleaned"
    tech_input = "dataset/technicals_cleaned"
    
    # Ensure database dir exists
    if not os.path.exists("database"):
        os.makedirs("database")
    
    fund_output = "database/fundamentals.parquet"
    tech_output = "database/technicals.parquet"
    ratings_output = "database/ratings.parquet"
    
    # Process
    fund_df = process_fundamentals(fund_input, fund_output)
    tech_df = process_technicals(tech_input, tech_output)
    process_ratings(fund_df, tech_df, ratings_output)

if __name__ == "__main__":
    main()
