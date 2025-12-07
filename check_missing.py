import pandas as pd
import glob
import os

def normalize(s):
    return s.lower().replace(" ", "").replace(".", "").replace("&", "").replace("limited", "").replace("ltd", "")

def check():
    print("Loading data...")
    try:
        tech = pd.read_parquet('database/technicals.parquet')
        ratings = pd.read_parquet('database/ratings.parquet')
        
        all_tickers = set(tech['Ticker'].unique())
        matched_tickers = set(ratings['Ticker'].unique())
        
        missing = all_tickers - matched_tickers
        
        with open("missing.log", "w", encoding="utf-8") as f:
            f.write(f"Total Technical Tickers: {len(all_tickers)}\n")
            f.write(f"Matched Tickers: {len(matched_tickers)}\n")
            f.write(f"Missing Tickers ({len(missing)}):\n")
            
            missing_list = sorted(list(missing))
            for t in missing_list:
                f.write(f" - {t}\n")
                
            f.write("\nAvailable Fundamental Files (cleaned):\n")
            fund_files = glob.glob("dataset/fundamentals_cleaned/*.csv")
            fund_names = [os.path.basename(file).replace(".csv", "") for file in fund_files]
            
            f.write("\n--- Potential Candidates for Missing ---\n")
            for m in missing_list:
                norm_m = normalize(m)
                candidates = []
                for file_name in fund_names:
                    norm_f = normalize(file_name)
                    if norm_m in norm_f or norm_f in norm_m:
                        candidates.append(file_name)
                    # fuzzy?
                    import difflib
                    if difflib.SequenceMatcher(None, norm_m, norm_f).ratio() > 0.6:
                         candidates.append(f"{file_name} (Fuzzy)")
                
                if candidates:
                    f.write(f"{m} ?= {candidates}\n")
    except Exception as e:
        print(e)
                
    except Exception as e:
        print(e)
        
if __name__ == "__main__":
    check()
