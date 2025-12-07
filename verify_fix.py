import pandas as pd

def verify():
    try:
        ratings = pd.read_parquet('database/ratings.parquet')
        tech = pd.read_parquet('database/technicals.parquet')
        
        target = 'ADANIENSOL'
        
        print(f"--- Checking {target} ---")
        r_rows = ratings[ratings['Ticker'] == target]
        if not r_rows.empty:
            print("Ratings Found:")
            print(r_rows.iloc[-1])
        else:
            print("Ratings MISSING.")
            
        t_rows = tech[tech['Ticker'] == target]
        if not t_rows.empty:
            print("\nTechnicals Found (Latest):")
            print(t_rows[['Date', 'Label', 'Technical_ML_Score']].iloc[-1])
        else:
            print("Technicals MISSING.")
            
    except Exception as e:
        print(e)

if __name__ == "__main__":
    verify()
