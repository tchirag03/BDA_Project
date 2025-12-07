import difflib

def normalize(s):
    return s.lower().replace(" ", "").replace(".", "").replace("&", "").replace("limited", "").replace("ltd", "")

tech_tickers = ["ULTRACEMCO", "VBL", "UNITDSPR", "VEDL", "ZYDUSLIFE", "TVSMOTOR", "TMPV"]
fund_names = ["UltraTech Cem", "Varun Beverages", "United Spirits", "Vedanta", "Zydus Lifesci", "TVS Motor Co"]

print("--- Debugging Matching ---")

for original_name in fund_names:
    norm_name = normalize(original_name)
    print(f"\nProcessing: {original_name} -> {norm_name}")
    
    matched_ticker = None
    
    # 1. Exact
    # (Skipped for test)
    
    # 2. Starts with
    if not matched_ticker:
        for t in tech_tickers:
            nt = normalize(t)
            if nt.startswith(norm_name) or norm_name.startswith(nt):
                matched_ticker = t
                print(f"MATCH (StartsWith): {t} ({nt})")
                break
    
    # 3. Contains
    if not matched_ticker:
            for t in tech_tickers:
                nt = normalize(t)
                if nt in norm_name or norm_name in nt:
                    matched_ticker = t
                    print(f"MATCH (Contains): {t} ({nt})")
                    break
    
    # 4. Fuzzy
    if not matched_ticker:
        normalized_tickers = [normalize(t) for t in tech_tickers]
        matches = difflib.get_close_matches(norm_name, normalized_tickers, n=1, cutoff=0.6)
        if matches:
            best = matches[0]
            print(f"MATCH (Fuzzy): {best} (Ratio: {difflib.SequenceMatcher(None, norm_name, best).ratio():.2f})")
            # find original
        else:
            print("NO MATCH")
