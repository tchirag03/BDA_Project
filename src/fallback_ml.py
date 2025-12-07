import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def train_and_predict():
    print("Starting ML Pipeline...")
    
    # Load Data
    try:
        fund_df = pd.read_parquet("database/fundamentals.parquet")
        tech_df = pd.read_parquet("database/technicals.parquet")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Merge Data
    # We need to merge fundamentals to technicals based on Ticker and Date.
    # Fundamentals are annual/quarterly, Technicals are daily.
    # We will forward fill fundamentals.
    
    print("Merging Data...")
    tech_df['Date'] = pd.to_datetime(tech_df['Date'])
    fund_df['Date'] = pd.to_datetime(fund_df['Date'])
    
    tech_df = tech_df.sort_values('Date')
    fund_df = fund_df.sort_values('Date')
    
    # Use merge_asof for point-in-time correctness (backward search)
    # tech_df is left, fund_df is right.
    merged_df = pd.merge_asof(tech_df, fund_df, on='Date', by='Ticker', direction='backward')
    
    # Feature Engineering
    print("Feature Engineering...")
    
    # Ratios
    # PE = Close / EPS
    merged_df['PE'] = merged_df['Close'] / merged_df['EPS']
    # PB = Close / BVPS
    merged_df['PB'] = merged_df['Close'] / merged_df['BVPS']
    
    # Technical Features
    merged_df['SMA50_200_Ratio'] = merged_df['SMA50'] / merged_df['SMA200']
    
    # Handle missing/infinite
    features = ['PE', 'PB', 'ROE', 'ROCE', 'SMA50_200_Ratio', 'DailyReturn']
    
    # Fill missing with median or 0
    for col in features:
        merged_df[col] = merged_df[col].replace([np.inf, -np.inf], np.nan)
        merged_df[col] = merged_df[col].fillna(0) # Simple imputation
        
    # Label Generation (Future 60-day return)
    # Calculate 60-day future return
    # Shift(-60) gets the price 60 days in future
    merged_df['FutureClose'] = merged_df.groupby('Ticker')['Close'].shift(-60)
    merged_df['FutureReturn'] = (merged_df['FutureClose'] - merged_df['Close']) / merged_df['Close']
    
    # Define Labels
    # > 10% = Strong (2)
    # < -10% = Weak (0)
    # Else = Neutral (1)
    
    conditions = [
        (merged_df['FutureReturn'] > 0.10),
        (merged_df['FutureReturn'] < -0.10)
    ]
    choices = [2, 0]
    merged_df['Target'] = np.select(conditions, choices, default=1)
    
    # Drop rows where FutureReturn is NaN (last 60 days) for training
    train_df = merged_df.dropna(subset=['FutureReturn'])
    
    if train_df.empty:
        print("Not enough data for training.")
        return

    X = train_df[features]
    y = train_df['Target']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Prediction on Full Dataset (including latest data)
    print("Generating Predictions...")
    full_X = merged_df[features]
    merged_df['Predicted_Label'] = clf.predict(full_X)
    
    # Probabilities for "Strong" (Class 2)
    # Classes are [0, 1, 2] usually, check clf.classes_
    strong_idx = np.where(clf.classes_ == 2)[0][0]
    probs = clf.predict_proba(full_X)
    merged_df['Strong_Prob'] = probs[:, strong_idx]
    
    # Convert to Technical_ML_Score (0-10)
    merged_df['Technical_ML_Score'] = merged_df['Strong_Prob'] * 10
    
    # Map Label to String
    label_map = {0: 'Weak', 1: 'Neutral', 2: 'Strong'}
    merged_df['Label'] = merged_df['Predicted_Label'].map(label_map)
    
    # Update Technicals Database
    # We only need to update the technicals.parquet with new columns
    # We should be careful to preserve the original structure or just overwrite with new columns
    
    # Select columns to save back to technicals
    # Original: Ticker, Date, Open, High, Low, Close, Volume, DailyReturn, SMA20, SMA50, SMA200
    # New: Label, Technical_ML_Score
    
    output_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'DailyReturn', 
                   'SMA20', 'SMA50', 'SMA200', 'Label', 'Technical_ML_Score']
    
    # Ensure columns exist
    for col in output_cols:
        if col not in merged_df.columns:
            merged_df[col] = None
            
    final_tech_df = merged_df[output_cols]
    
    # Save
    output_path = "database/technicals.parquet"
    final_tech_df.to_parquet(output_path, index=False)
    print(f"Updated technicals with ML predictions saved to {output_path}")

if __name__ == "__main__":
    train_and_predict()
