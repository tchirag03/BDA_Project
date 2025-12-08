import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os

def main():
    print("Initializing SparkML (Sklearn Fallback)...")
    
    # 1. Load Data
    input_path = "dataset/spark_processed/processed_data.parquet"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run spark_etl.py first.")
        return
        
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows.")
    
    # 2. Prepare Features
    features = ["Open", "High", "Low", "Volume", "Age", "Risk_Score", "Bullish_MA"]
    # We only have columns from ETL. 
    # Available: Close, Open, High, Low, 50_SMA, 200_SMA, Target, Bullish_MA, Risk_Score, Volatility
    
    feature_cols = ["Open", "High", "Low", "Volume", "Risk_Score", "Bullish_MA", "Volatility"]
    target_col = "Target"
    
    # Check if columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing}. Using available features.")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]
    
    # 3. Split
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. Train Model (Random Forest)
    print(f"Training RandomForestClassifier on {len(X_train)} samples...")
    # Limiting depth for speed and interpretability
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # 5. Predictions & Evaluation
    print("Evaluating Model...")
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5 # Single class edge case

    print("\n--- Model Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importances.head(5))

    # Save Predictions for Dashboard (Optional)
    # merged back to test set
    test_results = X_test.copy()
    test_results['Target'] = y_test
    test_results['Prediction'] = y_pred
    test_results['Probability'] = y_prob
    test_results['Ticker'] = df.loc[X_test.index, 'Ticker'] # Map back ticker using index
    
    output_pred = "dataset/spark_processed/predictions.parquet"
    test_results.to_parquet(output_pred)
    print(f"\nPredictions saved to {output_pred}")

if __name__ == "__main__":
    main()
