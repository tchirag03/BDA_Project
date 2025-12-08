import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(df, lookback=60):
    # Sort by Date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Feature Engineering
    # EMA 50, EMA 200
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Future Return (Target)
    # Predict return 60 days ahead
    df['FutureClose'] = df['Close'].shift(-60)
    df['TargetReturn'] = (df['FutureClose'] - df['Close']) / df['Close']
    
    # Drop NaN
    df = df.dropna()
    
    return df

def create_sequences(data, target, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)

def train_and_predict():
    print("Starting LSTM Pipeline...")
    
    try:
        tech_df = pd.read_parquet("database/technicals.parquet")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Preprocessing Data...")
    
    # We need to process each ticker separately to avoid data leakage across tickers
    tickers = tech_df['Ticker'].unique()
    
    all_X = []
    all_y = []
    
    # Store scalers and data for later prediction
    ticker_data = {}
    
    features = ['Close', 'EMA50', 'EMA200', 'RSI']
    lookback = 60
    
    for ticker in tickers:
        ticker_df = tech_df[tech_df['Ticker'] == ticker].copy()
        
        if len(ticker_df) < lookback + 60: # Need enough data for lookback + target shift
            continue
            
        processed_df = prepare_data(ticker_df, lookback)
        
        if processed_df.empty:
            continue
            
        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(processed_df[features])
        
        X, y = create_sequences(scaled_data, processed_df['TargetReturn'].values, lookback)
        
        if len(X) == 0:
            continue
            
        all_X.append(X)
        all_y.append(y)
        
        # Save last sequence for inference
        # We need the *latest* data available (not dropped due to shift)
        # Re-calc features on full latest data
        full_latest = ticker_df.copy()
        full_latest['Date'] = pd.to_datetime(full_latest['Date'])
        full_latest = full_latest.sort_values('Date')
        full_latest['EMA50'] = full_latest['Close'].ewm(span=50, adjust=False).mean()
        full_latest['EMA200'] = full_latest['Close'].ewm(span=200, adjust=False).mean()
        full_latest['RSI'] = calculate_rsi(full_latest['Close'])
        full_latest = full_latest.dropna(subset=features)
        
        if len(full_latest) >= lookback:
            last_sequence = full_latest[features].values[-lookback:]
            last_sequence_scaled = scaler.transform(last_sequence) # Use same scaler
            ticker_data[ticker] = last_sequence_scaled.reshape(1, lookback, len(features))

    if not all_X:
        print("Not enough data to train.")
        return

    print("Concatenating all sequences...")
    X_train = np.concatenate(all_X)
    y_train = np.concatenate(all_y)
    
    print(f"Training Data Shape: {X_train.shape}")
    
    # Build LSTM Model
    print("Building LSTM Model...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, len(features))),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1) # Regression output: Predicted Return
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Training Model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    
    print("Generating Predictions...")
    
    # Initialize output columns
    tech_df['Label'] = 'Neutral'
    tech_df['Technical_ML_Score'] = 5.0 # Default Neutral Score
    
    # Map predictions back to tech_df
    # We only predict for the *latest* state of each ticker to update the DB
    # Ideally we'd have a timeseries of predictions, but for the dashboard we usually show current status.
    # The user asked for "predict upward, downward, neutral and potential".
    
    updates = []
    
    for ticker, input_seq in ticker_data.items():
        predicted_return = model.predict(input_seq, verbose=0)[0][0]
        
        # Determine Label
        # Thresholds: > 10% (0.10) Upward, < -10% (-0.10) Downward
        if predicted_return > 0.10:
            label = "Strong" # "Upward"
            score_val = 8.0 + (predicted_return * 10) # rough scaling
        elif predicted_return < -0.10:
            label = "Weak" # "Downward"
            score_val = 2.0 + (predicted_return * 10)
        else:
            label = "Neutral"
            score_val = 5.0 + (predicted_return * 10) # slightly adjust around 5
            
        # Clip score 0-10
        score_val = max(0.0, min(10.0, score_val))
        
        updates.append({
            'Ticker': ticker,
            'Label': label,
            'Technical_ML_Score': score_val
        })
        
    # Apply updates
    print("Applying updates to dataframe...")
    # This is O(N), but N is small (number of tickers)
    for update in updates:
        mask = tech_df['Ticker'] == update['Ticker']
        # Apply to all rows or just latest? Usually latest features imply latest prediction.
        # But let's just update the whole column for simplicity or just the last row?
        # The prompt implies we want to update the DB so the dashboard sees it.
        # Let's update all rows for that ticker with the *latest* prediction (Static prediction for now)
        # OR better, purely leave it null for old rows and only set for latest?
        # App expects a value. Let's fill forward.
        
        tech_df.loc[mask, 'Label'] = update['Label']
        tech_df.loc[mask, 'Technical_ML_Score'] = update['Technical_ML_Score']

    output_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'DailyReturn', 
                   'SMA20', 'SMA50', 'SMA200', 'Label', 'Technical_ML_Score']
    
    # Ensure cols exist
    for col in output_cols:
        if col not in tech_df.columns:
            tech_df[col] = None
            
    final_tech_df = tech_df[output_cols]
    
    output_path = "database/technicals.parquet"
    final_tech_df.to_parquet(output_path, index=False)
    print(f"Updated technicals with LSTM predictions saved to {output_path}")

if __name__ == "__main__":
    train_and_predict()
