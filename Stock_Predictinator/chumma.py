import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model #type: ignore
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    return df


def add_technical_indicators(df):
    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    
    # Price Rate of Change
    df['ROC'] = df['close'].pct_change(periods=5) * 100
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[1:, 'OBV'] = (df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()
    
    # Add normalized price change features - daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Add volatility measures (rolling standard deviation)
    df['volatility_5d'] = df['close'].pct_change().rolling(window=5).std()
    df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
    
    # Add price momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    return df



# 📌 Preprocess the dataset (Ensure it matches training format)
def preprocess_data(df):
# Rename columns to match the original code
    df = df.rename(columns={
        'Date': 'date',
        'Price': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Vol.': 'volume'
    })
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    
    # Convert volume from string to numeric (remove 'M' suffix and multiply by 1,000,000)
    df['volume'] = df['volume'].str.replace('M', '').astype(float) * 1000000
    
    # Sort data by date
    df = df.sort_values('date',reversed= True)
    print(df)
    return df

# 📌 Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 0])  # Predicting 'close' price
    return np.array(sequences), np.array(labels)




# 📌 Load trained models and scalers
lstm_model = load_model('xlsm/lstm_model.h5')  # Load LSTM Model
xgb_model = pickle.load(open('xlsm/xgb_model.pkl', 'rb'))  # Load XGBoost Model
scaler_X = pickle.load(open('xlsm/scaler_X.pkl', 'rb'))  # Feature Scaler
scaler_y = pickle.load(open('xlsm/scaler_y.pkl', 'rb'))  # LSTM Target Scaler
scaler_y_xgb = pickle.load(open('xlsm/scaler_y_xgb.pkl', 'rb'))  # XGBoost Target Scaler


# 📌 Load last 30 days data from CSV
df = pd.read_csv('som.csv')

# Process the test dataset
df = preprocess_data(df)

# ✅ Ensure lagged features and technical indicators are added
df = add_lagged_features(df)
df = add_technical_indicators(df)

# Drop NaN values that may appear due to rolling windows and shifting
df = df.dropna() # Add technical indicators


# 📌 Define feature columns (Same as used in training)
features = ['open', 'high', 'low', 'close', 'volume', 
            'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal',
            'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'OBV',
            'daily_return', 'volatility_5d', 'volatility_20d', 
            'momentum_5d', 'momentum_10d']

lagged_features = [f'close_lag_{lag}' for lag in [1, 2, 3, 5, 10]]
features = features + lagged_features

# 📌 Scale the data
test_X = scaler_X.transform(df[features])
test_y = scaler_y.transform(df[['close']])

# 📌 Prepare LSTM sequences
sequence_length = 60  # Should match your LSTM training sequence length
X_test_seq, y_test_seq = create_sequences(test_X, sequence_length)
if X_test_seq.size == 0:
    raise ValueError("❌ Error: X_test_seq is empty. Not enough data for prediction.")

# 📌 Align XGBoost test data
test_X_xgb = df[features].values
test_X_xgb_aligned = test_X_xgb[sequence_length:]

# 📌 Make predictions using LSTM & XGBoost
lstm_preds = lstm_model.predict(X_test_seq)
# 📌 Check Expected Features
print(f"Expected features by XGBoost: {xgb_model.n_features_in_}")
print(f"Shape of test_X_xgb_aligned before selection: {test_X_xgb_aligned.shape}")

xgb_features = ['open', 'high', 'low']
test_X_xgb_aligned = df[xgb_features].values[sequence_length:]

# 📌 Check Shape Again After Fixing
print(f"Shape of test_X_xgb_aligned after selection: {test_X_xgb_aligned.shape}")

# 📌 Predict Using XGBoost (After Fixing Feature Mismatch)
xgb_preds = xgb_model.predict(test_X_xgb_aligned).reshape(-1, 1)

# 📌 Inverse scale predictions
lstm_preds_original = scaler_y.inverse_transform(lstm_preds)
xgb_preds_original = scaler_y_xgb.inverse_transform(xgb_preds)
y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))

# 📌 Ensemble Prediction (Weighted Average)
ensemble_preds = (lstm_preds_original * 0.6) + (xgb_preds_original * 0.4)

# 📌 Create a DataFrame with predictions
predictions_df = pd.DataFrame({
    'Date': df['date'][sequence_length:].values,
    'Actual_Close': y_test_original.flatten(),
    'LSTM_Prediction': lstm_preds_original.flatten(),
    'XGBoost_Prediction': xgb_preds_original.flatten(),
    'Ensemble_Prediction': ensemble_preds.flatten()
})

# 📌 Save predictions to CSV
predictions_df.to_csv('predictbruh.csv', index=False)
print("✅ Predictions saved to predictions_last_30_days.csv")

# 📌 Display first few predictions
print(predictions_df.head())
