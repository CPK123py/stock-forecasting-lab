import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
import ta  # Technical Analysis library

# === Data Loading ===
package_directory = os.path.dirname(os.path.abspath(__file__))
csv_folder = f'{package_directory}/../FullDataCSV'
hdf_file = f'{package_directory}/../FullData.h5'

master = None
hdf_master = None

def read_from_key(key, use_hdf=True):
    """Load stock market data from HDF5 or CSV."""
    if use_hdf:
        return pd.read_hdf(hdf_file, key)
    return pd.read_csv(os.path.join(csv_folder, f'{key}.csv'))

def read_master(use_hdf=True):
    """Load master file to get stock metadata."""
    global master, hdf_master
    key = 'master'
    master = read_from_key(key=key, use_hdf=use_hdf)
    hdf_master = use_hdf
    return master

def filter_symbol(query=None, search=None, use_hdf=True):
    """Filter stocks based on name or symbol."""
    if query is None and search is None:
        raise ValueError('Both query and search cannot be None')

    if hdf_master is None or hdf_master != use_hdf:
        read_master(use_hdf=use_hdf)

    # Build query
    if query is not None:
        if re.search('=|<|>', query) is None:
            query = f'tradingsymbol == "{query}" or name == "{query}"'
        query = f'({query})'
    if search is not None:
        search = f'(tradingsymbol.str.contains("{search}") or name.str.contains("{search}"))'
    
    final_query = ' and '.join(filter(None, [query, search]))
    match = master.query(final_query, engine='python').reset_index(drop=True)

    if match.empty:
        raise ValueError('No matching stock found.')
    return match.iloc[0]

def read_data(query=None, search=None, use_hdf=True):
    """Load individual stock data."""
    match = filter_symbol(query=query, search=search, use_hdf=use_hdf)
    return read_from_key(match['key'], use_hdf=use_hdf)

# === Feature Engineering ===
def add_technical_indicators(df):
    """Add technical indicators for feature engineering."""
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df = df.dropna()  # Remove rows with NaN values due to rolling
    return df

# === Data Preparation ===
def normalize_data(df):
    """Normalize data using Min-Max Scaler."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_lstm_data(data, time_steps=60):
    """Prepare sequential data for LSTM."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])  # Predict 'close' price
    return np.array(X), np.array(y)

# === Model Training ===
def train_lstm(X, y):
    """Train LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=64, verbose=1)
    return model

def train_random_forest(X, y):
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# === Hybrid Prediction ===
def hybrid_prediction(model_lstm, model_rf, X_lstm, X_rf):
    """Combine LSTM and RF predictions."""
    pred_lstm = model_lstm.predict(X_lstm).flatten()
    pred_rf = model_rf.predict(X_rf)
    return 0.6 * pred_lstm + 0.4 * pred_rf  # Weighted average

# === Evaluation ===
def evaluate(y_true, y_pred):
    """Evaluate model performance."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse}')

# === Main Execution ===
if __name__ == "__main__":
    # Step 1: Load Data
    df = read_data(query="RELIANCE")  # Replace with your desired stock
    df = df[['close']]  # Keep only 'close' price for simplicity

    # Step 2: Feature Engineering
    df = add_technical_indicators(df)

    # Step 3: Data Preparation
    scaled_data, scaler = normalize_data(df[['close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']])
    X_lstm, y_lstm = create_lstm_data(scaled_data)
    X_rf = df[['SMA_20', 'SMA_50', 'RSI', 'MACD']].shift(1).dropna().values[:-60]  # Align with LSTM
    y_rf = df['close'].shift(-1).dropna().iloc[:len(X_rf)].values

    # Step 4: Train Models
    model_lstm = train_lstm(X_lstm, y_lstm)
    model_rf = train_random_forest(X_rf, y_rf)

    # Step 5: Hybrid Prediction
    final_pred = hybrid_prediction(model_lstm, model_rf, X_lstm, X_rf)

    # Step 6: Evaluation
    evaluate(y_rf[-len(final_pred):], final_pred)
