import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Function to create technical indicators (same as in training script)
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
    
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Target is the 'close' price
    return np.array(X), np.array(y)

def load_hybrid_model():
    # Load LSTM model
    lstm_model = load_model('models/lstm_model.h5')
    
    # Load RF model
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Load scalers
    with open('models/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    
    with open('models/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    # Load ensemble weights
    with open('models/ensemble_weights.pkl', 'rb') as f:
        ensemble_weights = pickle.load(f)
    
    # Load sequence length
    with open('models/sequence_length.pkl', 'rb') as f:
        sequence_length = pickle.load(f)
    
    # Load feature list
    with open('models/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return {
        'lstm_model': lstm_model,
        'rf_model': rf_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'ensemble_weights': ensemble_weights,
        'sequence_length': sequence_length,
        'features': features
    }

def predict_future_prices(model_dict, data_file, days_to_predict=30):
    # Load data
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Extract models and parameters
    lstm_model = model_dict['lstm_model']
    rf_model = model_dict['rf_model']
    scaler_X = model_dict['scaler_X']
    scaler_y = model_dict['scaler_y']
    ensemble_weights = model_dict['ensemble_weights']
    sequence_length = model_dict['sequence_length']
    features = model_dict['features']
    
    # Scale the data
    scaled_data = scaler_X.transform(df[features])
    
    # Make a copy of the last 'sequence_length' days of data for predictions
    last_sequence = scaled_data[-sequence_length:].copy()
    
    # Lists to store predictions
    future_dates = pd.date_range(start=df['date'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)
    predictions = []
    
    # Predict future prices
    for _ in range(days_to_predict):
        # Prepare data for LSTM
        lstm_input = last_sequence.reshape(1, sequence_length, len(features))
        
        # Prepare data for RF
        rf_input = last_sequence[-1].reshape(1, -1)
        
        # Make predictions
        lstm_pred = lstm_model.predict(lstm_input, verbose=0)
        rf_pred = rf_model.predict(rf_input).reshape(-1, 1)
        
        # Create ensemble prediction
        ensemble_pred = (lstm_pred * ensemble_weights['lstm']) + (rf_pred * ensemble_weights['rf'])
        
        # Convert to original scale
        pred_price = scaler_y.inverse_transform(ensemble_pred)[0, 0]
        predictions.append(pred_price)
        
        # Update features for the next prediction
        new_row = np.zeros(len(features))
        new_row[3] = ensemble_pred[0, 0]  # Set 'close' price (index 3)
        
        # Shift the sequence and add the new prediction
        last_sequence = np.vstack([last_sequence[1:], new_row])
    
    # Create a DataFrame for future predictions
    future_df = pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions
    })
    
    return future_df

def evaluate_model(model_dict, test_file):
    # Load data
    df = pd.read_csv(test_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Extract models and parameters
    lstm_model = model_dict['lstm_model']
    rf_model = model_dict['rf_model']
    scaler_X = model_dict['scaler_X']
    scaler_y = model_dict['scaler_y']
    ensemble_weights = model_dict['ensemble_weights']
    sequence_length = model_dict['sequence_length']
    features = model_dict['features']
    
    # Scale the data
    scaled_data = scaler_X.transform(df[features])
    scaled_y = scaler_y.transform(df[['close']])
    
    # Create sequences for LSTM
    X_seq, y_seq = create_sequences(scaled_data, sequence_length)
    
    # Prepare data for Random Forest (using the last sequence_length days)
    X_rf = scaled_data[sequence_length:]
    y_rf = scaled_y[sequence_length:].flatten()
    
    # Make predictions
    lstm_preds = lstm_model.predict(X_seq, verbose=0)
    rf_preds = rf_model.predict(X_rf).reshape(-1, 1)
    
    # Create ensemble predictions
    ensemble_preds = (lstm_preds * ensemble_weights['lstm']) + (rf_preds * ensemble_weights['rf'])
    
    # Inverse transform predictions to original scale
    lstm_preds_original = scaler_y.inverse_transform(lstm_preds)
    rf_preds_original = scaler_y.inverse_transform(rf_preds)
    ensemble_preds_original = scaler_y.inverse_transform(ensemble_preds)
    y_original = scaler_y.inverse_transform(y_seq.reshape(-1, 1))
    
    # Calculate metrics
    metrics = {
        'lstm': {
            'rmse': np.sqrt(mean_squared_error(y_original, lstm_preds_original)),
            'mape': mean_absolute_percentage_error(y_original, lstm_preds_original),
            'r2': r2_score(y_original, lstm_preds_original)
        },
        'rf': {
            'rmse': np.sqrt(mean_squared_error(y_original, rf_preds_original)),
            'mape': mean_absolute_percentage_error(y_original, rf_preds_original),
            'r2': r2_score(y_original, rf_preds_original)
        },
        'ensemble': {
            'rmse': np.sqrt(mean_squared_error(y_original, ensemble_preds_original)),
            'mape': mean_absolute_percentage_error(y_original, ensemble_preds_original),
            'r2': r2_score(y_original, ensemble_preds_original)
        }
    }
    
    # Print metrics
    for model_name, model_metrics in metrics.items():
        print(f"=== {model_name.upper()} Model Metrics ===")
        print(f"RMSE: {model_metrics['rmse']:.4f}")
        print(f"MAPE: {model_metrics['mape']:.4f}")
        print(f"R²: {model_metrics['r2']:.4f}")
        print()
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'][sequence_length:], y_original, label='Actual')
    plt.plot(df['date'][sequence_length:], ensemble_preds_original, label='Hybrid Model Prediction')
    plt.title('Zomato Stock Price - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('zomato_stock_evaluation.png')
    plt.show()
    
    return metrics

if __name__ == '__main__':
    # Load models
    model_dict = load_hybrid_model()
    
    # Evaluate model on test data (if available)
    test_file = 'zom.csv'  # Replace with your test file
    print("Evaluating model on test data...")
    metrics = evaluate_model(model_dict, test_file)
    
    # Predict future prices
    print("\nPredicting future prices...")
    future_df = predict_future_prices(model_dict, test_file, days_to_predict=3)
    
    # Print future predictions
    print("\nFuture price predictions:")
    pd.set_option('display.max_rows', None)
    print(future_df)
    
    # Save predictions to CSV
    future_df.to_csv('zomato_future_predictions.csv', index=False)
    print("Future predictions saved to 'zomato_future_predictions.csv'")
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_df['date'], future_df['predicted_price'])
    plt.title('Zomato Stock Price - 30-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Predicted Stock Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('zomato_stock_forecast.png')
    plt.show()