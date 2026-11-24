import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow.keras.regularizers import l2 # type: ignore

def preprocess_data(df):
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    df = df.rename(columns={'Vol.': 'volume'})
    df['volume'] = df['volume'].str.replace('M', '').astype(float) * 1000000
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df.sort_values('Date')

    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

    df['time_idx'] = range(len(df))

    df['group'] = 0
    
    return df


def add_technical_indicators(df):
    # Moving Averages
    df['MA5'] = df['Price'].rolling(window=5).mean()  # Use 'Price' instead of 'price'
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['MA50'] = df['Price'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Price'].diff()  # Use 'Price' instead of 'price'
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Price'].ewm(span=12, adjust=False).mean()  # Use 'Price' instead of 'price'
    exp2 = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Price'].rolling(window=20).mean()  # Use 'Price' instead of 'price'
    std_dev = df['Price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    
    # Price Rate of Change
    df['ROC'] = df['Price'].pct_change(periods=5) * 100  # Use 'Price' instead of 'price'
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[1:, 'OBV'] = (df['Price'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()  # Use 'Price' instead of 'price'
    
    return df


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def train_model(file_path, sequence_length=60, test_size=0.2):
   
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = df.dropna()

    features = ["Price", "Open", "High", "Low", "volume", "Change %",
            "MA5", "MA20", "MA50", "RSI", "MACD", "MACD_signal",
            "BB_middle", "BB_upper", "BB_lower", "ROC", "OBV"]

    train_size = int(len(df) * (1 - test_size))
    train_data = df[:train_size].copy()
    test_data = df[train_size:].copy()
    
    train_dates = train_data['Date'].values
    test_dates = test_data['Date'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    

    train_X = scaler_X.fit_transform(train_data[features])
    train_y = scaler_y.fit_transform(train_data[['Price']])

    test_X = scaler_X.transform(test_data[features])
    test_y = scaler_y.transform(test_data[['Price']])
    

    X_train_seq, y_train_seq = create_sequences(train_X, sequence_length)
    X_test_seq, y_test_seq = create_sequences(test_X, sequence_length)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=100, return_sequences=True, activation='tanh',
                   kernel_regularizer=l2(0.001),
                   input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(units=1))
    
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    lstm_history = lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # # Train Random Forest model
    # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf_model.fit(X_train_rf, y_train_rf)
    
    # Make predictions
    lstm_preds = lstm_model.predict(X_test_seq)
    # rf_preds = rf_model.predict(X_test_rf).reshape(-1, 1)
    
    # Determine best weights for ensemble (using a simple 50-50 split for now)
    # You could optimize this using a validation set
    # ensemble_weights = {'lstm': 0.95, 'rf': 0.05}
    
    # Create ensemble predictions
    # ensemble_preds = (lstm_preds * ensemble_weights['lstm']) + (rf_preds * ensemble_weights['rf'])
    
    # Inverse transform predictions to original scale
    lstm_preds_original = scaler_y.inverse_transform(lstm_preds)
    # rf_preds_original = scaler_y.inverse_transform(rf_preds)
    # ensemble_preds_original = scaler_y.inverse_transform(ensemble_preds)
    y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
    
    # Calculate metrics
    metrics = {
        'lstm': {
            'rmse': np.sqrt(mean_squared_error(y_test_original, lstm_preds_original)),
            'mape': mean_absolute_percentage_error(y_test_original, lstm_preds_original),
            'r2': r2_score(y_test_original, lstm_preds_original)
        },
        # 'rf': {
        #     'rmse': np.sqrt(mean_squared_error(y_test_original, rf_preds_original)),
        #     'mape': mean_absolute_percentage_error(y_test_original, rf_preds_original),
        #     'r2': r2_score(y_test_original, rf_preds_original)
        # },
        # 'ensemble': {
        #     'rmse': np.sqrt(mean_squared_error(y_test_original, ensemble_preds_original)),
        #     'mape': mean_absolute_percentage_error(y_test_original, ensemble_preds_original),
        #     'r2': r2_score(y_test_original, ensemble_preds_original)
        # }
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
    plt.plot(test_dates[sequence_length:], y_test_original, label='Actual')
    plt.plot(test_dates[sequence_length:], lstm_preds_original, label='LSTM Model Prediction')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_prediction.png')
    plt.close()
    
    # Save models and necessary objects
    if not os.path.exists('lst_model'):
        os.makedirs('lst_model')
    
    # # Save LSTM model
    lstm_model.save('lst_model/lstm_model.h5')
    
    # Save RF model
    # with open('models/rf_model.pkl', 'wb') as f:
    #     pickle.dump(rf_model, f)
    
    # Save scalers
    with open('lst_model/scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    
    with open('lst_model/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Save ensemble weights
    # with open('models/ensemble_weights.pkl', 'wb') as f:
    #     pickle.dump(ensemble_weights, f)
    
    # Save sequence length
    with open('lst_model/sequence_length.pkl', 'wb') as f:
        pickle.dump(sequence_length, f)
    
    # Save feature list
    with open('lst_model/features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("Models and related objects saved successfully!")
    
    return lstm_model    #, rf_model, metrics, ensemble_weights



if __name__ == '__main__':
    # Replace with your actual CSV file path
    file_path = 'ZOMT.csv'
    # Train and save models
   # lstm_model, rf_model, metrics, ensemble_weights = train_model(file_path)
    lstm_model = train_model(file_path)
    # To make future predictions, use the following:
    # new_data = pd.read_csv('new_stock_data.csv')
    # future_predictions = predict_stock_prices(new_data, days_to_predict=5)
    # print(future_predictions)