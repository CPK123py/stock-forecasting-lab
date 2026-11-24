import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow.keras.regularizers import l2 #type: ignore
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(series):
    # ADF Test
    adf_result = adfuller(series)
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    
    # KPSS Test
    kpss_result = kpss(series)
    print(f'KPSS Statistic: {kpss_result[0]}')
    print(f'p-value: {kpss_result[1]}')
    
    # If both tests suggest non-stationarity, apply transformations
    if adf_result[1] > 0.05 and kpss_result[1] < 0.05:
        print("Series is likely non-stationary, transformation recommended")
    else:
        print("Series may be stationary")


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to preprocess the data according to the new format
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
    df = df.sort_values('date')
    
    return df

# Function to create technical indicators
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

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Target is the 'close' price
    return np.array(X), np.array(y)

# Function to train and get ARIMA predictions
def train_arima_model(train_data, test_data, exog_train=None, exog_test=None):
    """
    Train an ARIMA model and generate predictions
    
    Parameters:
    train_data (array): Training data for ARIMA (univariate series)
    test_data (array): Test data to predict
    exog_train (array, optional): Exogenous variables for training
    exog_test (array, optional): Exogenous variables for testing
    
    Returns:
    array: ARIMA predictions for test period
    object: Trained ARIMA model
    """
    print("Finding optimal ARIMA parameters...")
    # Use auto_arima to find the best parameters
    if exog_train is not None:
        auto_model = auto_arima(train_data, 
                       exogenous=exog_train,
                       seasonal=True,  # Enable seasonality
                       m=20,  # Set to appropriate seasonal cycle (e.g., 20 trading days for monthly)
                       stepwise=True,
                       suppress_warnings=True,
                       error_action="ignore",
                       max_p=3, max_q=3, max_d=1,
                       max_P=2, max_Q=2, max_D=1,  # Seasonal parameters
                       trace=True)
    else:
        auto_model = auto_arima(train_data, 
                               seasonal=False, 
                               stepwise=True,
                               suppress_warnings=True,
                               error_action="ignore",
                               max_p=5, max_q=5, max_d=2,
                               trace=True)
    
    # Get the best parameters
    best_order = auto_model.order
    print(f"Best ARIMA parameters: {best_order}")
    
    # Fit the ARIMA model with the best parameters
    if exog_train is not None:
        model = ARIMA(train_data, order=best_order, exog=exog_train)
    else:
        model = ARIMA(train_data, order=best_order)
    
    arima_model = model.fit()
    print(arima_model.summary())
    
    # Generate forecasts
    if exog_test is not None:
        arima_preds = arima_model.forecast(steps=len(test_data), exog=exog_test)
    else:
        arima_preds = arima_model.forecast(steps=len(test_data))
    
    return arima_preds, arima_model

# Main training function with new ARIMA integration
def train_enhanced_hybrid_model(file_path, sequence_length=60, test_size=0.2, use_exogenous=False):
    """
    Train an enhanced hybrid model with ARIMA and LSTM for stock price prediction
    
    Parameters:
    file_path (str): Path to the CSV file containing stock data
    sequence_length (int): Number of time steps to use for LSTM sequences
    test_size (float): Proportion of data to use for testing
    use_exogenous (bool): Whether to use exogenous variables for ARIMA
    
    Returns:
    Tuple of trained models and metrics
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = df.dropna()

    features = ['open', 'high', 'low', 'close', 'volume', 
                'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal',
                'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'OBV']

    train_size = int(len(df) * (1 - test_size))
    train_data = df[:train_size].copy()
    test_data = df[train_size:].copy()
    
    train_dates = train_data['date'].values
    test_dates = test_data['date'].values

    # # Scale features for LSTM
    scaler_X = MinMaxScaler()
    train_X = scaler_X.fit_transform(train_data[features])
    test_X = scaler_X.transform(test_data[features])
    
    # # Scale target variable separately
    scaler_y = MinMaxScaler()
    train_y = scaler_y.fit_transform(train_data[['close']])
    test_y = scaler_y.transform(test_data[['close']])
    

    # train_seq_X, _ = create_sequences(train_X, sequence_length)
    # lstm_train_preds = lstm_model.predict(train_seq_X)
    # lstm_train_preds = scaler_y.inverse_transform(lstm_train_preds).flatten()



    
    # # Create sequences for LSTM
    X_train_seq, y_train_seq = create_sequences(train_X, sequence_length)
    X_test_seq, y_test_seq = create_sequences(test_X, sequence_length)
    
    # # Define ensemble weights
    # ensemble_weights = {'arima': 0.7, 'lstm': 0.3}
    
    # print("Training LSTM model...")
    # # Train LSTM model
    # lstm_model = Sequential()
    # lstm_model.add(LSTM(units=100, return_sequences=True, activation='tanh',
    #                kernel_regularizer=l2(0.001),
    #                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    # lstm_model.add(Dropout(0.2))
    # lstm_model.add(LSTM(units=50))
    # lstm_model.add(Dropout(0.2))
    # lstm_model.add(Dense(units=1))
    
    # lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # lstm_history = lstm_model.fit(
    #     X_train_seq, y_train_seq,
    #     epochs=100,
    #     batch_size=32,
    #     validation_split=0.1,
    #     callbacks=[early_stopping],
    #     verbose=1
    # )
    
    # # Make LSTM predictions
    # lstm_preds = lstm_model.predict(X_test_seq)
    
    # Prepare data for ARIMA
    # For ARIMA, we need original scale data
    train_close = train_data['close'].values
    test_close = test_data['close'].values[sequence_length:]  # Match LSTM prediction length


    train_close_log = np.log(train_data['close'].values)
    check_stationarity(train_close_log)
    
    # Select exogenous variables (optional)
    if use_exogenous:
 # Use first differences of technical indicators to improve stationarity
        exog_features = ['MA5', 'MA20', 'volume']
    
    # Calculate first differences of these features
        for feature in exog_features:
            train_data[f'{feature}_diff'] = train_data[feature].diff()
            test_data[f'{feature}_diff'] = test_data[feature].diff()
    
    # Use differenced features for exogenous variables
        diff_features = [f'{feature}_diff' for feature in exog_features]
        train_exog = train_data[diff_features].fillna(0).values
        test_exog = test_data[diff_features].fillna(0).values[sequence_length:]
    else:
        train_exog = None
        test_exog = None


    print("Training ARIMA model...")
    # Train ARIMA and get predictions
    arima_preds, arima_model = train_arima_model(train_close, test_close, train_exog, test_exog)
    
    # # Reshape ARIMA predictions to match LSTM
    # arima_preds = arima_preds.reshape(-1, 1)
    
    # # Normalize ARIMA predictions for ensemble
    # arima_preds_norm = scaler_y.transform(arima_preds)
    
    # # Ensure predictions have the same shape
    # print(f"LSTM predictions shape: {lstm_preds.shape}")
    # print(f"ARIMA predictions shape: {arima_preds_norm.shape}")
    
    # Create ensemble predictions
    #ensemble_preds = (arima_preds_norm * ensemble_weights['arima']) + (lstm_preds * ensemble_weights['lstm'])
    
    # Inverse transform predictions to original scale for evaluation
    #lstm_preds_original = scaler_y.inverse_transform(lstm_preds)
    # ARIMA is already in original scale
    #ensemble_preds_original = scaler_y.inverse_transform(ensemble_preds)
    y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
    
    # Calculate metrics with proper error handling
    try:
        metrics = {
            # 'lstm': {
            #     'rmse': np.sqrt(mean_squared_error(y_test_original, lstm_preds_original)),
            #     'mape': mean_absolute_percentage_error(y_test_original, lstm_preds_original),
            #     'r2': r2_score(y_test_original, lstm_preds_original)
            # },
            'arima': {
                'rmse': np.sqrt(mean_squared_error(y_test_original, arima_preds)),
                'mape': mean_absolute_percentage_error(y_test_original, arima_preds),
                'r2': r2_score(y_test_original, arima_preds)
            },
            # 'ensemble': {
            #     'rmse': np.sqrt(mean_squared_error(y_test_original, ensemble_preds_original)),
            #     'mape': mean_absolute_percentage_error(y_test_original, ensemble_preds_original),
            #     'r2': r2_score(y_test_original, ensemble_preds_original)
            # }
        }
        
        # Print metrics
        for model_name_key, model_metrics in metrics.items():
            print(f"=== {model_name_key.upper()} Model Metrics ===")
            print(f"RMSE: {model_metrics['rmse']:.4f}")
            print(f"MAPE: {model_metrics['mape']:.4f}")
            print(f"R²: {model_metrics['r2']:.4f}")
            print()
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print("Continuing with model saving...")
    
    # Plot actual vs predicted
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates[sequence_length:], y_test_original, label='Actual')
        # plt.plot(test_dates[sequence_length:], lstm_preds_original, label='LSTM Prediction', alpha=0.8)
        plt.plot(test_dates[sequence_length:], arima_preds, label='ARIMA Prediction', alpha=0.7)
        # plt.plot(test_dates[sequence_length:], ensemble_preds_original, label=f'ARIMA+LSTM Ensemble', alpha=0.9)
        plt.title('Stock Price Prediction Comparison')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('stock_prediction_arima_lstm.png')
        plt.close()
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    # Save models and necessary objects
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Save LSTM model
    #lstm_model.save('models/lstm_model_arima.h5')
    
    # Save ARIMA model
    with open('model/arima_model.pkl', 'wb') as f:
        pickle.dump(arima_model, f)
    
    # Save scalers
    with open('model/scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    
    with open('model/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Save ensemble weights
    #with open('models/ensemble_weights_arima.pkl', 'wb') as f:
       # pickle.dump(ensemble_weights, f)
    
    # Save sequence length
    with open('model/sequence_length.pkl', 'wb') as f:
        pickle.dump(sequence_length, f)
    
    # Save feature list
    with open('model/features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    # Save exogenous feature flag and list
    with open('model/use_exogenous.pkl', 'wb') as f:
        pickle.dump(use_exogenous, f)
    
    if use_exogenous:
        with open('model/exog_features.pkl', 'wb') as f:
            pickle.dump(exog_features, f)
    
    print("Models and related objects saved successfully! Used ARIMA + LSTM Ensemble")
    
    # Return a prediction function for convenience
    def predict_stock_prices(new_data, days_to_predict=5):
        """
        Make predictions for stock prices using the trained models
        
        Parameters:
        new_data (pd.DataFrame): New stock data to make predictions from
        days_to_predict (int): Number of days to predict into the future
        
        Returns:
        pd.DataFrame: DataFrame with predicted prices
        """
        # Preprocess new data
        new_data = preprocess_data(new_data)
        new_data = add_technical_indicators(new_data)
        new_data = new_data.dropna()
        
        # Get the most recent data for sequence creation
        recent_data = new_data.tail(sequence_length)[features]
        recent_dates = new_data.tail(sequence_length)['date']
        recent_close = new_data.tail(sequence_length)['close'].values
        
        # Scale the data for LSTM
        recent_scaled = scaler_X.transform(recent_data)
        
        # Create a single sequence for LSTM prediction
        pred_sequence = recent_scaled.reshape(1, sequence_length, len(features))
        
        # Prepare exogenous variables if used
        if use_exogenous:
            recent_exog = new_data.tail(sequence_length)[exog_features].values
            future_exog = []  # This would need a strategy to forecast future exogenous variables
        else:
            recent_exog = None
            future_exog = None
        
        # Initialize predictions list
        predictions = []
        prediction_dates = []
        lstm_predictions = []
        arima_predictions = []
        
        # Get the last date for generating future dates
        last_date = new_data['date'].iloc[-1]
        
        # Make initial ARIMA predictions for the future days
        # For simplicity, we're forecasting all days at once
        # In a real application, you might want to update the model iteratively
        try:
            if use_exogenous and future_exog is not None:
                arima_future = arima_model.forecast(steps=days_to_predict, exog=future_exog)
            else:
                arima_future = arima_model.forecast(steps=days_to_predict)
        except Exception as e:
            print(f"ARIMA prediction error: {e}")
            # Fallback to simple forecast
            arima_future = np.array([recent_close[-1]] * days_to_predict)
        
        # Make iterative predictions
        for i in range(days_to_predict):
            # Make prediction with LSTM
            # lstm_pred = lstm_model.predict(pred_sequence)
            # lstm_pred_original = scaler_y.inverse_transform(lstm_pred)[0, 0]
            # lstm_predictions.append(lstm_pred_original)
            
            # Get ARIMA prediction for this step
            arima_pred = arima_future[i]
            arima_predictions.append(arima_pred)
            
            # # Create ensemble prediction
            # ensemble_pred = (arima_pred * ensemble_weights['arima']) + (lstm_pred_original * ensemble_weights['lstm'])
            # predictions.append(ensemble_pred)
            
            # Create next date
            next_date = last_date + pd.Timedelta(days=i+1)
            prediction_dates.append(next_date)
            
            # Update the sequence for the next LSTM prediction
            # Create a new row with all features
            # This is simplified by copying the last row's values for other features
            new_row = pred_sequence[0, -1].copy()
            # Update the close price (assuming 'close' is at index 3)
            # Convert ensemble prediction back to normalized scale for LSTM
            new_close_normalized = scaler_y.transform([[arima_pred]])[0, 0]
            new_row[3] = new_close_normalized
            
            # Update the prediction sequence by removing the first row and adding the new row
            pred_sequence = np.append(pred_sequence[:, 1:, :], new_row.reshape(1, 1, len(features)), axis=1)
        
        # Create a DataFrame with the predictions
        prediction_df = pd.DataFrame({
            'date': prediction_dates,
            # 'ensemble_prediction': predictions,
            # 'lstm_prediction': lstm_predictions,
            'arima_prediction': arima_predictions
        })
        
        return prediction_df
        
    return lstm_model, arima_model, metrics, ensemble_weights, predict_stock_prices


if __name__ == '__main__':
    # Replace with your actual CSV file path
    file_path = 'ZOMT.csv'
    
    # Train and save models
    lstm_model, arima_model, metrics, ensemble_weights, predict_func = train_enhanced_hybrid_model(
        file_path, 
        sequence_length=60, 
        test_size=0.2,
        use_exogenous=True  # Set to True to use exogenous variables for ARIMA
    )
    
    # Example of making future predictions
    new_data = pd.read_csv('som.csv')
    future_predictions = predict_func(new_data, days_to_predict=5)
    print("Future price predictions:")
    print(future_predictions)