import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# Function to preprocess the data according to the FELM dataset format
def preprocess_data(df):
    # No need to rename columns as they already match the expected format
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Convert volume from string to numeric (remove 'M' suffix and multiply by 1,000,000)
    df['Vol.'] = df['Vol.'].str.replace('M', '').astype(float) * 1000000
    
    # Convert percentage change to numeric
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100
    
    # Convert price columns to numeric (they should already be numeric, but just to be safe)
    price_columns = ['Price', 'Open', 'High', 'Low']
    for col in price_columns:
        df[col] = pd.to_numeric(df[col])
    
    # Sort data by date
    df = df.sort_values('Date')
    
    return df

# Function to create technical indicators
def add_technical_indicators(df):
    # Moving Averages
    df['MA5'] = df['Price'].rolling(window=5).mean()
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['MA50'] = df['Price'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Price'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Price'].ewm(span=12, adjust=False).mean()
    exp2 = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Price'].rolling(window=20).mean()
    std_dev = df['Price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    
    # Price Rate of Change
    df['ROC'] = df['Price'].pct_change(periods=5) * 100
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[1:, 'OBV'] = (df['Price'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['Vol.']).cumsum()
    
    return df

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Target is the 'Price' or 'close' price
    return np.array(X), np.array(y)

# Unified ARIMA model training function
def train_arima_model(train_data, test_data, exog_train=None, exog_test=None):
    if exog_train is not None:
        auto_model = auto_arima(train_data, 
                       exogenous=exog_train,
                       seasonal=True,  # Enable seasonality
                       m=20,  # Set to appropriate seasonal cycle (e.g., 20 trading days for monthly)
                       stepwise=True,
                       suppress_warnings=True,
                       error_action="ignore",
                       max_p=5, max_q=3, max_d=2,
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

# Main function to process, train and evaluate
def process_and_train_arima(file_path, sequence_length=60, test_size=0.3, use_exogenous=False):
    print("Loading and preprocessing data...")

    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = df.dropna()

    # Update features to match the FELM dataset column names
    features = ['Open', 'High', 'Low', 'Price', 'Vol.', 
                'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal',
                'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'OBV']

    train_size = int(len(df) * (1 - test_size))
    train_data = df[:train_size].copy()
    test_data = df[train_size:].copy()
    
    train_dates = train_data['Date'].values
    test_dates = test_data['Date'].values

    scaler_X = MinMaxScaler()
    train_X = scaler_X.fit_transform(train_data[features])
    test_X = scaler_X.transform(test_data[features])
  
    scaler_y = MinMaxScaler()
    train_y = scaler_y.fit_transform(train_data[['Price']])
    test_y = scaler_y.transform(test_data[['Price']])
    
    X_train_seq, y_train_seq = create_sequences(train_X, sequence_length)
    X_test_seq, y_test_seq = create_sequences(test_X, sequence_length)
   
    # Use 'Price' instead of 'close'
    train_close = train_data['Price'].values
    test_close = test_data['Price'].values
   
    # Check if we have enough test data after sequence_length
    if len(test_close) <= sequence_length:
        print(f"Warning: Not enough test data after applying sequence_length. Adjusting test data usage.")
        test_close_for_evaluation = test_close
    else:
        test_close_for_evaluation = test_close[sequence_length:]

    train_close_log = np.log(train_data['Price'])
    train_close_log_diff = train_close_log.diff().dropna()
    check_stationarity(train_close_log_diff)
    
    # Select exogenous variables (optional)
    if use_exogenous:
        # Use first differences of technical indicators to improve stationarity
        exog_features = ['Vol.', 'RSI']

        print("Correlation between exogenous variables:")
        print(train_data[exog_features].corr())
        
        # Calculate first differences of these features
        # for feature in exog_features:
        #     train_data[f'{feature}_diff'] = train_data[feature].diff()
        #     test_data[f'{feature}_diff'] = test_data[feature].diff()
    
        # Use differenced features for exogenous variables
        # diff_features = [f'{feature}_diff' for feature in exog_features]
        # train_exog = train_data[diff_features].fillna(0).values

        train_exog = train_data[exog_features].fillna(0).values
        test_exog = test_data[exog_features].fillna(0).values[sequence_length:]
        
        # if len(test_close) <= sequence_length:
        #     test_exog = test_data[diff_features].fillna(0).values
        # else:
        #     test_exog = test_data[diff_features].fillna(0).values[sequence_length:]
    else:
        train_exog = None
        test_exog = None

    print("Training ARIMA model...")
    # Train ARIMA and get predictions
    # arima_preds, arima_model = train_arima_model(train_close, test_close_for_evaluation, train_exog, test_exog)
    

    sarima_model = SARIMAX(train_data['Price'], 
                      order=(1, 1, 1),           # Non-seasonal parameters (p,d,q)
                      seasonal_order=(1, 1, 1, 20),  # Seasonal parameters (P,D,Q,s)
                      enforce_stationarity=False,
                      enforce_invertibility=False)

    sarima_results = sarima_model.fit(disp=False)
    print(sarima_results.summary())

# Make predictions
    sarima_preds = sarima_results.forecast(steps=len(test_data))


    # For evaluation, convert test sequence back to original scale
    if len(y_test_seq) > 0:
        y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
    else:
        # If not enough data for sequence, use the test data directly
        y_test_original = test_close_for_evaluation.reshape(-1, 1)
    
    # Calculate metrics with proper error handling
    try:
        metrics = {
            # 'arima': {
            #     'rmse': np.sqrt(mean_squared_error(y_test_original, arima_preds)),
            #     'mape': mean_absolute_percentage_error(y_test_original, arima_preds),
            #     'r2': r2_score(y_test_original, arima_preds)
            # },
            'sarima': {
        'rmse': np.sqrt(mean_squared_error(test_data['Price'], sarima_preds)),
        'mape': mean_absolute_percentage_error(test_data['Price'], sarima_preds),
        'r2': r2_score(test_data['Price'], sarima_preds)
    }
        }
        
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
        # Use appropriate dates depending on sequence length handling
        if len(test_close) <= sequence_length:
            plot_dates = test_dates
        else:
            plot_dates = test_dates[sequence_length:]
            
        plt.plot(plot_dates, y_test_original, label='Actual')
        plt.plot(plot_dates, sarima_preds, label='ARIMA Prediction', alpha=0.7)
        plt.title('Stock Price Prediction Comparison (FELM)')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('stock_prediction_arima_felm.png')
        plt.close()
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    # Save models and necessary objects
    if not os.path.exists('model'):
        os.makedirs('model')

    # Save ARIMA model
    with open('model/sarima_model.pkl', 'wb') as f:
        pickle.dump(sarima_model, f)
    
    # Save scalers
    with open('model/scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    
    with open('model/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
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
    
    print("Models and related objects saved successfully! Used ARIMA")
    
    return sarima_model, metrics

if __name__ == '__main__':
    # Replace with your actual CSV file path
    file_path = 'ZOMT.csv'
    
    # Train and save models
    arima_model, metrics = process_and_train_arima(
        file_path, 
        sequence_length=105,  # Reduced sequence length due to small dataset
        test_size=0.2,
        use_exogenous=True  # Set to True to use exogenous variables for ARIMA
    )