import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import os
from datetime import timedelta

# Function to add lagged features
def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['RSI'].shift(lag)
        df[f'Change_lag_{lag}'] = df['Change %'].shift(lag)
    return df

# Function to preprocess the data
def preprocess_data(df):
    df = df.copy()
    df = df.rename(columns={'Vol.': 'volume'})
    df['volume'] = df['volume'].str.replace('M', '').astype(float) * 1000000
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    df['time_idx'] = range(len(df))
    df['group'] = 0
    return df

# Function to add technical indicators
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
    df.loc[1:, 'OBV'] = (df['Price'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()
    
    return df

def load_model():
    """Load the trained model and related objects"""
    with open('xgbm/xgbm_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open('xgbm/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    with open('xgbm/important_features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return {
        'xgb_model': xgb_model,
        'scaler_y': scaler_y,
        'features': features
    }

def retrain_model(file_path):
    """Simplified training function for demonstration"""
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    
    # Load and process data
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = df.dropna()
    
    # Define target and features
    target = 'Change %'
    
    # Basic features - keep it simple but effective
    basic_features = [
        'Open', 'High', 'Low', 'Price', 'volume',
        'MA5', 'MA20', 'RSI', 'ROC',
        'close_lag_1', 'close_lag_2', 'close_lag_3',
        'volume_lag_1', 'rsi_lag_1', 'Change_lag_1'
    ]
    
    # Check which features are available
    features = [f for f in basic_features if f in df.columns]
    print(f"Using these features: {features}")
    
    # Split data
    X = df[features]
    y = df[target].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale target
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Train model
    xgb_model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        objective="reg:squarederror",
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train_scaled)
    
    # Evaluate model
    preds_scaled = xgb_model.predict(X_test).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(np.abs(y_test), np.abs(preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Model training results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save model
    if not os.path.exists('xgb'):
        os.makedirs('xgb')
    
    with open('xgb/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open('xgb/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    with open('xgb/important_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    return {
        'xgb_model': xgb_model,
        'scaler_y': scaler_y,
        'features': features
    }

def predict_future_prices(model_dict, data_file, days_to_predict=5):
    """Predict future stock prices with proper feature updates"""
    # Load and process data
    df = pd.read_csv(data_file)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = df.dropna()
    
    # Extract models and parameters
    xgb_model = model_dict['xgb_model']
    scaler_y = model_dict['scaler_y']
    features = model_dict['features']
    
    print(f"Using features for prediction: {features}")
    
    # Get the last date and create future dates
    last_date = df['Date'].iloc[-1]
    current_date = last_date
    future_dates = []
    
    for _ in range(days_to_predict):
        current_date = current_date + timedelta(days=1)
        # Skip weekends
        while current_date.weekday() > 4:  # 5=Sat, 6=Sun
            current_date = current_date + timedelta(days=1)
        future_dates.append(current_date)
    
    # Create a dataframe to store future data
    future_df = pd.DataFrame({
        'Date': future_dates,
        'formatted_date': [d.strftime('%d-%m-%Y') for d in future_dates]
    })
    
    # Get the latest price
    last_price = df['Price'].iloc[-1]
    
    # Save a copy of the original dataframe for simulation
    sim_df = df.copy()
    
    # Iterate through future dates and make predictions
    predicted_prices = []
    for i, date in enumerate(future_dates):
        # Get the features for the current prediction
        X = sim_df[features].iloc[-1:].values
        
        # Predict percentage change
        pred_change_scaled = xgb_model.predict(X).reshape(-1, 1)
        pred_change = scaler_y.inverse_transform(pred_change_scaled)[0, 0]
        
        # Calculate the new price based on percentage change
        if i == 0:
            new_price = last_price * (1 + pred_change/100)
        else:
            new_price = predicted_prices[-1] * (1 + pred_change/100)
            
        predicted_prices.append(new_price)
        
        # Create a new row for the next prediction
        new_row = pd.DataFrame({
            'Date': [date],
            'Price': [new_price],
            'Open': [new_price],  # Simplified assumptions
            'High': [new_price * 1.01],
            'Low': [new_price * 0.99],
            'volume': [sim_df['volume'].iloc[-1]],
            'Change %': [pred_change]
        })
        
        # Add the new row to the simulation dataframe
        sim_df = pd.concat([sim_df, new_row], ignore_index=True)
        
        # Update technical indicators and lagged features for the next prediction
        sim_df = add_technical_indicators(sim_df)
        sim_df = add_lagged_features(sim_df)
    
    # Add predictions to the future dataframe
    future_df['predicted_price'] = predicted_prices
    
    return future_df

def evaluate_model(model_dict, test_file):
    """Evaluate the model using test data"""
    # Load and process data
    df = pd.read_csv(test_file)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = df.dropna()
    
    # Extract models and parameters
    xgb_model = model_dict['xgb_model']
    scaler_y = model_dict['scaler_y']
    features = model_dict['features']
    
    # Prepare the data for evaluation
    X = df[features].values
    y = df['Change %'].values.reshape(-1, 1)
    
    # Make predictions
    preds_scaled = xgb_model.predict(X).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    # Handle division by zero in MAPE calculation
    mape = mean_absolute_percentage_error(
        np.abs(y) + 0.1,  # Add small constant to avoid division by zero
        np.abs(preds) + 0.1
    )
    
    r2 = r2_score(y, preds)
    
    metrics = {
        'xgb': {
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    }
    
    # Print metrics
    for model_name, model_metrics in metrics.items():
        print(f"=== {model_name.upper()} Model Metrics ===")
        print(f"RMSE: {model_metrics['rmse']:.4f}")
        print(f"MAPE: {model_metrics['mape']:.4f}")
        print(f"R²: {model_metrics['r2']:.4f}")
    
    # Calculate predicted prices for visualization
    df['Predicted_Change'] = preds.flatten()
    df['Predicted_Price'] = df['Price'].shift(1) * (1 + df['Predicted_Change']/100)
    df = df.dropna()
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Actual', linewidth=1)
    plt.plot(df['Date'], df['Predicted_Price'], label='Predicted', linewidth=1, alpha=0.7)
    plt.title('Stock Price - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_price_evaluation_new.png')
    
    # Plot change percentages
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Change %'], label='Actual %', linewidth=1)
    plt.plot(df['Date'], df['Predicted_Change'], label='Predicted %', linewidth=1, alpha=0.7)
    plt.title('Stock Price Change % - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Change %')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_change_evaluation_new.png')
    
    return metrics

if __name__ == '__main__':
    try:
        # Option 1: Load existing model
        try:
            print("Attempting to load existing model...")
            model_dict = load_model()
            print("Model loaded successfully!")
        except:
            # Option 2: Retrain model if loading fails
            print("No existing model found or loading failed. Retraining...")
            #model_dict = retrain_model('ZOMT.csv')
            #print("Model retrained and saved successfully!")
        
        # Make predictions
        predictions = predict_future_prices(model_dict, 'test_data.csv', days_to_predict=33)
        print("Future Predictions:")
        print(predictions)
        
        # Plot future predictions
        plt.figure(figsize=(12, 6))
        plt.plot(predictions['Date'], predictions['predicted_price'], marker='o')
        plt.title('Stock Price - 5-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('Predicted Stock Price')
        plt.xticks(predictions['Date'], predictions['formatted_date'], rotation=45)
        plt.tight_layout()
        plt.savefig('stock_forecast.png')
        plt.show()
        
        # Evaluate the model
        metrics = evaluate_model(model_dict, 'som.csv')
        
    except FileNotFoundError:
        print("Error: Model files not found. Please run the training script first to create the models.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")