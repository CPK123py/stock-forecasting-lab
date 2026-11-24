import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model  # type: ignore
import os

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['RSI'].shift(lag)
      #  df[f'Change_la{lag}'] =  df['Change %'].shift(lag)
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

def load_hybrid_model():

    with open('xgbm/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Load scaler
    with open('xgbm/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    # Load feature list
    with open('xgbm/important_features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return {
        'xgb_model': xgb_model,
        'scaler_y': scaler_y,
        'features': features
    }

def predict_future_prices(model_dict, data_file, days_to_predict=5):
    df = pd.read_csv(data_file)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = df.dropna()
    
    # Extract models and parameters
    xgb_model = model_dict['xgb_model']
    scaler_y = model_dict['scaler_y']
    features = model_dict['features']
    
    # Lists to store future data
    future_dates = []
    predictions = []
    
    # Get the last date and price
    last_date = df['Date'].iloc[-1]
    last_price = df['Price'].iloc[-1]
    
    # Create a copy of the last row as our starting point
    current_row = df.iloc[-1:].copy()
    
    # Generate future dates (skipping weekends)
    current_date = last_date
    for i in range(days_to_predict):
        # Move to next business day
        current_date = current_date + pd.Timedelta(days=1)
        while current_date.weekday() > 4:  # Skip weekends
            current_date = current_date + pd.Timedelta(days=1)
        
        # Store the date
        future_dates.append(current_date)
        
        # Prepare features for prediction
        X = current_row[features].values.reshape(1, -1)
        
        # Predict percentage change
        scaled_change_pred = xgb_model.predict(X).reshape(-1, 1)
        change_pred = scaler_y.inverse_transform(scaled_change_pred)[0, 0]
        
        # Calculate new price
        new_price = last_price * (1 + change_pred/100)
        predictions.append(new_price)
        last_price = new_price
        
        # Create a new row with updated values for the next prediction
        new_row = current_row.copy()
        new_row['Date'] = current_date
        new_row['Price'] = new_price
        
        # Update technical indicators
        # For simplicity, we'll update just the most critical ones
        # Moving averages - approximating with last few values
        if 'MA5' in new_row.columns:
            recent_prices = list(df['Price'].iloc[-4:]) + [new_price]
            new_row['MA5'] = sum(recent_prices) / 5
        
        if 'MA20' in new_row.columns:
            recent_prices = list(df['Price'].iloc[-19:]) + [new_price]
            new_row['MA20'] = sum(recent_prices) / 20
        
        # Update lagged features
        for lag in [1, 2, 3, 5, 10]:
            if f'close_lag_{lag}' in new_row.columns:
                if lag == 1:
                    new_row[f'close_lag_{lag}'] = last_price
                elif lag > 1 and i >= lag-1:
                    new_row[f'close_lag_{lag}'] = predictions[i-(lag-1)]
        
        # Update 'Change %' for the next prediction
        if i == 0:
            new_row['Change %'] = ((new_price / df['Price'].iloc[-1]) - 1) * 100
        else:
            new_row['Change %'] = ((new_price / predictions[i-1]) - 1) * 100
        
        # Update ROC if it's in features
        if 'ROC' in new_row.columns:
            if i >= 4:  # We have enough history
                new_row['ROC'] = ((new_price / predictions[i-4]) - 1) * 100
            else:
                # Use available history
                old_price = df['Price'].iloc[-5+i] if i < 5 else predictions[0]
                new_row['ROC'] = ((new_price / old_price) - 1) * 100
        
        # Add the new row to the dataframe for the next iteration
        df = pd.concat([df, new_row])
        current_row = new_row
    
    # Create a DataFrame for future predictions
    future_df = pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions
    })
    
    # Format the date column
    future_df['formatted_date'] = future_df['date'].dt.strftime('%d-%m-%Y')
    
    return future_df

def evaluate_model(model_dict, test_file):
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
    
    # Scale the target variable
    y_scaled = scaler_y.transform(y)
    
    # Make predictions
    xgb_preds_scaled = xgb_model.predict(X).reshape(-1, 1)
    xgb_preds = scaler_y.inverse_transform(xgb_preds_scaled)
    
    # Calculate metrics
    metrics = {
        'xgb': {
            'rmse': np.sqrt(mean_squared_error(y, xgb_preds)),
            'mape': mean_absolute_percentage_error(np.abs(y), np.abs(xgb_preds)),
            'r2': r2_score(y, xgb_preds)
        }
    }
    
    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Date': df['Date'],
        'Actual_Price': df['Price'],
        'Actual_Change': y.flatten(),
        'Predicted_Change': xgb_preds.flatten()
    })
    
    # Calculate predicted prices
    results_df['Predicted_Price'] = results_df['Actual_Price'].shift(1) * (1 + results_df['Predicted_Change']/100)
    results_df = results_df.dropna()
    
    # Print metrics
    for model_name, model_metrics in metrics.items():
        print(f"=== {model_name.upper()} Model Metrics ===")
        print(f"RMSE: {model_metrics['rmse']:.4f}")
        print(f"MAPE: {model_metrics['mape']:.4f}")
        print(f"R²: {model_metrics['r2']:.4f}")
        print()
    
    # Plot actual vs predicted changes
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], results_df['Actual_Change'], label='Actual Changes')
    plt.plot(results_df['Date'], results_df['Predicted_Change'], label='XGBoost Predicted Changes')
    plt.title('Stock Price Change (%) - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price Change (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_change_evaluation.png')
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], results_df['Actual_Price'], label='Actual Price')
    plt.plot(results_df['Date'], results_df['Predicted_Price'], label='XGBoost Predicted Price')
    plt.title('Stock Price - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_price_evaluation.png')
    
    return metrics, results_df

if __name__ == '__main__':
    try:
        model_dict = load_hybrid_model()
        print("Models loaded successfully!")
        
        # Make predictions
        predictions = predict_future_prices(model_dict, 'test_data.csv', days_to_predict=5)
        print("Future Predictions:")
        print(predictions)
        
        # Plot future predictions
        plt.figure(figsize=(12, 6))
        plt.plot(predictions['date'], predictions['predicted_price'], marker='o')
        plt.title('Stock Price - 5-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('Predicted Stock Price')
        plt.xticks(predictions['date'], predictions['formatted_date'], rotation=45)
        plt.tight_layout()
        plt.savefig('stock_forecast.png')
        plt.show()
        
        # Evaluate the model
        metrics = evaluate_model(model_dict, 'ZOMT.csv')
        
    except FileNotFoundError:
        print("Error: Model files not found. Please run the training script first to create the models.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")