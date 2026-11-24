import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import os
from datetime import timedelta
import shap

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
    with open('xgb/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open('xgb/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    with open('xgb/important_features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    explainer = shap.Explainer(xgb_model)

    return {
        'xgb_model': xgb_model,
        'scaler_y': scaler_y,
        'features': features,
        'explainer': explainer
    }


def predict_future_prices(model_dict, data_file, days_to_predict=5):
    # Load and process data
    df = pd.read_csv(data_file)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = df.dropna()

    xgb_model = model_dict['xgb_model']
    scaler_y = model_dict['scaler_y']
    features = model_dict['features']
    explainer = model_dict['explainer'] 
    
    print(f"Using features for prediction: {features}")

    last_date = df['Date'].iloc[-1]
    current_date = last_date
    future_dates = []
    
    for _ in range(days_to_predict):
        current_date = current_date + timedelta(days=1)
        while current_date.weekday() > 4:  # 5=Sat, 6=Sun
            current_date = current_date + timedelta(days=1)
        future_dates.append(current_date)
    

    future_df = pd.DataFrame({
        'Date': future_dates,
        'formatted_date': [d.strftime('%d-%m-%Y') for d in future_dates]
    })
    
    last_price = df['Price'].iloc[-1]

    sim_df = df.copy()

    predicted_prices = []
    for i, date in enumerate(future_dates):
        X = sim_df[features].iloc[-1:].values
        
        pred_change_scaled = xgb_model.predict(X).reshape(-1, 1)
        pred_change = scaler_y.inverse_transform(pred_change_scaled)[0, 0]
        
        if i == 0:
            new_price = last_price * (1 + pred_change/100)
        else:
            new_price = predicted_prices[-1] * (1 + pred_change/100)
            
        predicted_prices.append(new_price)
        
  
        new_row = pd.DataFrame({
            'Date': [date],
            'Price': [new_price],
            'Open': [new_price],
            'High': [new_price * 1.01],
            'Low': [new_price * 0.99],
            'volume': [sim_df['volume'].iloc[-1]],
            'Change %': [pred_change]
        })

        sim_df = pd.concat([sim_df, new_row], ignore_index=True)
 
        sim_df = add_technical_indicators(sim_df)
        sim_df = add_lagged_features(sim_df)
    
   
    future_df['predicted_price'] = predicted_prices

    X = df[features].iloc[-1:].values  # Use the last row for explanation
    shap_values = explainer(X)

    # Plot SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.title('SHAP Summary Plot for Last Prediction')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    plt.show()

    # Plot SHAP force plot for the last prediction
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values.values[0], X[0], feature_names=features, matplotlib=True, show=False)
    plt.title('SHAP Force Plot for Last Prediction')
    plt.tight_layout()
    plt.savefig('shap_force_plot.png')
    plt.show()

    
    return future_df

def evaluate_model(model_dict, test_file):
   
    df = pd.read_csv(test_file)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = df.dropna()
    
  
    xgb_model = model_dict['xgb_model']
    scaler_y = model_dict['scaler_y']
    features = model_dict['features']
    explainer = model_dict['explainer'] 
   
    X = df[features].values
    y = df['Change %'].values.reshape(-1, 1)
    
    # Make predictions
    preds_scaled = xgb_model.predict(X).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    mape = mean_absolute_percentage_error(
        np.abs(y) + 0.1, 
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

    for model_name, model_metrics in metrics.items():
        print(f"=== {model_name.upper()} Model Metrics ===")
        print(f"RMSE: {model_metrics['rmse']:.4f}")
        print(f"MAPE: {model_metrics['mape']:.4f}")
        print(f"R²: {model_metrics['r2']:.4f}")
    
    df['Predicted_Change'] = preds.flatten()
    df['Predicted_Price'] = df['Price'].shift(1) * (1 + df['Predicted_Change']/100)
    df = df.dropna()
    
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
    

    shap_values = explainer(X)

    # Plot SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.title('SHAP Summary Plot for Test Data')
    plt.tight_layout()
    plt.savefig('shap_summary_plot_test.png')
    plt.show()

    # Plot SHAP force plot for the first prediction
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values.values[0], X[0], feature_names=features, matplotlib=True, show=False)
    plt.title('SHAP Force Plot for First Prediction')
    plt.tight_layout()
    plt.savefig('shap_force_plot_test.png')
    plt.show()

    return metrics

if __name__ == '__main__':
   
    
    print("Attempting to load existing model...")
    model_dict = load_model()
    print("Model loaded successfully!")
   
    predictions = predict_future_prices(model_dict, 'ZOMT2.csv', days_to_predict=13)
    print("Future Predictions:")
    print(predictions)
    
    plt.figure(figsize=(12, 6))
    plt.plot(predictions['Date'], predictions['predicted_price'], marker='o')
    plt.title('Stock Price - 5-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Predicted Stock Price')
    plt.xticks(predictions['Date'], predictions['formatted_date'], rotation=45)
    plt.tight_layout()
    plt.savefig('stock_forecast.png')
    plt.show()

    metrics = evaluate_model(model_dict, 'ZOMT2.csv')
    