import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score



#Function to preprocess the data
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

#Function to add technical indicators
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


#Function to add lagged features
def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['RSI'].shift(lag)
        df[f'Change_lag_{lag}'] = df['Change %'].shift(lag)
    return df


def train_model(file_path):
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
        X, y, test_size=0.3,random_state=42
    )
    
    # Scale target
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    

    y_train_class = (y_train_scaled > 0).astype(int).flatten()
    y_test_class = (y_test_scaled > 0).astype(int).flatten()

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
    
    preds_class = (preds_scaled > 0).astype(int).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(np.abs(y_test), np.abs(preds))
    r2 = r2_score(y_test, preds)

    accuracy = accuracy_score(y_test_class, preds_class)
    precision = precision_score(y_test_class, preds_class)
    
    print(f"Model training results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    
    
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



def evaluate_model(model_dict, test_file):
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
    
    y_scaled = scaler_y.transform(y)

    # Make predictions
    preds_scaled = xgb_model.predict(X).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    preds_class = (preds_scaled > 0).astype(int).flatten()
    y_class = (y_scaled > 0).astype(int).flatten()  
    

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    # Handle division by zero in MAPE calculation
    mape = mean_absolute_percentage_error(
        np.abs(y) + 0.1,  # Add small constant to avoid division by zero
        np.abs(preds) + 0.1
    )
    
    r2 = r2_score(y, preds)
    
    
    accuracy = accuracy_score(y_class, preds_class)
    precision = precision_score(y_class, preds_class)

    metrics = {
        'xgb': {
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'accuracy': accuracy,
            'precision': precision
        }
    }
    
    # Print metrics
    for model_name, model_metrics in metrics.items():
        print(f"=== {model_name.upper()} Trained Model Evaluation Metrics ===")
        print(f"RMSE: {model_metrics['rmse']:.4f}")
        print(f"MAPE: {model_metrics['mape']:.4f}")
        print(f"R²: {model_metrics['r2']:.4f}")
        print(f"Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"Precision: {model_metrics['precision']:.4f}")
    
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
    file_path = 'ZOMT.csv'
    xgb_model = train_model(file_path)
    evaluate_model(xgb_model, file_path)
