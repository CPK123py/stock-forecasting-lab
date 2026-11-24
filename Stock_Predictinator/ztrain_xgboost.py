import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import os
import shap


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

# Function to add lagged features
def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['RSI'].shift(lag)
    return df

# Function to evaluate feature importance
def select_important_features(model, feature_names, threshold=0.01):
    feature_importance = model.feature_importances_
    important_idx = np.where(feature_importance > threshold)[0]
    important_features = [feature_names[i] for i in important_idx]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return important_features, feature_importance

# Main training function
def train_hybrid_model(file_path, sequence_length=60, test_size=0.2):
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = df.dropna()

    # Transform target variable to percentage change
    df['Price_change'] = df['Price'].pct_change()
    df['Price_change'] = df['Price_change'].fillna(0)

    features = ["Price", "Open", "High", "Low", "volume", "Change %",
                "MA5", "MA20", "MA50", "RSI", "MACD", "MACD_signal",
                "BB_middle", "BB_upper", "BB_lower", "ROC", "OBV"]

    # Add lagged features
    lagged_features = [f'close_lag_{lag}' for lag in [1, 2, 3, 5, 10]]
    lagged_features += [f'volume_lag_{lag}' for lag in [1, 2, 3, 5, 10]]
    lagged_features += [f'rsi_lag_{lag}' for lag in [1, 2, 3, 5, 10]]
    features = features + lagged_features

    train_size = int(len(df) * (1 - test_size))
    train_data = df[:train_size].copy()
    test_data = df[train_size:].copy()

    train_dates = train_data['Date'].values
    test_dates = test_data['Date'].values

    # Scale the target variable
    scaler_y = MinMaxScaler()
    train_y = scaler_y.fit_transform(train_data[['Price_change']])
    test_y = scaler_y.transform(test_data[['Price_change']])

    # Train XGBoost model
    print("Training preliminary XGBoost model for feature selection...")
    initial_xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective="reg:squarederror",
        random_state=42
    )
    initial_xgb_model.fit(train_data[features], train_y)

    # Select important features
    important_features, feature_importance = select_important_features(
        initial_xgb_model, features, threshold=0.01
    )
    print(f"Selected important features: {important_features}")

    # Train final XGBoost model
    print("Training final XGBoost model...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        objective="reg:squarederror",
        random_state=42
    )
    xgb_model.fit(train_data[important_features], train_y)

    # Make predictions
    xgb_preds = xgb_model.predict(test_data[important_features])
    xgb_preds_original = scaler_y.inverse_transform(xgb_preds.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(test_y.reshape(-1, 1))

    # Evaluate model
    xgb_rmse = np.sqrt(mean_squared_error(y_test_original, xgb_preds_original))
    xgb_mape = mean_absolute_percentage_error(y_test_original, xgb_preds_original)
    xgb_r2 = r2_score(y_test_original, xgb_preds_original)

    metrics = {
        'xgb': {
            'rmse': xgb_rmse,
            'mape': xgb_mape,
            'r2': xgb_r2
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
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_original, label='Actual', color='black', linewidth=2)
    plt.plot(test_dates, xgb_preds_original, label='XGBoost', color='green', linewidth=1, alpha=0.7)
    plt.title('Stock Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Stock Price Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_prediction.png')

    # Save models
    if not os.path.exists('xgbm'):
        os.makedirs('xgbm')
    with open('xgbm/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open('xgbm/important_features.pkl', 'wb') as f:
        pickle.dump(important_features, f)
    with open('xgbm/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    print("Models and related objects saved successfully!")
    return xgb_model, metrics, train_data, important_features

if __name__ == '__main__':
    file_path = 'train_data.csv'
    xgb_model, metrics, train_data, important_features = train_hybrid_model(file_path)
    # Explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(train_data[important_features])

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, train_data[important_features], plot_type="bar")
    plt.savefig('shap_summary_plot.png')
    plt.close()

    # Plot SHAP dependence plots for top features
    for feature in important_features:
        shap.dependence_plot(feature, shap_values, train_data[important_features])
        plt.savefig(f'shap_dependence_{feature}.png')
        plt.close()