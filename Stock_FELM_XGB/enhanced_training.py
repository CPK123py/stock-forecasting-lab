"""
Enhanced XGBoost Training Script for Real-Time Stock Prediction
Optimized for faster training and better real-time performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Preprocess the data"""
    df = df.copy()
    
    # Handle volume column
    if 'Vol.' in df.columns:
        df = df.rename(columns={'Vol.': 'volume'})
        if df['volume'].dtype == 'object':
            df['volume'] = df['volume'].str.replace('M', '').str.replace('K', '')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce') * 1000000
    
    # Handle date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Handle change percentage
    if 'Change %' in df.columns:
        if df['Change %'].dtype == 'object':
            df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    else:
        # Calculate change percentage if not available
        df['Change %'] = ((df['Price'] - df['Open']) / df['Open'] * 100)
    
    # Add time index
    df['time_idx'] = range(len(df))
    df['group'] = 0
    
    # Remove any remaining NaN in critical columns
    df = df.dropna(subset=['Price', 'Open', 'High', 'Low', 'volume'])
    
    return df

def add_technical_indicators(df):
    """Add technical indicators with error handling"""
    df = df.copy()
    
    # Moving Averages
    df['MA5'] = df['Price'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['Price'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['Price'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Price'].rolling(window=50, min_periods=1).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Price'].ewm(span=26, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Price'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['Price'].rolling(window=20, min_periods=1).mean()
    bb_std = df['Price'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    
    # Price Rate of Change
    df['ROC'] = df['Price'].pct_change(periods=5) * 100
    df['ROC_10'] = df['Price'].pct_change(periods=10) * 100
    
    # Volume indicators
    df['Volume_MA5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['Volume_ratio'] = df['volume'] / (df['Volume_MA5'] + 1)
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Price'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Price'].shift())
    low_close = np.abs(df['Low'] - df['Price'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    
    # Price position relative to daily range
    df['Price_position'] = (df['Price'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # Momentum
    df['Momentum'] = df['Price'] - df['Price'].shift(10)
    
    return df

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """Add lagged features for time series"""
    df = df.copy()
    
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['RSI'].shift(lag)
        df[f'change_lag_{lag}'] = df['Change %'].shift(lag)
        df[f'macd_lag_{lag}'] = df['MACD'].shift(lag)
    
    return df

def add_time_features(df):
    """Add time-based features"""
    df = df.copy()
    
    if 'Date' in df.columns:
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Market opening/closing effects
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    return df

def select_features(df):
    """Select the most important features"""
    # Core features
    core_features = ['Open', 'High', 'Low', 'Price', 'volume']
    
    # Technical indicators
    technical_features = [
        'MA5', 'MA20', 'RSI', 'MACD', 'MACD_signal', 
        'ROC', 'ATR', 'BB_width', 'Volume_ratio'
    ]
    
    # Lagged features
    lag_features = [
        'close_lag_1', 'close_lag_2', 'close_lag_3',
        'volume_lag_1', 'rsi_lag_1', 'change_lag_1'
    ]
    
    # Combine all features
    all_features = core_features + technical_features + lag_features
    
    # Filter features that exist in dataframe
    available_features = [f for f in all_features if f in df.columns]
    
    return available_features

def train_model(file_path, output_dir='xgb'):
    """Train XGBoost model with enhanced features"""
    print("="*60)
    print("ENHANCED REAL-TIME STOCK PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    print(f"   Loaded {len(df)} rows")
    
    df = preprocess_data(df)
    print(f"   After preprocessing: {len(df)} rows")
    
    # Add features
    print("\n[2/6] Engineering features...")
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = add_time_features(df)
    
    # Remove NaN values
    initial_len = len(df)
    df = df.dropna()
    print(f"   Removed {initial_len - len(df)} rows with NaN")
    print(f"   Final dataset: {len(df)} rows")
    
    if len(df) < 100:
        raise ValueError("Not enough data after preprocessing. Need at least 100 rows.")
    
    # Select features
    print("\n[3/6] Selecting features...")
    features = select_features(df)
    print(f"   Using {len(features)} features:")
    for i, feat in enumerate(features, 1):
        print(f"      {i}. {feat}")
    
    # Prepare data
    target = 'Change %'
    X = df[features].values
    y = df[target].values.reshape(-1, 1)
    
    print(f"\n   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Time series split for validation
    print("\n[4/6] Splitting data with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Scale target
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Train model with cross-validation
    print("\n[5/6] Training XGBoost model...")
    
    # Model configuration optimized for real-time
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation scores
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_scaled[train_idx], y_scaled[val_idx]
        
        xgb_model.fit(X_train_fold, y_train_fold.ravel())
        
        preds_fold = xgb_model.predict(X_val_fold).reshape(-1, 1)
        preds_fold_orig = scaler_y.inverse_transform(preds_fold)
        y_val_orig = scaler_y.inverse_transform(y_val_fold)
        
        r2_fold = r2_score(y_val_orig, preds_fold_orig)
        cv_scores.append(r2_fold)
        print(f"   Fold {fold} R²: {r2_fold:.4f}")
    
    print(f"   Average CV R²: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Train final model on all data
    print("\n   Training final model on all data...")
    xgb_model.fit(X, y_scaled.ravel())
    
    # Evaluate on full dataset
    print("\n[6/6] Evaluating model performance...")
    preds_scaled = xgb_model.predict(X).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = np.mean(np.abs(y - preds))
    mape = mean_absolute_percentage_error(np.abs(y), np.abs(preds))
    r2 = r2_score(y, preds)
    
    # Direction accuracy
    y_direction = (y > 0).astype(int).flatten()
    pred_direction = (preds > 0).astype(int).flatten()
    direction_accuracy = accuracy_score(y_direction, pred_direction)
    
    # Metrics for positive predictions
    positive_mask = pred_direction == 1
    if positive_mask.sum() > 0:
        precision = precision_score(y_direction, pred_direction)
    else:
        precision = 0.0
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"RMSE:                 {rmse:.4f}")
    print(f"MAE:                  {mae:.4f}")
    print(f"MAPE:                 {mape:.4f}")
    print(f"R² Score:             {r2:.4f}")
    print(f"Direction Accuracy:   {direction_accuracy*100:.2f}%")
    print(f"Precision (Buy):      {precision*100:.2f}%")
    print("="*60)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s} : {row['importance']:.4f}")
    
    # Save model and artifacts
    print(f"\n[SAVING] Saving model to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open(f'{output_dir}/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    with open(f'{output_dir}/important_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    with open(f'{output_dir}/feature_importance.pkl', 'wb') as f:
        pickle.dump(feature_importance, f)
    
    print("   ✓ Model saved")
    print("   ✓ Scaler saved")
    print("   ✓ Features saved")
    print("   ✓ Feature importance saved")
    
    # Plot predictions vs actual
    print("\n[PLOTTING] Creating visualizations...")
    
    # Calculate predicted prices
    df_plot = df.copy()
    df_plot['Predicted_Change'] = preds.flatten()
    df_plot['Predicted_Price'] = df_plot['Price'].shift(1) * (1 + df_plot['Predicted_Change']/100)
    df_plot = df_plot.dropna()
    
    # Plot 1: Price comparison
    plt.figure(figsize=(15, 6))
    plt.plot(df_plot['Date'], df_plot['Price'], label='Actual Price', linewidth=2, alpha=0.8)
    plt.plot(df_plot['Date'], df_plot['Predicted_Price'], label='Predicted Price', linewidth=2, alpha=0.7)
    plt.title('Stock Price: Actual vs Predicted', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price_prediction.png', dpi=300)
    print("   ✓ Price prediction plot saved")
    
    # Plot 2: Change percentage comparison
    plt.figure(figsize=(15, 6))
    plt.plot(df_plot['Date'], df_plot['Change %'], label='Actual Change %', linewidth=2, alpha=0.8)
    plt.plot(df_plot['Date'], df_plot['Predicted_Change'], label='Predicted Change %', linewidth=2, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.title('Price Change %: Actual vs Predicted', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Change %', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/change_prediction.png', dpi=300)
    print("   ✓ Change prediction plot saved")
    
    # Plot 3: Feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 15 Feature Importance', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
    print("   ✓ Feature importance plot saved")
    
    # Plot 4: Prediction error distribution
    plt.figure(figsize=(12, 5))
    errors = (preds - y).flatten()
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y, preds, alpha=0.5, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Change %', fontsize=12)
    plt.ylabel('Predicted Change %', fontsize=12)
    plt.title('Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_analysis.png', dpi=300)
    print("   ✓ Error analysis plot saved")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel files saved in '{output_dir}/' directory")
    print("You can now use these files for real-time prediction.")
    print("\nNext steps:")
    print("1. Start the Flask backend: python app.py")
    print("2. Open the dashboard in your browser")
    print("3. Configure stock symbol and start live trading")
    print("\n" + "="*60)
    
    return {
        'model': xgb_model,
        'scaler_y': scaler_y,
        'features': features,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'precision': precision
        },
        'feature_importance': feature_importance
    }

def quick_evaluate(model_path='xgb', test_file=None):
    """Quick evaluation of saved model"""
    print("\n" + "="*60)
    print("QUICK MODEL EVALUATION")
    print("="*60)
    
    # Load model
    print("\n[1/3] Loading saved model...")
    with open(f'{model_path}/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(f'{model_path}/scaler_y.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f'{model_path}/important_features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print("   ✓ Model loaded")
    print(f"   ✓ Using {len(features)} features")
    
    if test_file is None:
        print("\n   No test file provided. Evaluation complete.")
        return
    
    # Load test data
    print(f"\n[2/3] Loading test data from {test_file}...")
    df = pd.read_csv(test_file)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = add_lagged_features(df)
    df = add_time_features(df)
    df = df.dropna()
    
    print(f"   Loaded {len(df)} test samples")
    
    # Predict
    print("\n[3/3] Making predictions...")
    X = df[features].values
    y = df['Change %'].values.reshape(-1, 1)
    
    preds_scaled = model.predict(X).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    
    y_direction = (y > 0).astype(int).flatten()
    pred_direction = (preds > 0).astype(int).flatten()
    accuracy = accuracy_score(y_direction, pred_direction)
    
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(f"RMSE:                 {rmse:.4f}")
    print(f"R² Score:             {r2:.4f}")
    print(f"Direction Accuracy:   {accuracy*100:.2f}%")
    print("="*60)

if __name__ == '__main__':
    import sys
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python enhanced_training.py <data_file> [test_file]")
        print("\nExample:")
        print("  python enhanced_training.py ZOMT.csv")
        print("  python enhanced_training.py ZOMT.csv ZOMT2.csv")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Train model
    result = train_model(train_file)
    
    # Evaluate on test set if provided
    if test_file:
        print("\n" + "="*60)
        quick_evaluate('xgb', test_file)
    
    print("\n✅ All done! Model is ready for real-time trading.")
    print("\nTo start real-time predictions:")
    print("  python app.py")