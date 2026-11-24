import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# Define the preprocessing function
def preprocess_data(df):
    df = df.copy()
    df = df.rename(columns={'Vol.': 'volume'})
    df['volume'] = df['volume'].str.replace('M', '').astype(float) * 1000000
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    
    # Extract temporal features
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(str)  # Convert to string
    df['month'] = df['Date'].dt.month.astype(str)  # Convert to string
    df['quarter'] = df['Date'].dt.quarter.astype(str)  # Convert to string
    df['year'] = df['Date'].dt.year
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    
    # Index for time series
    df['time_idx'] = range(len(df))
    df['group'] = 0
    return df

# Define the function to add technical indicators
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
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['Price'].rolling(window=20).mean()
    std_dev = df['Price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct_b'] = (df['Price'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Price Rate of Change
    df['ROC'] = df['Price'].pct_change(periods=5) * 100
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[1:, 'OBV'] = (df['Price'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Price'].shift())
    low_close = abs(df['Low'] - df['Price'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Price momentum
    df['momentum_1d'] = df['Price'].pct_change(periods=1)
    df['momentum_5d'] = df['Price'].pct_change(periods=5)
    df['momentum_20d'] = df['Price'].pct_change(periods=20)
    
    # Volatility features
    df['volatility_5d'] = df['Price'].rolling(window=5).std() / df['Price'].rolling(window=5).mean()
    df['volatility_20d'] = df['Price'].rolling(window=20).std() / df['Price'].rolling(window=20).mean()
    
    # Price gap
    df['gap'] = df['Open'] - df['Price'].shift(1)
    df['gap_pct'] = df['gap'] / df['Price'].shift(1) * 100
    
    # High-Low range
    df['hl_range'] = (df['High'] - df['Low']) / df['Price'] * 100
    
    return df


# Load the dataset
print("Loading and preprocessing data...")
df = pd.read_csv('ZOMT.csv')

# Preprocess the data
processed_df = preprocess_data(df)
processed_df = add_technical_indicators(processed_df)
processed_df = processed_df.dropna()

# Reset the time index after dropping NaN values
processed_df['time_idx'] = range(len(processed_df))

# Define the parameters
max_prediction_length = 5  # Forecast window (5 days)
max_encoder_length = 60  # Look-back window (60 days)

# Create the training dataset
print("Creating TimeSeriesDataSet...")
training = TimeSeriesDataSet(
    data=processed_df,
    time_idx="time_idx",
    target="Price",
    group_ids=["group"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=["day_of_week", "month", "quarter"],
    time_varying_known_reals=["time_idx", "day_of_month", "week_of_year", "year"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "Price", "Open", "High", "Low", "volume", "Change %",
        "MA5", "MA20", "MA50", "RSI", "MACD", "MACD_signal", "MACD_hist",
        "BB_middle", "BB_upper", "BB_lower", "BB_width", "BB_pct_b",
        "ROC", "OBV", "ATR", "momentum_1d", "momentum_5d", "momentum_20d",
        "volatility_5d", "volatility_20d", "gap", "gap_pct", "hl_range"
    ],
    target_normalizer=GroupNormalizer(
        groups=["group"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# ====== IMPORTANT: USE MODEL STATE DICT INSTEAD OF CHECKPOINT ======
# Try loading the model from the state dict file instead
print("Loading model from state dict...")
try:
    # Check if state dict file exists
    state_dict_path = 'tft_stock_model_3.pth'
    
    # Initialize the model with the same parameters as the trained model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=64,  # Make sure these match your training parameters
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
    )
    
    # Load state dict
    tft.load_state_dict(torch.load(state_dict_path))
    print("Model loaded successfully from state dict!")
    
except FileNotFoundError:
    print(f"State dict file not found. Please ensure 'tft_stock_model.pth' exists.")
    print("Attempting alternative loading method...")
    
    try:
        # Try loading from checkpoint directly but with map_location
        checkpoint_path = 'tft_stock_model_checkpoint.ckpt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Initialize the model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.2,
            hidden_continuous_size=32,
            loss=QuantileLoss(),
        )
        
        # Manually extract and load state dict
        if "state_dict" in checkpoint:
            # Filter out keys that don't belong to the model
            model_state_dict = {k.replace("tft.", ""): v for k, v in checkpoint["state_dict"].items() 
                              if k.startswith("tft.")}
            tft.load_state_dict(model_state_dict)
            print("Model loaded successfully from checkpoint!")
        else:
            print("Could not find state_dict in checkpoint!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)


# Function to predict future values
def predict_future(model, df, days_to_predict=5):
    # Prepare the latest data
    encoder_data = df.iloc[-max_encoder_length:].copy()
    
    # Create time indices for prediction
    last_time_idx = df['time_idx'].max()
    future_time_idx = range(last_time_idx + 1, last_time_idx + days_to_predict + 1)
    
    # Create prediction dataframe
    future_data = pd.DataFrame({
        'time_idx': future_time_idx,
        'group': [0] * days_to_predict
    })
    
    # Add dates
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    future_data['Date'] = future_dates
    
    # Add temporal features
    future_data['day_of_week'] = future_data['Date'].dt.dayofweek.astype(str)
    future_data['month'] = future_data['Date'].dt.month.astype(str)
    future_data['quarter'] = future_data['Date'].dt.quarter.astype(str)
    future_data['year'] = future_data['Date'].dt.year
    future_data['day_of_month'] = future_data['Date'].dt.day
    future_data['week_of_year'] = future_data['Date'].dt.isocalendar().week
    
    # Fill in missing columns with zeros or appropriate values
    for col in df.columns:
        if col not in future_data.columns and col not in ['Date', 'time_idx', 'group']:
            if col == 'Price':
                # Use the last known price as the starting point
                future_data[col] = df[col].iloc[-1]
            elif col in ['Open', 'High', 'Low']:
                # Use the last known values as the starting point
                future_data[col] = df[col].iloc[-1]
            elif col == 'volume':
                # Use the average of the last 5 days
                future_data[col] = df[col].iloc[-5:].mean()
            elif col == 'Change %':
                # Start with 0 change
                future_data[col] = 0.0
            else:
                # Fill technical indicators with the last values
                future_data[col] = df[col].iloc[-1] if col in df.columns else 0.0
    
    # Create combined dataset
    prediction_data = pd.concat([encoder_data, future_data])
    
    # Create a unique dataset for prediction
    batch_size = 128  # Larger batch size for prediction
    
    # Move model to CPU for prediction
    model.to("cpu")
    
    try:
        # Create prediction dataset
        prediction_dataset = TimeSeriesDataSet.from_dataset(training, prediction_data, predict=True)
        prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        
        # Make predictions - in raw mode first to debug
        print("Making predictions...")
        raw_predictions = model.predict(prediction_dataloader, mode="raw", return_x=True)
        print(f"Raw predictions shape: {raw_predictions.output.shape}")
        
        # Now get the actual predictions
        predictions = model.predict(prediction_dataloader)
        print(f"Predictions shape: {predictions.shape}")
        
        # Create result dataframe
        result = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions.numpy().flatten()
        })
        
        return result
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'Date': future_dates, 'Predicted_Price': [None] * len(future_dates)})

# Make future predictions
print("\nMaking future predictions...")
try:
    future_predictions = predict_future(tft, processed_df, days_to_predict=10)
    print("Future Price Predictions:")
    print(future_predictions)

    # Plot the future predictions alongside historical data
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(processed_df['Date'].iloc[-30:], processed_df['Price'].iloc[-30:], 'b-', label='Historical Price')

    # Plot predictions
    plt.plot(future_predictions['Date'], future_predictions['Predicted_Price'], 'r--', label='Predicted Price')
    plt.fill_between(
        future_predictions['Date'],
        future_predictions['Predicted_Price'] * 0.95,  # Lower bound (5% below prediction)
        future_predictions['Predicted_Price'] * 1.05,  # Upper bound (5% above prediction)
        color='r', alpha=0.2
    )

    plt.title('Stock Price: Historical vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_forecast.png')
    plt.show()
    
    # Print summary statistics
    current_price = processed_df['Price'].iloc[-1]
    future_min = future_predictions['Predicted_Price'].min()
    future_max = future_predictions['Predicted_Price'].max()
    future_mean = future_predictions['Predicted_Price'].mean()
    
    print(f"\nCurrent Price: {current_price:.2f}")
    print(f"Predicted Price Range: {future_min:.2f} - {future_max:.2f}")
    print(f"Average Predicted Price: {future_mean:.2f}")
except Exception as e:
    print(e)