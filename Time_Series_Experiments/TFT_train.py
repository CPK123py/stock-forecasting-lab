import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer

class LightningTFT(pl.LightningModule):
    def __init__(self, tft):
        super().__init__()
        self.tft = tft
        self.loss_function = QuantileLoss()

    def forward(self, x):
        return self.tft(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # y_hat is a tuple, extract the first element
        loss = self.loss_function(y_hat[0], y)  # Use y_hat[0] instead of y_hat
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # y_hat is a tuple, extract the first element
        loss = self.loss_function(y_hat[0], y)  # Use y_hat[0] instead of y_hat
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    
df = pd.read_csv('ZOMT.csv')


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

# Enhanced technical indicators function
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


# Preprocessing steps
processed_df = preprocess_data(df)
processed_df = add_technical_indicators(processed_df)
processed_df = processed_df.dropna()

# Reset the time index after dropping NaN values
processed_df['time_idx'] = range(len(processed_df))

# Define the training and validation cutoffs
max_prediction_length = 5  # Forecast window (5 days)
max_encoder_length = 60    # Look-back window (increased from 30 to 60 days)

# Make sure we have enough data
if len(processed_df) <= max_encoder_length + max_prediction_length:
    raise ValueError(f"Not enough data points. Need at least {max_encoder_length + max_prediction_length + 1} points after preprocessing.")

# Use a larger portion of the data for training
train_split = 0.8
training_cutoff = int(len(processed_df) * train_split)

# Create the training dataset
training = TimeSeriesDataSet(
    data=processed_df[processed_df['time_idx'] < training_cutoff],
    time_idx="time_idx",
    target="Price",
    group_ids=["group"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=["day_of_week", "month", "quarter"],  # Ensure these are strings
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
# Create the validation dataset using the same parameters
validation = TimeSeriesDataSet.from_dataset(training, processed_df, min_prediction_idx=training_cutoff)

# Create data loaders for model training
batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Initialize the TFT model
pl.seed_everything(42)

# Initialize the TFT model with more complexity
tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=64,  # Increased from 32 to 64
    attention_head_size=4,  # Increased from 1 to 4
    dropout=0.2,  # Increased from 0.1 to 0.2
    hidden_continuous_size=32,  # Increased from 16 to 32
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=5,
    learning_rate=0.001,
)

# Wrap TFT in a LightningModule
lit_model = LightningTFT(tft)

# Configure training
early_stop_callback = EarlyStopping(monitor="val_loss", patience=15, min_delta=0.001, mode="min")
lr_logger = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs=50,  # Increased from 30 to 50
    accelerator="auto",
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback, lr_logger],
    limit_train_batches=100,  # Increased from 50 to 100
)

lit_model.trainer = trainer
# Train the model
trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Save the model - both full model and state dict
model_checkpoint_path = 'tft_stock_model_checkpoint_3.ckpt'
trainer.save_checkpoint(model_checkpoint_path)
print(f"Model checkpoint saved to {model_checkpoint_path}")



# Save the model state dict (optional)
torch.save(tft.state_dict(), 'tft_stock_model_state_dict.pth')

model_state_dict_path = 'tft_stock_model_3.pth'
torch.save(tft.state_dict(), model_state_dict_path)
print(f"Model state dict saved to {model_state_dict_path}")

# Load best model for predictions
best_tft = tft
best_tft.to("cpu")

# Create predictions for the validation period
predictions = best_tft.predict(val_dataloader, return_x=True, trainer_kwargs=dict(accelerator="cpu"))

# Visualize predictions
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))

def plot_predictions():
    for idx in range(min(5, len(predictions[0]))):  # Plot up to 5 examples
        best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
        plt.show()

plot_predictions()

# Calculate performance metrics
actuals_list = []
for x, y in iter(val_dataloader):
    # Extract the target tensor from y
    if isinstance(y, tuple):
        actuals_list.append(y[0])
    elif isinstance(y, dict):
        actuals_list.append(y['target'])
    else:
        actuals_list.append(y)

# Concatenate all target tensors
actuals = torch.cat(actuals_list)

# Make predictions
predictions = best_tft.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"))

# Calculate Mean Absolute Error
mean_absolute_error = (actuals - predictions).abs().mean()
print(f"Mean Absolute Error: {mean_absolute_error}")

# Calculate Mean Absolute Percentage Error
mape = ((actuals - predictions).abs() / actuals).mean() * 100
print(f"Mean Absolute Percentage Error: {mape}%")

# Calculate Root Mean Square Error
rmse = torch.sqrt(((actuals - predictions) ** 2).mean())
print(f"Root Mean Square Error: {rmse}")

# Function to make future predictions
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
    future_data['day_of_week'] = future_data['Date'].dt.dayofweek
    future_data['month'] = future_data['Date'].dt.month
    future_data['quarter'] = future_data['Date'].dt.quarter
    future_data['year'] = future_data['Date'].dt.year
    future_data['day_of_month'] = future_data['Date'].dt.day
    future_data['week_of_year'] = future_data['Date'].dt.isocalendar().week
    
    # Fill in missing columns with zeros or appropriate values
    for col in processed_df.columns:
        if col not in future_data.columns and col not in ['Date', 'time_idx', 'group']:
            if col == 'Price':
                # Use the last known price as the starting point
                future_data[col] = processed_df[col].iloc[-1]
            elif col in ['Open', 'High', 'Low']:
                # Use the last known values as the starting point
                future_data[col] = processed_df[col].iloc[-1]
            elif col == 'volume':
                # Use the average of the last 5 days
                future_data[col] = processed_df[col].iloc[-5:].mean()
            elif col == 'Change %':
                # Start with 0 change
                future_data[col] = 0.0
            else:
                # Fill technical indicators with the last values
                future_data[col] = processed_df[col].iloc[-1] if col in processed_df.columns else 0.0
    
    # Create combined dataset
    prediction_data = pd.concat([encoder_data, future_data])
    
    # Use the model to predict
    prediction_dataset = TimeSeriesDataSet.from_dataset(training, prediction_data, predict=True)
    prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=batch_size)
    
    predictions = model.predict(prediction_dataloader)
    
    # Create result dataframe
    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions.numpy().flatten()
    })
    
    return result

# Make future predictions
try:
    future_predictions = predict_future(best_tft, processed_df, days_to_predict=10)  # Increased to 10 days
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
    
except Exception as e:
    print(f"Error making future predictions: {e}")


# Feature importance analysis
def plot_feature_importance():
    # Get feature importance from the model
    interpretation = best_tft.interpret_output