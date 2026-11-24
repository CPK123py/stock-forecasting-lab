import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
from tensorflow.keras.models import load_model  # type: ignore

class LightningTFT(pl.LightningModule):
    def __init__(self, tft):
        super().__init__()
        self.tft = tft
        self.loss_function = QuantileLoss()

    def forward(self, x):
        return self.tft(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) 
        loss = self.loss_function(y_hat[0], y) 
        self.log("train_loss", loss)
       
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat[0], y) 
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
def preprocess_data(df):
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Rename volume column
    df = df.rename(columns={'Vol.': 'volume'})

    # Convert volume format (e.g., "1.5M" to 1500000)
    df['volume'] = df['volume'].str.replace('M', '').astype(float) * 1000000

    # Convert date with dayfirst=True
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Sort by date
    df = df.sort_values('Date')
   
    # Clean Change % column
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

    # Create time index
    df['time_idx'] = range(len(df))
    
    # Add group ID
    df['group'] = 0
    
    return df






def predict_future(model, df, days_to_predict=5):
        encoder_data = df.iloc[-max_encoder_length:].copy()
        last_time_idx = df['time_idx'].max()
        future_time_idx = range(last_time_idx + 1, last_time_idx + days_to_predict + 1)
        future_data = pd.DataFrame({
            'time_idx': future_time_idx,
            'group': [0] * days_to_predict
        })
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
        future_data['Date'] = future_dates
        for col in processed_df.columns:
            if col not in future_data.columns and col not in ['Date', 'time_idx', 'group']:
                if col == 'Price':
                    future_data[col] = processed_df[col].iloc[-1]
                else:
                    future_data[col] = 0.0
        prediction_data = pd.concat([encoder_data, future_data])

        prediction_dataset = TimeSeriesDataSet.from_dataset(training, prediction_data, predict=True)
        prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=batch_size,num_workers=7)

        predictions = model.predict(prediction_dataloader)
        result = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions.numpy().flatten()
        })
        return result



df = pd.read_csv('ZOMT.csv')
processed_df = preprocess_data(df)
processed_df = processed_df.dropna()
processed_df['time_idx'] = range(len(processed_df))

print(processed_df['time_idx'].duplicated().sum())  # Number of duplicates
print(processed_df[processed_df.duplicated(subset=['time_idx'], keep=False)])  # Show duplicate rows
processed_df = processed_df.drop_duplicates(subset=['time_idx']).reset_index(drop=True)
processed_df['time_idx'] = processed_df.groupby("group").cumcount()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    max_prediction_length = 5
    max_encoder_length = 30
    training_cutoff = processed_df['time_idx'].max() - max_prediction_length
    training = TimeSeriesDataSet(
    data=processed_df[processed_df['time_idx'] <= training_cutoff],
    time_idx="time_idx",
    target="Price",
    group_ids=["group"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
    "Price", "Open", "High", "Low", "volume", "Change %"
    ],
    target_normalizer=GroupNormalizer(
    groups=["group"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, processed_df, min_prediction_idx=training_cutoff + 1)

    batch_size = 32
    num_workers = multiprocessing.cpu_count()  # Use all available CPU cores
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=3)


    model = TemporalFusionTransformer.from_dataset(
        validation,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    model.load_state_dict(torch.load('tft_stock_model_0.pth'))

    try:
        future_predictions = predict_future(model, processed_df)
        print("Future Price Predictions:")
        print(future_predictions)
    except Exception as e:
        print(f"Error making future predictions: {e}")
