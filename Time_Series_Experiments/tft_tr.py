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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time
from sklearn.preprocessing import StandardScaler

def add_technical_indicators(df):
    # Moving Averages
    df['MA5'] = df['Price'].rolling(window=5).mean()  # Use 'Price' instead of 'price'
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['MA50'] = df['Price'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Price'].diff()  # Use 'Price' instead of 'price'
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Price'].ewm(span=12, adjust=False).mean()  # Use 'Price' instead of 'price'
    exp2 = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Price'].rolling(window=20).mean()  # Use 'Price' instead of 'price'
    std_dev = df['Price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    
    # Price Rate of Change
    df['ROC'] = df['Price'].pct_change(periods=5) * 100  # Use 'Price' instead of 'price'
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[1:, 'OBV'] = (df['Price'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()  # Use 'Price' instead of 'price'
    
    return df

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10  # Small constant to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

# Define your LightningTFT class with enhanced metrics
class LightningTFT(pl.LightningModule):
    def __init__(self, tft, learning_rate=1e-3):
        super().__init__()
        self.tft = tft
        self.loss_function = QuantileLoss()
        self.learning_rate = learning_rate
        
        # Save all metrics for logging
        self.train_metrics = {}
        self.val_metrics = {}

    def forward(self, x):
        return self.tft(x)
    

    
    def calculate_metrics(self, y_pred, y_true):
        """Calculate multiple performance metrics safely with type conversion"""
        
        # Ensure y_true is a tensor and extract if it’s a tuple
        if isinstance(y_true, tuple):
            y_true = y_true[0]  # Extract first element if it's a tuple

        # Convert tensors to float32 before detaching and moving to CPU
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Convert to numpy arrays and reshape to 1D (flatten)
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        y_true_np = y_true.detach().cpu().numpy().reshape(-1)
        
        # **Ensure y_pred and y_true have the same shape**
        min_length = min(len(y_pred_np), len(y_true_np))
        y_pred_np = y_pred_np[:min_length]  # Trim extra elements
        y_true_np = y_true_np[:min_length]  # Trim extra elements

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        mae = mean_absolute_error(y_true_np, y_pred_np)
        
        # Handle potential division by zero in MAPE calculation
        mape = np.mean(np.abs((y_true_np - y_pred_np) / np.maximum(np.abs(y_true_np), 1e-10))) * 100
    
        # R2 score
        r2 = r2_score(y_true_np, y_pred_np)
        
        return {
            "loss": self.loss_function(y_pred, y_true),
            "rmse": torch.tensor(rmse, dtype=torch.float32),
            "mae": torch.tensor(mae, dtype=torch.float32),
            "mape": torch.tensor(mape, dtype=torch.float32),
            "r2": torch.tensor(r2, dtype=torch.float32)
        }


    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # y_hat is a tuple, extract the first element if needed

        # Ensure batch size is determined correctly
        batch_size = y.shape[0]

        # Calculate all metrics
        metrics = self.calculate_metrics(y_hat[0], y)
        loss = metrics["loss"]

        # Log all metrics with batch_size
        for name, value in metrics.items():
            self.log(f"train_{name}", value, prog_bar=(name == "loss"), batch_size=batch_size)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # y_hat is a tuple, extract the first element if needed

        # Ensure batch size is determined correctly
        batch_size = y.shape[0]

        # Calculate all metrics
        metrics = self.calculate_metrics(y_hat[0], y)

        # Log all metrics with batch_size
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=True, batch_size=batch_size)

        return metrics


    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
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


def evaluate_model(model, test_dataloader):
    model.eval()  # Set model to evaluation mode
    actuals_list = []
    for x, y in iter(test_dataloader):
        if isinstance(y, tuple):
            actuals_list.append(y[0])
        elif isinstance(y, dict):
            actuals_list.append(y['target'])
        else:
            actuals_list.append(y)

    actuals = torch.cat(actuals_list).float().cpu().numpy()
    
    with torch.no_grad():
        predictions = model.predict(test_dataloader, trainer_kwargs=dict(accelerator="cpu"))
        predictions = predictions.float().cpu().numpy()
        mean_value = np.nanmean(predictions)
        predictions = np.nan_to_num(predictions, nan=mean_value)
    
    # Reverse log transformation
    actuals = np.expm1(actuals)
    predictions = np.expm1(predictions)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = calculate_mape(actuals, predictions)  # Use the modified MAPE function
    r2 = r2_score(actuals, predictions)
    
    # Print results
    print("\n===== Model Evaluation =====")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"R-squared (R²): {r2:.4f}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }


if __name__ == '__main__':
    # Record start time
    start_time = time.time()
    
    # Ensure multiprocessing works correctly on Windows
    multiprocessing.freeze_support()

    # Set default tensor type
    torch.set_default_tensor_type(torch.FloatTensor)

    # Load and preprocess data
    df = pd.read_csv('ZOMT.csv')
    processed_df = preprocess_data(df)
    processed_df = add_technical_indicators(processed_df)
    processed_df = processed_df.dropna()
    processed_df['time_idx'] = range(len(processed_df))
    processed_df['Price'] = np.log1p(processed_df['Price'])
    processed_df.dropna()


    scaler = StandardScaler()
    processed_df[['MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'OBV']] = scaler.fit_transform(
    processed_df[['MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'OBV']])

    # Define the training and validation cutoffs
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
            "Price", "Open", "High", "Low", "volume", "Change %",
            "MA5", "MA20", "MA50", "RSI", "MACD", "MACD_signal",
            "BB_middle", "BB_upper", "BB_lower", "ROC", "OBV"
        ],
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create the validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, processed_df, min_prediction_idx=training_cutoff + 1)

    # Create data loaders with optimized parameters
    batch_size = 64  # Increased batch size
    train_dataloader = training.to_dataloader(
        train=True, 
        batch_size=batch_size, 
        num_workers=3,
        pin_memory=True,  # Speed up data transfer to GPU
        persistent_workers=True  # Keep workers alive between batches
    )
    val_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=batch_size * 2,  # Can use larger batch size for validation
        num_workers=3,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize the TFT model
    pl.seed_everything(42)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=64,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    # Wrap TFT in a LightningModule with a higher learning rate
    lit_model = LightningTFT(tft, learning_rate=2e-3) 

    # Configure training with optimized callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=5,  # Reduced patience for faster training
        mode="min",
        min_delta=0.001  # Stop when improvements are small
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        limit_train_batches=50,
        precision="32",  # Use standard precision instead of mixed precision
        logger=True,
        enable_progress_bar=True,
        log_every_n_steps=5,
        enable_checkpointing=True,
    )

    # Train the model
    print("Starting model training...")
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Save the model
    model_checkpoint_path = 'tft_stock_model_checkpoint.ckpt'
    trainer.save_checkpoint(model_checkpoint_path)
    print(f"Model checkpoint saved to {model_checkpoint_path}")

    model_state_dict_path = 'tft_stock_model_0.pth'
    torch.save(tft.state_dict(), model_state_dict_path)
    print(f"Model state dict saved to {model_state_dict_path}")

    # Load best model for predictions
    best_tft = tft
    best_tft.to("cpu")

    # Evaluate model performance
    print("\nEvaluating model performance...")
    metrics = evaluate_model(best_tft, val_dataloader)
    
    # Create predictions for the validation period
    print("\nGenerating predictions and visualizations...")
    with torch.no_grad():
        predictions = best_tft.predict(val_dataloader, return_x=True, trainer_kwargs=dict(accelerator="cpu"))


        # Visualize predictions
        raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))

    def plot_predictions():
        for idx in range(min(5, len(predictions[0]))):  # Plot up to 5 examples
            fig, ax = plt.subplots(figsize=(10, 6))
            best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True, ax=ax)
            plt.title(f"Prediction vs Actual (Example {idx+1})", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"prediction_example_{idx+1}.png")
            plt.show()


                        
    # Create a summary figure with metrics
    def plot_metrics_summary():
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_text = f"""
        Model Evaluation Metrics:
        -------------------------
        MAE:  {metrics['mae']:.4f}
        RMSE: {metrics['rmse']:.4f}
        MAPE: {metrics['mape']:.2f}%
        R²:   {metrics['r2']:.4f}
        
        Training time: {training_time/60:.2f} minutes
        """
        ax.text(0.5, 0.5, metrics_text, fontsize=14, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("model_metrics_summary.png")
        plt.show()
        
    plot_predictions()
    plot_metrics_summary()