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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import optuna
from optuna.integration import PyTorchLightningPruningCallback

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

def calculate_mape(y_true, y_pred):
    # Avoid division by zero with a small epsilon
    epsilon = 1e-10
    # Ensure only positive values are used to avoid infinite MAPE
    mask = y_true > epsilon
    if mask.sum() == 0:
        return np.nan  # Return NaN if no valid points
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Define your LightningTFT class with enhanced metrics
class LightningTFT(pl.LightningModule):
    def __init__(self, tft, learning_rate=1e-3, weight_decay=1e-5):
        super().__init__()
        self.tft = tft
        self.loss_function = QuantileLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Save metrics for logging
        self.train_metrics = {}
        self.val_metrics = {}

    def forward(self, x):
        return self.tft(x)
    
    def calculate_metrics(self, y_pred, y_true):
        """Calculate multiple performance metrics safely with type conversion"""
        
        # Ensure y_true is a tensor and extract if it's a tuple
        if isinstance(y_true, tuple):
            y_true = y_true[0]  # Extract first element if it's a tuple

        # Convert tensors to float32 before detaching and moving to CPU
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Convert to numpy arrays and reshape to 1D (flatten)
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        y_true_np = y_true.detach().cpu().numpy().reshape(-1)
        
        # Ensure y_pred and y_true have the same shape
        min_length = min(len(y_pred_np), len(y_true_np))
        y_pred_np = y_pred_np[:min_length]
        y_true_np = y_true_np[:min_length]
        
        # Skip empty arrays
        if min_length == 0:
            return {
                "loss": torch.tensor(0.0, requires_grad=True),
                "rmse": torch.tensor(0.0),
                "mae": torch.tensor(0.0),
                "mape": torch.tensor(0.0),
                "r2": torch.tensor(0.0)
            }

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        mae = mean_absolute_error(y_true_np, y_pred_np)
        
        # Handle MAPE calculation with safeguards
        epsilon = 1e-10
        mask = np.abs(y_true_np) > epsilon
        if np.any(mask):
            mape = np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100
        else:
            mape = np.nan
        
        # Handle potential NaN in MAPE
        if np.isnan(mape):
            mape = 0.0
        
        # R2 score with error handling
        try:
            r2 = r2_score(y_true_np, y_pred_np)
            # Clip extremely negative R2 values
            r2 = max(r2, -1.0)
        except:
            r2 = -1.0
        
        return {
            "loss": self.loss_function(y_pred, y_true),
            "rmse": torch.tensor(rmse, dtype=torch.float32),
            "mae": torch.tensor(mae, dtype=torch.float32),
            "mape": torch.tensor(mape, dtype=torch.float32),
            "r2": torch.tensor(r2, dtype=torch.float32)
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate all metrics
        metrics = self.calculate_metrics(y_hat[0], y)
        loss = metrics["loss"]
        
        # Log all metrics
        for name, value in metrics.items():
            self.log(f"train_{name}", value, prog_bar=True if name == "loss" else False)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate all metrics
        metrics = self.calculate_metrics(y_hat[0], y)
        
        # Log all metrics
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=True)
            
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5,  # Reduce learning rate by half
            patience=3,   # Wait 3 epochs
            min_lr=1e-5,  # Don't go below this LR
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


def preprocess_data(df, log_transform=True):
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
    
    # Add technical indicators
    df = add_technical_indicators(df)
    df = df.dropna()
    
    # Apply log transformation if specified
    if log_transform:
        df['Price'] = np.log1p(df['Price'])
    
    # Standardize features for better model convergence
    feature_columns = ['MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 
                       'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'OBV']
    
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # Recalculate time_idx after removing NaN rows
    df['time_idx'] = range(len(df))
    
    return df


def evaluate_model(model, test_dataloader, original_scale=True):
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
        # Handle NaN values safely
        mean_value = np.nanmean(predictions)
        predictions = np.nan_to_num(predictions, nan=mean_value)
    
    # Reverse log transformation if applied during preprocessing
    if original_scale:
        actuals = np.expm1(actuals)
        predictions = np.expm1(predictions)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # Handle potential division by zero or infinite values in MAPE
    mape = calculate_mape(actuals, predictions)
    if np.isnan(mape) or np.isinf(mape):
        mape = float('inf')  # Set to infinity if calculation fails
    
    # Calculate R2 with error handling
    try:
        r2 = r2_score(actuals, predictions)
    except:
        r2 = float('-inf')  # Set to negative infinity if calculation fails
    
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
        "r2": r2,
        "predictions": predictions,
        "actuals": actuals
    }

def objective(trial, processed_df, max_encoder_length, max_prediction_length):
    """Optuna objective function for hyperparameter optimization"""
    
    # Define hyperparameter search space
    hidden_size = trial.suggest_int('hidden_size', 16, 64, 8)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    hidden_continuous_size = trial.suggest_int('hidden_continuous_size', 8, 32, 4)
    attention_head_size = trial.suggest_int('attention_head_size', 1, 4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Use cross-validation for more robust evaluation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Store validation scores across folds
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(processed_df)))):
        print(f"\nTraining fold {fold+1}/3")
        
        # Create training cutoff based on fold indices
        training_cutoff = max(train_idx)
        
        # Debug: Check dataset size and indices
        print(f"Total rows in dataset: {len(processed_df)}")
        print(f"Training cutoff: {training_cutoff}")
        print(f"Max encoder length: {max_encoder_length}")
        print(f"Max prediction length: {max_prediction_length}")
        
        # Ensure there are enough rows for validation
        if training_cutoff + max_prediction_length >= len(processed_df):
            print(f"Skipping fold {fold+1} due to insufficient data for validation.")
            continue
        
        # Create training dataset
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
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            processed_df, 
            min_prediction_idx=training_cutoff + 1
        )
        
        # Debug: Check validation dataset size
        print(f"Validation dataset size: {len(validation)}")
        
        # Skip if validation dataset is empty
        if len(validation) == 0:
            print(f"Skipping fold {fold+1} due to empty validation dataset.")
            continue
        
        # Create data loaders
        batch_size = 64
        train_dataloader = training.to_dataloader(
            train=True, 
            batch_size=batch_size, 
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        val_dataloader = validation.to_dataloader(
            train=False, 
            batch_size=batch_size * 2,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize the TFT model with trial parameters
        pl.seed_everything(42 + fold)  # Different seed for each fold
        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )
        
        # Create the Lightning module
        lit_model = LightningTFT(
            tft, 
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Configure callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=0.001
        )
        
        # PyTorch Lightning pruning callback
        pruning_callback = PyTorchLightningPruningCallback(
            trial, 
            monitor="val_loss"
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=15,  # Reduced for faster trials
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[early_stop_callback, pruning_callback],
            limit_train_batches=30,  # Reduced for faster trials
            logger=True,
            enable_progress_bar=True,
            log_every_n_steps=10,
            enable_checkpointing=False,  # Disable for faster trials
        )
        
        # Train the model
        try:
            trainer.fit(
                lit_model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader
            )
            # Get best validation loss
            val_loss = trainer.callback_metrics.get("val_loss").item()
            val_scores.append(val_loss)
        except Exception as e:
            print(f"Error during training: {e}")
            val_scores.append(float('inf'))  # Assign worst possible score
        
        # Clean up
        del tft, lit_model, trainer
        torch.cuda.empty_cache()
    
    # Average validation loss across folds
    mean_val_loss = np.mean(val_scores)
    return mean_val_loss


def run_hyperparameter_optimization(processed_df, max_encoder_length, max_prediction_length, n_trials=20):
    """Run Optuna hyperparameter optimization"""
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, processed_df, max_encoder_length, max_prediction_length),
        n_trials=n_trials
    )
    
    # Print best parameters
    print("\n===== Best Hyperparameters =====")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params


def cross_validate_model(processed_df, best_params, n_folds=5):
    """Cross-validate the model with the best hyperparameters"""
    
    max_encoder_length = 30
    max_prediction_length = 5
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(processed_df)))):
        print(f"\n===== Fold {fold+1}/{n_folds} =====")
        
        # Create training cutoff based on fold indices
        training_cutoff = max(train_idx)
        
        # Create training dataset
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
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            processed_df, 
            min_prediction_idx=training_cutoff + 1
        )
        
        # Create data loaders
        batch_size = 64
        train_dataloader = training.to_dataloader(
            train=True, 
            batch_size=batch_size, 
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        val_dataloader = validation.to_dataloader(
            train=False, 
            batch_size=batch_size * 2,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize the TFT model with best parameters
        pl.seed_everything(42 + fold)  # Different seed for each fold
        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=best_params['hidden_size'],
            attention_head_size=best_params['attention_head_size'],
            dropout=best_params['dropout'],
            hidden_continuous_size=best_params['hidden_continuous_size'],
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )
        
        # Create the Lightning module
        lit_model = LightningTFT(
            tft, 
            learning_rate=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
        
        # Configure callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            min_delta=0.001
        )
        
        lr_logger = LearningRateMonitor()
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=50,  # Train for longer with best parameters
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_logger],
            logger=True,
            enable_progress_bar=True,
            log_every_n_steps=5,
            enable_checkpointing=True,
        )
        
        # Train the model
        trainer.fit(
            lit_model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader
        )
        
        # Evaluate the model
        metrics = evaluate_model(tft, val_dataloader)
        fold_metrics.append(metrics)
        
        # Clean up
        del tft, lit_model, trainer
        torch.cuda.empty_cache()
    
    # Calculate average metrics across folds
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'mape': np.mean([m['mape'] for m in fold_metrics if m['mape'] != float('inf')]),
        'r2': np.mean([m['r2'] for m in fold_metrics])
    }
    
    print("\n===== Cross-Validation Results =====")
    print(f"Average MAE: {avg_metrics['mae']:.4f}")
    print(f"Average RMSE: {avg_metrics['rmse']:.4f}")
    print(f"Average MAPE: {avg_metrics['mape']:.4f}%")
    print(f"Average R²: {avg_metrics['r2']:.4f}")
    
    return fold_metrics, avg_metrics


def train_final_model(processed_df, best_params):
    """Train the final model with the best hyperparameters"""
    
    max_encoder_length = 30
    max_prediction_length = 5
    
    # Create training cutoff
    training_cutoff = processed_df['time_idx'].max() - max_prediction_length
    
    # Create training dataset
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
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        processed_df, 
        min_prediction_idx=training_cutoff + 1
    )
    
    # Create data loaders
    batch_size = 64
    train_dataloader = training.to_dataloader(
        train=True, 
        batch_size=batch_size, 
        num_workers=3,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=batch_size * 2,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize the TFT model with best parameters
    pl.seed_everything(42)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=best_params['hidden_size'],
        attention_head_size=best_params['attention_head_size'],
        dropout=best_params['dropout'],
        hidden_continuous_size=best_params['hidden_continuous_size'],
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    
    # Create the Lightning module
    lit_model = LightningTFT(
        tft, 
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # Configure callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        min_delta=0.001
    )
    
    lr_logger = LearningRateMonitor()
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=100,  # Train final model for longer
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        logger=True,
        enable_progress_bar=True,
        log_every_n_steps=5,
        enable_checkpointing=True,
    )
    
    # Train the model
    print("\n===== Training Final Model =====")
    trainer.fit(
        lit_model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )
    
    # Evaluate the model
    metrics = evaluate_model(tft, val_dataloader)
    
    # Save the model
    model_checkpoint_path = 'tft_stock_model_best.ckpt'
    trainer.save_checkpoint(model_checkpoint_path)
    print(f"Model checkpoint saved to {model_checkpoint_path}")
    
    torch.save(tft.state_dict(), 'tft_stock_model_best.pth')
    print(f"Model state dict saved to tft_stock_model_best.pth")
    
    return tft, metrics


def plot_predictions(model, val_dataloader, num_examples=5):
    """Plot predictions and actuals for validation data"""
    
    with torch.no_grad():
        raw_predictions = model.predict(
            val_dataloader, 
            mode="raw", 
            return_x=True, 
            trainer_kwargs=dict(accelerator="cpu")
        )
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4 * num_examples))
    for idx in range(min(num_examples, len(raw_predictions.x))):
        if num_examples > 1:
            ax = axes[idx]
        else:
            ax = axes
            
        model.plot_prediction(
            raw_predictions.x, 
            raw_predictions.output, 
            idx=idx, 
            add_loss_to_title=True,
            ax=ax
        )
        ax.set_title(f"Prediction vs Actual (Example {idx+1})", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("prediction_examples.png")
    plt.show()


def plot_feature_importance(model):
    """Plot feature importance for the model"""
    
    interpretation = model.interpret_output(
        attention_prediction_horizon=0
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    model.plot_interpretation(interpretation, ax=ax)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
    
    # Plot attention weights for more detailed insight
    fig, ax = plt.subplots(figsize=(10, 8))
    model.plot_attention(
        interpretation,
        ax=ax
    )
    plt.tight_layout()
    plt.savefig("attention_weights.png")
    plt.show()


def plot_metrics_summary(metrics, training_time, best_params=None):
    """Create a summary figure with metrics"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_text = f"""
    Model Evaluation Metrics:
    -------------------------
    MAE:  {metrics['mae']:.4f}
    RMSE: {metrics['rmse']:.4f}
    MAPE: {metrics['mape']:.2f}%
    R²:   {metrics['r2']:.4f}
    
    Training time: {training_time/60:.2f} minutes
    """
    
    if best_params:
        params_text = "\nBest Hyperparameters:\n-------------------------\n"
        for key, value in best_params.items():
            params_text += f"{key}: {value}\n"
        metrics_text += params_text
    
    ax.text(0.5, 0.5, metrics_text, fontsize=14, ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("model_metrics_summary.png")
    plt.show()


if __name__ == '__main__':
    # Record start time
    start_time = time.time()
    
    # Ensure multiprocessing works correctly on Windows
    multiprocessing.freeze_support()
    
    # Set default tensor type
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv('ZOMT.csv')
    processed_df = preprocess_data(df, log_transform=True)
    
    # Define key parameters
    max_encoder_length = 10
    max_prediction_length = 3
    
    # Hyperparameter optimization
    print("\n===== Starting Hyperparameter Optimization =====")
    best_params = run_hyperparameter_optimization(
        processed_df, 
        max_encoder_length, 
        max_prediction_length,
        n_trials=15  # Adjust based on available time/resources
    )
    
    # Cross-validation with best parameters
    print("\n===== Starting Cross-Validation =====")
    fold_metrics, avg_metrics = cross_validate_model(
        processed_df,
        best_params,
        n_folds=5
    )
    
    # Train final model
    print("\n===== Training Final Model =====")
    final_model, final_metrics = train_final_model(
        processed_df, 
        best_params
    )
    
    # Record total time
    training_time = time.time() - start_time
    print(f"\nTotal time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Generate visualizations
    print("\n===== Generating Visualizations =====")
    
    # Create cutoff for validation data
    training_cutoff = processed_df['time_idx'].max() - max_prediction_length
    validation = TimeSeriesDataSet.from_dataset(
        TimeSeriesDataSet(
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
        ), 
        processed_df, 
        min_prediction_idx=training_cutoff + 1
    )
    
    val_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Plot predictions
    plot_predictions(final_model, val_dataloader, num_examples=5)
    
    # Plot feature importance
    plot_feature_importance(final_model)
    
    # Plot metrics summary
    plot_metrics_summary(final_metrics, training_time, best_params)
    
    print("\n===== Analysis Complete =====")