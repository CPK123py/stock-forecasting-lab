"""
Real-Time Stock Prediction Backend Service
This Flask API handles real-time data collection, model training, and predictions
"""

from flask import Flask, jsonify, request
from flask_cors import CORS #type:ignore
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import threading
import time
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import yfinance as yf  # or use your preferred API
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')



# Global state
class TradingState:
    def __init__(self):
        self.model = None
        self.scaler_y = None
        self.features = None
        self.realtime_data = pd.DataFrame()
        self.predictions = []
        self.is_training = False
        self.is_live = False
        self.last_train_time = None
        self.metrics = {}
        self.data_queue = Queue()
        self.symbol = 'ZOMATO.NS'
        self.last_price = None
        
state = TradingState()

# Technical indicators functions
def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    if len(df) < 50:
        return df
    
    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ROC
    df['ROC'] = df['close'].pct_change(periods=5) * 100
    
    return df

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """Add lagged features"""
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        if 'RSI' in df.columns:
            df[f'rsi_lag_{lag}'] = df['RSI'].shift(lag)
        df[f'change_lag_{lag}'] = df['change_pct'].shift(lag)
    return df
def fetch_realtime_data_thread():
    """Background thread to fetch real-time data"""
    while state.is_live:
        try:
            # Fetch data from API
            # For Yahoo Finance (15-20 min delay)
            ticker = yf.Ticker(state.symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                latest = data.iloc[-1]
                day_open = data.iloc[0]['Open']  # Get day's opening price
                
                # Create new data point
                new_point = {
                    'timestamp': datetime.now(),
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'close': latest['Close'],
                    'volume': latest['Volume'],
                    'day_open': day_open,
                    'change_pct': ((latest['Close'] - day_open) / day_open * 100) if day_open != 0 else 0
                }
                
                # Add to queue
                state.data_queue.put(new_point)
                state.last_price = latest['Close']
                
                print(f"Fetched data: {new_point['close']:.2f} at {new_point['timestamp']}")
                
        except Exception as e:
            print(f"Error fetching data: {e}")
        
        # Fetch every 10 seconds
        time.sleep(10)

def process_data_queue():
    """Process incoming data from queue"""
    while state.is_live:
        try:
            if not state.data_queue.empty():
                new_point = state.data_queue.get()
                
                # Add to dataframe
                new_df = pd.DataFrame([new_point])
                state.realtime_data = pd.concat([state.realtime_data, new_df], ignore_index=True)
                
                # Keep only last 500 data points
                if len(state.realtime_data) > 500:
                    state.realtime_data = state.realtime_data.iloc[-500:]
                
        except Exception as e:
            print(f"Error processing data: {e}")
        
        time.sleep(1)

def train_model_periodic():
    """Retrain model every 10 minutes"""
    while state.is_live:
        try:
            # Wait for 10 minutes
            time.sleep(600)
            
            if len(state.realtime_data) < 100:
                print("Not enough data to train")
                continue
            
            print("Starting model retraining...")
            state.is_training = True
            
            # Prepare data
            df = state.realtime_data.copy()
            df = add_technical_indicators(df)
            df = add_lagged_features(df)
            df = df.dropna()
            
            if len(df) < 50:
                print("Not enough data after processing")
                state.is_training = False
                continue
            
            # Define features
            basic_features = [
                'open', 'high', 'low', 'close', 'volume',
                'MA5', 'MA20', 'RSI', 'ROC',
                'close_lag_1', 'close_lag_2', 'close_lag_3',
                'volume_lag_1', 'rsi_lag_1', 'change_lag_1'
            ]
            
            features = [f for f in basic_features if f in df.columns]
            
            # Prepare training data
            X = df[features].values
            y = df['change_pct'].values.reshape(-1, 1)
            
            # Scale target
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(y)
            
            # Train model
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                objective="reg:squarederror",
                random_state=42
            )
            
            model.fit(X, y_scaled.ravel())
            
            # Update global state
            state.model = model
            state.scaler_y = scaler_y
            state.features = features
            state.last_train_time = datetime.now()
            
            # Calculate metrics
            preds_scaled = model.predict(X).reshape(-1, 1)
            preds = scaler_y.inverse_transform(preds_scaled)
            
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            
            state.metrics = {
                'rmse': float(rmse),
                'r2': float(r2),
                'samples': len(df)
            }
            
            print(f"Model retrained successfully. RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Generate predictions
            generate_predictions()
            
            state.is_training = False
            
        except Exception as e:
            print(f"Error training model: {e}")
            state.is_training = False

def generate_predictions():
    """Generate predictions for next 10 minutes"""
    try:
        if state.model is None or len(state.realtime_data) < 50:
            return
        
        df = state.realtime_data.copy()
        df = add_technical_indicators(df)
        df = add_lagged_features(df)
        df = df.dropna()
        
        if len(df) == 0:
            return
        
        predictions = []
        sim_df = df.copy()
        last_actual_price = df['close'].iloc[-1]  # Store the last actual price for change% calculation
        
        for i in range(1, 11):  # 10 minutes ahead
            # Get features for prediction
            X = sim_df[state.features].iloc[-1:].values
            
            # Predict change percentage
            pred_change_scaled = state.model.predict(X).reshape(-1, 1)
            pred_change = state.scaler_y.inverse_transform(pred_change_scaled)[0, 0]
            
            # Calculate new price
            if i == 1:
                new_price = last_actual_price * (1 + pred_change/100)
            else:
                new_price = predictions[-1]['price'] * (1 + pred_change/100)
            
            # Calculate change % from last actual price
            change_from_actual = ((new_price - last_actual_price) / last_actual_price) * 100
            
            # Add prediction
            future_time = datetime.now() + timedelta(minutes=i)
            predictions.append({
                'timestamp': future_time.strftime('%H:%M:%S'),
                'price': float(new_price),
                'change_pct': float(change_from_actual),  # Changed to use change from actual
                'confidence': float(np.random.uniform(0.75, 0.95))  # Simulated confidence
            })
            
            # Update simulation dataframe for next iteration
            new_row = pd.DataFrame({
                'timestamp': [future_time],
                'open': [new_price],
                'high': [new_price * 1.005],
                'low': [new_price * 0.995],
                'close': [new_price],
                'volume': [sim_df['volume'].iloc[-1]],
                'change_pct': [pred_change]
            })
            
            sim_df = pd.concat([sim_df, new_row], ignore_index=True)
            sim_df = add_technical_indicators(sim_df)
            sim_df = add_lagged_features(sim_df)
        
        state.predictions = predictions
        print(f"Generated {len(predictions)} predictions")
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
# API Endpoints

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify({
        'is_live': state.is_live,
        'is_training': state.is_training,
        'last_train_time': state.last_train_time.isoformat() if state.last_train_time else None,
        'data_points': len(state.realtime_data),
        'last_price': float(state.last_price) if state.last_price else None,
        'metrics': state.metrics,
        'symbol': state.symbol
    })

@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start live trading"""
    try:
        data = request.json
        state.symbol = data.get('symbol', 'ZOMATO.NS')
        
        if state.is_live:
            return jsonify({'error': 'Already running'}), 400
        
        state.is_live = True
        
        # Start background threads
        threading.Thread(target=fetch_realtime_data_thread, daemon=True).start()
        threading.Thread(target=process_data_queue, daemon=True).start()
        threading.Thread(target=train_model_periodic, daemon=True).start()
        
        # Initial training after 30 seconds
        def initial_train():
            time.sleep(30)
            if state.is_live and len(state.realtime_data) >= 20:
                print("Starting initial training...")
                state.is_training = True
                
                df = state.realtime_data.copy()
                df = add_technical_indicators(df)
                df = add_lagged_features(df)
                df = df.dropna()
                
                if len(df) >= 20:
                    basic_features = [
                        'open', 'high', 'low', 'close', 'volume',
                        'MA5', 'MA20', 'RSI', 'ROC',
                        'close_lag_1', 'close_lag_2', 'close_lag_3',
                        'volume_lag_1', 'rsi_lag_1', 'change_lag_1'
                    ]
                    
                    features = [f for f in basic_features if f in df.columns]
                    
                    X = df[features].values
                    y = df['change_pct'].values.reshape(-1, 1)
                    
                    scaler_y = MinMaxScaler()
                    y_scaled = scaler_y.fit_transform(y)
                    
                    model = XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=5,
                        objective="reg:squarederror",
                        random_state=42
                    )
                    
                    model.fit(X, y_scaled.ravel())
                    
                    state.model = model
                    state.scaler_y = scaler_y
                    state.features = features
                    state.last_train_time = datetime.now()
                    
                    generate_predictions()
                    
                state.is_training = False
        
        threading.Thread(target=initial_train, daemon=True).start()
        
        return jsonify({'message': 'Trading started successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop live trading"""
    state.is_live = False
    return jsonify({'message': 'Trading stopped'})

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get current realtime data"""
    try:
        if state.realtime_data.empty:
            return jsonify({'data': []})
        
        df = state.realtime_data.tail(60).copy()
        df['timestamp'] = df['timestamp'].dt.strftime('%H:%M:%S')
        
        return jsonify({
            'data': df.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get current predictions"""
    return jsonify({
        'predictions': state.predictions
    })

@app.route('/api/train', methods=['POST'])
def manual_train():
    """Manually trigger model training"""
    try:
        if state.is_training:
            return jsonify({'error': 'Training already in progress'}), 400
        
        if len(state.realtime_data) < 50:
            return jsonify({'error': 'Not enough data'}), 400
        
        # Trigger training in background
        def train():
            state.is_training = True
            
            df = state.realtime_data.copy()
            df = add_technical_indicators(df)
            df = add_lagged_features(df)
            df = df.dropna()
            
            basic_features = [
                'open', 'high', 'low', 'close', 'volume',
                'MA5', 'MA20', 'RSI', 'ROC',
                'close_lag_1', 'close_lag_2', 'close_lag_3',
                'volume_lag_1', 'rsi_lag_1', 'change_lag_1'
            ]
            
            features = [f for f in basic_features if f in df.columns]
            
            X = df[features].values
            y = df['change_pct'].values.reshape(-1, 1)
            
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(y)
            
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                objective="reg:squarederror",
                random_state=42
            )
            
            model.fit(X, y_scaled.ravel())
            
            state.model = model
            state.scaler_y = scaler_y
            state.features = features
            state.last_train_time = datetime.now()
            
            from sklearn.metrics import mean_squared_error, r2_score
            preds_scaled = model.predict(X).reshape(-1, 1)
            preds = scaler_y.inverse_transform(preds_scaled)
            
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            
            state.metrics = {
                'rmse': float(rmse),
                'r2': float(r2),
                'samples': len(df)
            }
            
            generate_predictions()
            
            state.is_training = False
        
        threading.Thread(target=train, daemon=True).start()
        
        return jsonify({'message': 'Training started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        data = request.json
        state.symbol = data.get('symbol', state.symbol)
        return jsonify({'message': 'Configuration updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Real-Time Stock Prediction Backend...")
    print("API will be available at http://localhost:5000")
    print("Endpoints:")
    print("  POST /api/start - Start live trading")
    print("  POST /api/stop - Stop live trading")
    print("  GET /api/status - Get system status")
    print("  GET /api/data - Get real-time data")
    print("  GET /api/predictions - Get predictions")
    print("  POST /api/train - Manual training trigger")
    print("  POST /api/config - Update configuration")
    
    app.run(debug=True, host='0.0.0.0', port=5000)