#!/usr/bin/env python3
"""
Simple training script that works with the generated data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_data():
    """Load the generated data"""
    data_files = []
    for file in os.listdir('data/raw'):
        if file.startswith('vehicle_') and file.endswith('.parquet'):
            data_files.append(f'data/raw/{file}')
    
    if not data_files:
        print("No data files found. Run the simulator first.")
        return None
    
    # Load first few files
    dfs = []
    for file in data_files[:5]:  # Use first 5 vehicles
        df = pd.read_parquet(file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} records from {len(dfs)} vehicles")
    return combined

def create_simple_features(df):
    """Create basic features"""
    # Sort by vehicle and time
    df = df.sort_values(['vehicle_id', 'timestamp'])
    
    # Basic features
    df['cell_imbalance'] = df['cell_voltage_max'] - df['cell_voltage_min']
    df['temp_diff'] = df['pack_temp'] - df['ambient_temp']
    df['is_charging'] = (df['pack_current'] > 0).astype(int)
    df['is_discharging'] = (df['pack_current'] < 0).astype(int)
    
    # Rolling features (simple 24-hour windows)
    df['pack_temp_24h_max'] = df.groupby('vehicle_id')['pack_temp'].rolling(288).max().values  # 24h at 5min resolution
    df['current_24h_mean'] = df.groupby('vehicle_id')['pack_current'].rolling(288).mean().values
    df['soc_24h_std'] = df.groupby('vehicle_id')['soc'].rolling(288).std().values
    
    # Vehicle age
    df['days_since_start'] = (df['timestamp'] - df.groupby('vehicle_id')['timestamp'].transform('min')).dt.days
    
    return df

def train_model():
    """Train a simple model"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create features
    print("Creating features...")
    df = create_simple_features(df)
    
    # Select features for training
    feature_cols = [
        'pack_voltage', 'cell_voltage_min', 'cell_voltage_max', 'pack_current',
        'soc', 'pack_temp', 'ambient_temp', 'vehicle_speed',
        'cell_imbalance', 'temp_diff', 'is_charging', 'is_discharging',
        'days_since_start'
    ]
    
    # Add rolling features where available
    rolling_cols = ['pack_temp_24h_max', 'current_24h_mean', 'soc_24h_std']
    for col in rolling_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    # Prepare data
    df_clean = df.dropna(subset=['soh_percent'] + feature_cols)
    X = df_clean[feature_cols]
    y = df_clean['soh_percent']
    
    print(f"Training data: {len(X)} samples, {len(feature_cols)} features")
    
    # Split data by vehicle (no leakage)
    vehicles = df_clean['vehicle_id'].unique()
    train_vehicles = vehicles[:int(len(vehicles) * 0.8)]
    
    train_mask = df_clean['vehicle_id'].isin(train_vehicles)
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Important Features:")
    print(importance.head())
    
    # Save model
    os.makedirs('models/artifacts', exist_ok=True)
    joblib.dump(model, 'models/artifacts/simple_model.joblib')
    joblib.dump(feature_cols, 'models/artifacts/feature_columns.joblib')
    
    print(f"\nModel saved to models/artifacts/")
    return model

if __name__ == "__main__":
    train_model()