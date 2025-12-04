#!/usr/bin/env python3
"""
Feature Engineering Pipeline for EV Battery Prediction
Creates time-windowed features, cycle counts, and degradation indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BatteryFeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        
    def create_time_windows(self, df: pd.DataFrame, windows: List[str] = ['24H', '7D', '30D']) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        features = []
        
        for window in windows:
            window_suffix = window.replace('H', 'h').replace('D', 'd')
            
            # Rolling statistics for key metrics
            rolling = df.groupby('vehicle_id').rolling(window, on='timestamp')
            
            # Voltage features
            features.extend([
                f'pack_voltage_mean_{window_suffix}',
                f'pack_voltage_std_{window_suffix}',
                f'pack_voltage_min_{window_suffix}',
                f'pack_voltage_max_{window_suffix}',
                f'cell_voltage_imbalance_mean_{window_suffix}',
                f'cell_voltage_imbalance_max_{window_suffix}'
            ])
            
            df[f'pack_voltage_mean_{window_suffix}'] = rolling['pack_voltage'].mean().values
            df[f'pack_voltage_std_{window_suffix}'] = rolling['pack_voltage'].std().values
            df[f'pack_voltage_min_{window_suffix}'] = rolling['pack_voltage'].min().values
            df[f'pack_voltage_max_{window_suffix}'] = rolling['pack_voltage'].max().values
            
            # Cell imbalance - create column first
            if 'cell_imbalance' not in df.columns:
                df['cell_imbalance'] = df['cell_voltage_max'] - df['cell_voltage_min']
            
            rolling_imbalance = df.groupby('vehicle_id').rolling(window, on='timestamp')['cell_imbalance']
            df[f'cell_voltage_imbalance_mean_{window_suffix}'] = rolling_imbalance.mean().values
            df[f'cell_voltage_imbalance_max_{window_suffix}'] = rolling_imbalance.max().values
            
            # Current features
            features.extend([
                f'current_mean_{window_suffix}',
                f'current_std_{window_suffix}',
                f'discharge_current_mean_{window_suffix}',
                f'charge_current_mean_{window_suffix}'
            ])
            
            df[f'current_mean_{window_suffix}'] = rolling['pack_current'].mean().values
            df[f'current_std_{window_suffix}'] = rolling['pack_current'].std().values
            
            # Separate charge/discharge currents
            df['discharge_current'] = np.where(df['pack_current'] < 0, -df['pack_current'], 0)
            df['charge_current'] = np.where(df['pack_current'] > 0, df['pack_current'], 0)
            
            df[f'discharge_current_mean_{window_suffix}'] = rolling['discharge_current'].mean().values
            df[f'charge_current_mean_{window_suffix}'] = rolling['charge_current'].mean().values
            
            # Temperature features
            features.extend([
                f'pack_temp_mean_{window_suffix}',
                f'pack_temp_max_{window_suffix}',
                f'temp_gradient_mean_{window_suffix}',
                f'thermal_stress_hours_{window_suffix}'
            ])
            
            df[f'pack_temp_mean_{window_suffix}'] = rolling['pack_temp'].mean().values
            df[f'pack_temp_max_{window_suffix}'] = rolling['pack_temp'].max().values
            
            # Temperature gradient
            df['temp_gradient'] = df['pack_temp'] - df['ambient_temp']
            df[f'temp_gradient_mean_{window_suffix}'] = rolling['temp_gradient'].mean().values
            
            # Thermal stress (temp > 40°C)
            df['thermal_stress'] = (df['pack_temp'] > 40).astype(int)
            df[f'thermal_stress_hours_{window_suffix}'] = rolling['thermal_stress'].sum().values
            
            # SOC features
            features.extend([
                f'soc_mean_{window_suffix}',
                f'soc_std_{window_suffix}',
                f'soc_range_{window_suffix}',
                f'low_soc_hours_{window_suffix}'
            ])
            
            df[f'soc_mean_{window_suffix}'] = rolling['soc'].mean().values
            df[f'soc_std_{window_suffix}'] = rolling['soc'].std().values
            df[f'soc_range_{window_suffix}'] = rolling['soc'].max().values - rolling['soc'].min().values
            
            # Low SOC stress (< 20%)
            df['low_soc'] = (df['soc'] < 0.2).astype(int)
            df[f'low_soc_hours_{window_suffix}'] = rolling['low_soc'].sum().values
            
            # Charging features
            features.extend([
                f'charging_sessions_{window_suffix}',
                f'fast_charge_sessions_{window_suffix}',
                f'total_charge_energy_{window_suffix}'
            ])
            
            # Count charging sessions
            df['charging'] = (df['pack_current'] > 10).astype(int)
            df['charging_session_start'] = (df['charging'].diff() == 1).astype(int)
            df[f'charging_sessions_{window_suffix}'] = rolling['charging_session_start'].sum().values
            
            # Fast charging sessions
            df['fast_charging_start'] = (df['is_fast_charge'].astype(int).diff() == 1).astype(int)
            df[f'fast_charge_sessions_{window_suffix}'] = rolling['fast_charging_start'].sum().values
            
            # Total charge energy (approximation)
            df['charge_energy'] = np.where(df['pack_current'] > 0, 
                                         df['pack_current'] * df['pack_voltage'] / 1000, 0)  # kWh
            df[f'total_charge_energy_{window_suffix}'] = rolling['charge_energy'].sum().values
        
        self.feature_columns = features
        return df
    
    def create_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create battery cycle-based features"""
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        # Depth of Discharge (DoD) calculation
        df['soc_diff'] = df.groupby('vehicle_id')['soc'].diff()
        df['discharge_event'] = (df['soc_diff'] < -0.1).astype(int)  # >10% discharge
        
        # Cumulative cycles (simplified: count discharge events)
        df['cumulative_cycles'] = df.groupby('vehicle_id')['discharge_event'].cumsum()
        
        # Cycle depth distribution
        df['shallow_cycles'] = df.groupby('vehicle_id')['discharge_event'].rolling(100).sum().values
        df['deep_cycles'] = ((df['soc_diff'] < -0.5) & (df['soc_diff'].notna())).astype(int)
        df['cumulative_deep_cycles'] = df.groupby('vehicle_id')['deep_cycles'].cumsum()
        
        # C-rate features (current relative to capacity)
        df['c_rate'] = abs(df['pack_current']) / (df['capacity_mAh'] / 1000)  # Current/Capacity
        df['high_c_rate_events'] = (df['c_rate'] > 1.0).astype(int)  # >1C rate
        df['cumulative_high_c_events'] = df.groupby('vehicle_id')['high_c_rate_events'].cumsum()
        
        # Time-based features
        df['days_since_manufacture'] = (df['timestamp'] - df.groupby('vehicle_id')['timestamp'].transform('min')).dt.days
        df['vehicle_age_years'] = df['days_since_manufacture'] / 365.25
        
        cycle_features = [
            'cumulative_cycles', 'shallow_cycles', 'cumulative_deep_cycles',
            'c_rate', 'cumulative_high_c_events', 'days_since_manufacture', 'vehicle_age_years'
        ]
        
        self.feature_columns.extend(cycle_features)
        return df
    
    def create_degradation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that indicate battery degradation"""
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        # Internal resistance proxy (voltage drop under load)
        df['voltage_drop'] = df.groupby('vehicle_id')['pack_voltage'].diff()
        df['current_change'] = df.groupby('vehicle_id')['pack_current'].diff()
        
        # Resistance approximation (ΔV/ΔI)
        df['resistance_proxy'] = np.where(
            abs(df['current_change']) > 10,
            abs(df['voltage_drop'] / df['current_change']),
            np.nan
        )
        
        # Rolling resistance trend
        df['resistance_trend_7d'] = df.groupby('vehicle_id')['resistance_proxy'].rolling(
            '7D', on='timestamp', min_periods=10
        ).mean().values
        
        # Capacity fade indicators
        df['soh_change'] = df.groupby('vehicle_id')['soh_percent'].diff()
        df['capacity_fade_rate'] = df.groupby('vehicle_id')['soh_change'].rolling(30).mean().values
        
        # Voltage recovery after discharge
        df['is_rest_period'] = (abs(df['pack_current']) < 5).astype(int)
        df['voltage_recovery'] = np.where(
            df['is_rest_period'] == 1,
            df['pack_voltage'] - df.groupby('vehicle_id')['pack_voltage'].shift(1),
            0
        )
        
        # Energy efficiency
        df['energy_in'] = np.where(df['pack_current'] > 0, 
                                  df['pack_current'] * df['pack_voltage'], 0)
        df['energy_out'] = np.where(df['pack_current'] < 0, 
                                   -df['pack_current'] * df['pack_voltage'], 0)
        
        df['efficiency_7d'] = (
            df.groupby('vehicle_id')['energy_out'].rolling('7D', on='timestamp').sum() /
            df.groupby('vehicle_id')['energy_in'].rolling('7D', on='timestamp').sum()
        ).values
        
        degradation_features = [
            'resistance_proxy', 'resistance_trend_7d', 'capacity_fade_rate',
            'voltage_recovery', 'efficiency_7d'
        ]
        
        self.feature_columns.extend(degradation_features)
        return df
    
    def create_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operational and usage pattern features"""
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['season'] = ((df['timestamp'].dt.month % 12) // 3).map({0: 'winter', 1: 'spring', 2: 'summer', 3: 'fall'})
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Usage intensity
        df['daily_distance'] = df.groupby(['vehicle_id', df['timestamp'].dt.date])['vehicle_speed'].transform('sum') / 60  # km
        df['daily_energy_usage'] = df.groupby(['vehicle_id', df['timestamp'].dt.date])['energy_out'].transform('sum')
        
        # Driving patterns
        df['is_highway_speed'] = (df['vehicle_speed'] > 80).astype(int)
        df['is_city_driving'] = ((df['vehicle_speed'] > 0) & (df['vehicle_speed'] <= 50)).astype(int)
        df['stop_and_go'] = (df['vehicle_speed'].diff().abs() > 20).astype(int)
        
        # Environmental stress
        df['extreme_cold'] = (df['ambient_temp'] < 0).astype(int)
        df['extreme_heat'] = (df['ambient_temp'] > 35).astype(int)
        
        operational_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'daily_distance', 'daily_energy_usage', 'is_highway_speed', 'is_city_driving',
            'stop_and_go', 'extreme_cold', 'extreme_heat'
        ]
        
        self.feature_columns.extend(operational_features)
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Creating time window features...")
        df = self.create_time_windows(df)
        
        print("Creating cycle features...")
        df = self.create_cycle_features(df)
        
        print("Creating degradation indicators...")
        df = self.create_degradation_indicators(df)
        
        print("Creating operational features...")
        df = self.create_operational_features(df)
        
        # Remove intermediate columns
        intermediate_cols = [
            'cell_imbalance', 'discharge_current', 'charge_current', 'temp_gradient',
            'thermal_stress', 'low_soc', 'charging', 'charging_session_start',
            'fast_charging_start', 'charge_energy', 'soc_diff', 'discharge_event',
            'deep_cycles', 'voltage_drop', 'current_change', 'soh_change',
            'is_rest_period', 'energy_in', 'energy_out'
        ]
        
        df = df.drop(columns=[col for col in intermediate_cols if col in df.columns])
        
        print(f"Created {len(self.feature_columns)} engineered features")
        return df
    
    def prepare_model_data(self, df: pd.DataFrame, target_col: str = 'soh_percent') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_col])
        
        # Get feature columns (exclude metadata and targets)
        exclude_cols = [
            'vehicle_id', 'timestamp', 'charge_session_id', 'soh_percent', 
            'capacity_mAh', 'rul_days'
        ]
        
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by category for analysis"""
        groups = {
            'voltage': [col for col in self.feature_columns if 'voltage' in col],
            'current': [col for col in self.feature_columns if 'current' in col],
            'temperature': [col for col in self.feature_columns if 'temp' in col],
            'soc': [col for col in self.feature_columns if 'soc' in col],
            'charging': [col for col in self.feature_columns if 'charg' in col],
            'cycles': [col for col in self.feature_columns if 'cycle' in col],
            'degradation': [col for col in self.feature_columns if any(x in col for x in ['resistance', 'fade', 'efficiency'])],
            'operational': [col for col in self.feature_columns if any(x in col for x in ['hour', 'day', 'month', 'distance', 'speed'])],
            'environmental': [col for col in self.feature_columns if any(x in col for x in ['extreme', 'ambient'])]
        }
        return groups

def main():
    """Example usage"""
    # Load sample data
    df = pd.read_parquet('data/raw/vehicle_EV_0001.parquet')
    
    # Initialize feature engineer
    fe = BatteryFeatureEngineer()
    
    # Engineer features
    df_features = fe.engineer_all_features(df)
    
    # Prepare for modeling
    X, y = fe.prepare_model_data(df_features)
    
    print(f"Feature engineering complete!")
    print(f"Features created: {len(fe.feature_columns)}")
    print(f"Feature groups: {list(fe.get_feature_importance_groups().keys())}")
    
    # Save processed data
    df_features.to_parquet('data/processed/features_sample.parquet', index=False)

if __name__ == "__main__":
    main()