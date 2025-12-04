#!/usr/bin/env python3
"""
Unit tests for Feature Engineering Pipeline
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.feature_engineering import BatteryFeatureEngineer

class TestBatteryFeatureEngineer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.fe = BatteryFeatureEngineer()
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample battery data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create time series data
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
        
        data = {
            'vehicle_id': ['TEST_001'] * n_samples,
            'timestamp': timestamps,
            'pack_voltage': np.random.normal(380, 20, n_samples),
            'cell_voltage_min': np.random.normal(3.8, 0.2, n_samples),
            'cell_voltage_max': np.random.normal(4.0, 0.2, n_samples),
            'pack_current': np.random.normal(-20, 30, n_samples),
            'soc': np.random.uniform(0.2, 0.9, n_samples),
            'pack_temp': np.random.normal(30, 10, n_samples),
            'ambient_temp': np.random.normal(25, 8, n_samples),
            'charging_power': np.random.exponential(10, n_samples),
            'vehicle_speed': np.random.exponential(30, n_samples),
            'charge_session_id': [''] * n_samples,
            'is_fast_charge': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
            'soh_percent': 95 - np.arange(n_samples) * 0.01 + np.random.normal(0, 1, n_samples),
            'capacity_mAh': 75000 - np.arange(n_samples) * 5 + np.random.normal(0, 100, n_samples),
            'rul_days': 1000 - np.arange(n_samples) * 0.5 + np.random.normal(0, 50, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_time_window_features(self):
        """Test time window feature creation"""
        df_with_windows = self.fe.create_time_windows(self.sample_data.copy())
        
        # Check that new columns were created
        window_columns = [col for col in df_with_windows.columns if any(w in col for w in ['24h', '7d', '30d'])]
        self.assertGreater(len(window_columns), 0)
        
        # Check specific features
        expected_features = [
            'pack_voltage_mean_24h',
            'pack_voltage_std_7d',
            'cell_voltage_imbalance_mean_30d',
            'current_mean_24h',
            'pack_temp_max_7d',
            'soc_std_30d'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_with_windows.columns, f"Missing feature: {feature}")
        
        # Check that rolling features have reasonable values
        self.assertTrue(df_with_windows['pack_voltage_mean_24h'].notna().any())
        self.assertTrue(df_with_windows['pack_temp_max_7d'].notna().any())
    
    def test_cycle_features(self):
        """Test cycle-based feature creation"""
        df_with_cycles = self.fe.create_cycle_features(self.sample_data.copy())
        
        # Check that cycle features were created
        cycle_features = [
            'cumulative_cycles',
            'shallow_cycles',
            'cumulative_deep_cycles',
            'c_rate',
            'cumulative_high_c_events',
            'days_since_manufacture',
            'vehicle_age_years'
        ]
        
        for feature in cycle_features:
            self.assertIn(feature, df_with_cycles.columns, f"Missing cycle feature: {feature}")
        
        # Check value ranges
        self.assertTrue(all(df_with_cycles['cumulative_cycles'] >= 0))
        self.assertTrue(all(df_with_cycles['vehicle_age_years'] >= 0))
        self.assertTrue(all(df_with_cycles['days_since_manufacture'] >= 0))
    
    def test_degradation_indicators(self):
        """Test degradation indicator features"""
        df_with_degradation = self.fe.create_degradation_indicators(self.sample_data.copy())
        
        # Check that degradation features were created
        degradation_features = [
            'resistance_proxy',
            'resistance_trend_7d',
            'capacity_fade_rate',
            'voltage_recovery',
            'efficiency_7d'
        ]
        
        for feature in degradation_features:
            self.assertIn(feature, df_with_degradation.columns, f"Missing degradation feature: {feature}")
        
        # Check that some values are not NaN (after initial period)
        self.assertTrue(df_with_degradation['capacity_fade_rate'].notna().any())
    
    def test_operational_features(self):
        """Test operational feature creation"""
        df_with_operational = self.fe.create_operational_features(self.sample_data.copy())
        
        # Check time-based features
        time_features = [
            'hour_sin', 'hour_cos',
            'day_sin', 'day_cos',
            'month_sin', 'month_cos'
        ]
        
        for feature in time_features:
            self.assertIn(feature, df_with_operational.columns, f"Missing time feature: {feature}")
        
        # Check cyclical encoding ranges
        self.assertTrue(all(-1 <= x <= 1 for x in df_with_operational['hour_sin'].dropna()))
        self.assertTrue(all(-1 <= x <= 1 for x in df_with_operational['hour_cos'].dropna()))
        
        # Check operational features
        operational_features = [
            'daily_distance',
            'daily_energy_usage',
            'is_highway_speed',
            'is_city_driving',
            'extreme_cold',
            'extreme_heat'
        ]
        
        for feature in operational_features:
            self.assertIn(feature, df_with_operational.columns, f"Missing operational feature: {feature}")
    
    def test_full_feature_engineering(self):
        """Test complete feature engineering pipeline"""
        df_engineered = self.fe.engineer_all_features(self.sample_data.copy())
        
        # Check that we have more features than we started with
        original_features = len(self.sample_data.columns)
        engineered_features = len(df_engineered.columns)
        self.assertGreater(engineered_features, original_features)
        
        # Check that feature_columns list was populated
        self.assertGreater(len(self.fe.feature_columns), 0)
        
        # Check that original data is preserved
        self.assertIn('vehicle_id', df_engineered.columns)
        self.assertIn('timestamp', df_engineered.columns)
        self.assertIn('soh_percent', df_engineered.columns)
        
        # Check for no infinite values
        numeric_columns = df_engineered.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.assertFalse(np.isinf(df_engineered[col]).any(), f"Infinite values in {col}")
    
    def test_model_data_preparation(self):
        """Test model data preparation"""
        df_engineered = self.fe.engineer_all_features(self.sample_data.copy())
        X, y = self.fe.prepare_model_data(df_engineered)
        
        # Check shapes
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 10)  # Should have many features
        
        # Check that target variable is not in features
        self.assertNotIn('soh_percent', X.columns)
        self.assertNotIn('capacity_mAh', X.columns)
        self.assertNotIn('rul_days', X.columns)
        
        # Check for no missing values in features (should be filled)
        self.assertEqual(X.isnull().sum().sum(), 0)
        
        # Check target variable
        self.assertTrue(pd.api.types.is_numeric_dtype(y))
        self.assertTrue(all(0 <= val <= 100 for val in y.dropna()))
    
    def test_feature_importance_groups(self):
        """Test feature importance grouping"""
        # Run feature engineering to populate feature_columns
        self.fe.engineer_all_features(self.sample_data.copy())
        
        groups = self.fe.get_feature_importance_groups()
        
        # Check that groups are returned
        self.assertIsInstance(groups, dict)
        self.assertGreater(len(groups), 0)
        
        # Check expected groups
        expected_groups = ['voltage', 'current', 'temperature', 'soc', 'charging', 'cycles']
        for group in expected_groups:
            self.assertIn(group, groups)
        
        # Check that features are properly categorized
        all_grouped_features = []
        for group_features in groups.values():
            all_grouped_features.extend(group_features)
        
        # Should have some features in each major group
        self.assertGreater(len(groups['voltage']), 0)
        self.assertGreater(len(groups['temperature']), 0)
    
    def test_multiple_vehicles(self):
        """Test feature engineering with multiple vehicles"""
        # Create data for multiple vehicles
        multi_vehicle_data = []
        for i in range(3):
            vehicle_data = self.sample_data.copy()
            vehicle_data['vehicle_id'] = f'TEST_{i:03d}'
            multi_vehicle_data.append(vehicle_data)
        
        combined_data = pd.concat(multi_vehicle_data, ignore_index=True)
        
        # Run feature engineering
        df_engineered = self.fe.engineer_all_features(combined_data)
        
        # Check that all vehicles are present
        unique_vehicles = df_engineered['vehicle_id'].nunique()
        self.assertEqual(unique_vehicles, 3)
        
        # Check that features are calculated per vehicle
        for vehicle_id in df_engineered['vehicle_id'].unique():
            vehicle_data = df_engineered[df_engineered['vehicle_id'] == vehicle_id]
            
            # Should have rolling features for each vehicle
            self.assertTrue(vehicle_data['pack_voltage_mean_24h'].notna().any())
            self.assertTrue(vehicle_data['cumulative_cycles'].notna().any())
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with minimal data
        minimal_data = self.sample_data.head(10).copy()
        df_minimal = self.fe.engineer_all_features(minimal_data)
        self.assertGreater(len(df_minimal), 0)
        
        # Test with missing values
        data_with_na = self.sample_data.copy()
        data_with_na.loc[0:10, 'pack_voltage'] = np.nan
        data_with_na.loc[20:30, 'pack_temp'] = np.nan
        
        df_with_na = self.fe.engineer_all_features(data_with_na)
        self.assertGreater(len(df_with_na), 0)
        
        # Test model data preparation with missing target
        data_missing_target = df_with_na.copy()
        data_missing_target.loc[0:5, 'soh_percent'] = np.nan
        
        X, y = self.fe.prepare_model_data(data_missing_target)
        
        # Should remove rows with missing target
        self.assertLess(len(X), len(data_missing_target))
        self.assertEqual(len(X), len(y))
        self.assertFalse(y.isnull().any())

if __name__ == '__main__':
    unittest.main()