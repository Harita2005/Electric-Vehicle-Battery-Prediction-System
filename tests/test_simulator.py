#!/usr/bin/env python3
"""
Unit tests for EV Battery Data Simulator
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.generate_data import EVBatterySimulator

class TestEVBatterySimulator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = EVBatterySimulator(n_vehicles=2, years=1, resolution_minutes=60)
        self.temp_dir = tempfile.mkdtemp()
    
    def test_vehicle_profile_generation(self):
        """Test vehicle profile generation"""
        profile = self.simulator.generate_vehicle_profile("TEST_001")
        
        # Check required fields
        required_fields = [
            'vehicle_id', 'manufacturing_date', 'usage_pattern', 
            'climate', 'fast_charge_frequency', 'driving_aggressiveness', 'initial_capacity'
        ]
        
        for field in required_fields:
            self.assertIn(field, profile)
        
        # Check value ranges
        self.assertEqual(profile['vehicle_id'], "TEST_001")
        self.assertIn(profile['usage_pattern'], ['urban', 'highway', 'mixed'])
        self.assertIn(profile['climate'], ['hot', 'cold', 'moderate'])
        self.assertGreaterEqual(profile['fast_charge_frequency'], 0)
        self.assertLessEqual(profile['fast_charge_frequency'], 1)
        self.assertGreater(profile['initial_capacity'], 0)
    
    def test_degradation_simulation(self):
        """Test battery degradation simulation"""
        profile = self.simulator.generate_vehicle_profile("TEST_001")
        capacity_retention, cycles = self.simulator.simulate_degradation(365, profile)
        
        # Check output shapes
        self.assertEqual(len(capacity_retention), 365)
        self.assertIsInstance(cycles, (int, float))
        
        # Check degradation trend (should generally decrease)
        self.assertLessEqual(capacity_retention[-1], capacity_retention[0])
        
        # Check value ranges
        self.assertTrue(all(0.5 <= x <= 1.0 for x in capacity_retention))
        self.assertGreaterEqual(cycles, 0)
    
    def test_telemetry_generation(self):
        """Test telemetry data generation"""
        profile = self.simulator.generate_vehicle_profile("TEST_001")
        telemetry = self.simulator.generate_telemetry("TEST_001", profile)
        
        # Check data structure
        self.assertIsInstance(telemetry, pd.DataFrame)
        self.assertGreater(len(telemetry), 0)
        
        # Check required columns
        required_columns = [
            'vehicle_id', 'timestamp', 'pack_voltage', 'cell_voltage_min', 
            'cell_voltage_max', 'pack_current', 'soc', 'pack_temp', 
            'ambient_temp', 'charging_power', 'vehicle_speed', 
            'charge_session_id', 'is_fast_charge', 'soh_percent', 
            'capacity_mAh', 'rul_days'
        ]
        
        for col in required_columns:
            self.assertIn(col, telemetry.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(telemetry['timestamp']))
        self.assertTrue(pd.api.types.is_numeric_dtype(telemetry['soh_percent']))
        self.assertTrue(pd.api.types.is_bool_dtype(telemetry['is_fast_charge']))
        
        # Check value ranges
        self.assertTrue(all(0 <= x <= 100 for x in telemetry['soh_percent'].dropna()))
        self.assertTrue(all(0 <= x <= 1 for x in telemetry['soc'].dropna()))
        self.assertTrue(all(x >= 0 for x in telemetry['rul_days'].dropna()))
    
    def test_data_consistency(self):
        """Test data consistency and relationships"""
        profile = self.simulator.generate_vehicle_profile("TEST_001")
        telemetry = self.simulator.generate_telemetry("TEST_001", profile)
        
        # Check that all records have the same vehicle_id
        self.assertTrue(all(telemetry['vehicle_id'] == "TEST_001"))
        
        # Check timestamp ordering
        timestamps = telemetry['timestamp'].values
        self.assertTrue(all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)))
        
        # Check cell voltage relationships
        valid_voltages = telemetry.dropna(subset=['cell_voltage_min', 'cell_voltage_max'])
        if len(valid_voltages) > 0:
            self.assertTrue(all(
                valid_voltages['cell_voltage_min'] <= valid_voltages['cell_voltage_max']
            ))
        
        # Check SoH degradation trend (should generally decrease over time)
        soh_values = telemetry['soh_percent'].dropna()
        if len(soh_values) > 10:
            # Use rolling average to smooth noise
            soh_smooth = soh_values.rolling(window=min(10, len(soh_values)//2)).mean()
            trend = soh_smooth.iloc[-1] - soh_smooth.iloc[0]
            self.assertLessEqual(trend, 5)  # Allow some tolerance for noise
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        profile = self.simulator.generate_vehicle_profile("TEST_001")
        telemetry = self.simulator.generate_telemetry("TEST_001", profile)
        
        # Check for sensor dropouts (NaN values)
        nan_count = telemetry['pack_voltage'].isna().sum()
        total_count = len(telemetry)
        nan_percentage = nan_count / total_count
        
        # Should have some sensor dropouts but not too many
        self.assertLess(nan_percentage, 0.01)  # Less than 1%
        
        # Check for fast charging events
        fast_charge_count = telemetry['is_fast_charge'].sum()
        self.assertGreaterEqual(fast_charge_count, 0)
        
        # Check for charging sessions
        charging_sessions = telemetry['charge_session_id'].dropna().nunique()
        self.assertGreater(charging_sessions, 0)
    
    def test_multiple_vehicles(self):
        """Test generation of multiple vehicles"""
        # Use small dataset for testing
        test_simulator = EVBatterySimulator(n_vehicles=3, years=1, resolution_minutes=1440)  # Daily resolution
        
        # Generate data for multiple vehicles
        all_data = []
        for i in range(3):
            vehicle_id = f"TEST_{i:03d}"
            profile = test_simulator.generate_vehicle_profile(vehicle_id)
            vehicle_data = test_simulator.generate_telemetry(vehicle_id, profile)
            all_data.append(vehicle_data)
        
        # Combine data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Check that we have data for all vehicles
        unique_vehicles = combined_data['vehicle_id'].nunique()
        self.assertEqual(unique_vehicles, 3)
        
        # Check that each vehicle has reasonable amount of data
        for i in range(3):
            vehicle_id = f"TEST_{i:03d}"
            vehicle_data = combined_data[combined_data['vehicle_id'] == vehicle_id]
            self.assertGreater(len(vehicle_data), 300)  # At least 300 days of data
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class TestDataQuality(unittest.TestCase):
    """Test data quality and statistical properties"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = EVBatterySimulator(n_vehicles=1, years=2, resolution_minutes=60)
        self.profile = self.simulator.generate_vehicle_profile("QUALITY_TEST")
        self.telemetry = self.simulator.generate_telemetry("QUALITY_TEST", self.profile)
    
    def test_statistical_properties(self):
        """Test statistical properties of generated data"""
        # Temperature should be reasonable
        temp_mean = self.telemetry['pack_temp'].mean()
        self.assertGreater(temp_mean, 10)  # Above 10°C
        self.assertLess(temp_mean, 50)     # Below 50°C
        
        # SOC should vary reasonably
        soc_std = self.telemetry['soc'].std()
        self.assertGreater(soc_std, 0.1)   # Some variation
        self.assertLess(soc_std, 0.4)      # Not too much variation
        
        # Voltage should be in reasonable range
        voltage_mean = self.telemetry['pack_voltage'].mean()
        self.assertGreater(voltage_mean, 300)  # Above 300V
        self.assertLess(voltage_mean, 450)     # Below 450V
    
    def test_correlation_structure(self):
        """Test expected correlations in the data"""
        # SOC and voltage should be positively correlated
        soc_voltage_corr = self.telemetry['soc'].corr(self.telemetry['pack_voltage'])
        self.assertGreater(soc_voltage_corr, 0.3)
        
        # SoH should generally decrease over time
        self.telemetry['time_index'] = range(len(self.telemetry))
        soh_time_corr = self.telemetry['soh_percent'].corr(self.telemetry['time_index'])
        self.assertLess(soh_time_corr, 0)  # Negative correlation (degradation)

if __name__ == '__main__':
    unittest.main()