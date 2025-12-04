#!/usr/bin/env python3
"""
EV Battery Telemetry Data Simulator
Generates realistic BMS data with degradation patterns, edge cases, and labels.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path
import boto3
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EVBatterySimulator:
    def __init__(self, n_vehicles: int = 1000, years: int = 3, resolution_minutes: int = 5):
        self.n_vehicles = n_vehicles
        self.years = years
        self.resolution_minutes = resolution_minutes
        self.samples_per_day = 24 * 60 // resolution_minutes
        self.total_days = years * 365
        
        # Battery parameters
        self.nominal_capacity = 75000  # mAh (75 kWh pack)
        self.nominal_voltage = 400  # V
        self.cell_count = 96
        self.cell_nominal_voltage = 4.2  # V
        
    def generate_vehicle_profile(self, vehicle_id: str) -> Dict:
        """Generate unique characteristics for each vehicle"""
        np.random.seed(hash(vehicle_id) % 2**32)
        
        return {
            'vehicle_id': vehicle_id,
            'manufacturing_date': datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 365)),
            'usage_pattern': np.random.choice(['urban', 'highway', 'mixed'], p=[0.3, 0.2, 0.5]),
            'climate': np.random.choice(['hot', 'cold', 'moderate'], p=[0.2, 0.2, 0.6]),
            'fast_charge_frequency': np.random.beta(2, 5),  # 0-1, lower = less frequent
            'driving_aggressiveness': np.random.beta(2, 3),  # 0-1
            'initial_capacity': self.nominal_capacity * np.random.normal(1.0, 0.02),  # ±2% variation
        }
    
    def simulate_degradation(self, days: int, profile: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate battery capacity degradation over time"""
        # Calendar aging (time-based)
        calendar_rate = 0.05 / 365  # 5% per year base rate
        
        # Cycle aging factors
        temp_factor = {'hot': 1.5, 'cold': 1.2, 'moderate': 1.0}[profile['climate']]
        usage_factor = {'urban': 1.3, 'highway': 0.9, 'mixed': 1.0}[profile['usage_pattern']]
        fast_charge_factor = 1 + profile['fast_charge_frequency'] * 0.3
        
        # Generate daily degradation
        daily_degradation = []
        cumulative_cycles = 0
        
        for day in range(days):
            # Calendar aging
            cal_deg = calendar_rate * temp_factor
            
            # Cycle aging (varies by day)
            daily_cycles = np.random.poisson(1.2)  # Average 1.2 cycles per day
            cumulative_cycles += daily_cycles
            
            cycle_deg = (daily_cycles * 0.01 / 1000) * usage_factor * fast_charge_factor  # 0.01% per 1000 cycles
            
            # Random events (thermal stress, deep discharge)
            if np.random.random() < 0.01:  # 1% chance of stress event
                stress_deg = np.random.exponential(0.002)
            else:
                stress_deg = 0
                
            total_deg = cal_deg + cycle_deg + stress_deg
            daily_degradation.append(total_deg)
        
        # Convert to capacity retention
        cumulative_degradation = np.cumsum(daily_degradation)
        capacity_retention = np.exp(-cumulative_degradation)  # Exponential decay
        
        # Add noise
        noise = np.random.normal(0, 0.005, len(capacity_retention))
        capacity_retention = np.clip(capacity_retention + noise, 0.5, 1.0)
        
        return capacity_retention, cumulative_cycles
    
    def generate_telemetry(self, vehicle_id: str, profile: Dict) -> pd.DataFrame:
        """Generate minute-by-minute telemetry for one vehicle"""
        print(f"Generating telemetry for vehicle {vehicle_id}")
        
        # Generate degradation curve
        capacity_retention, total_cycles = self.simulate_degradation(self.total_days, profile)
        
        # Create time index
        start_date = profile['manufacturing_date']
        timestamps = pd.date_range(
            start=start_date, 
            periods=self.total_days * self.samples_per_day,
            freq=f'{self.resolution_minutes}min'
        )
        
        n_samples = len(timestamps)
        data = []
        
        # Generate daily patterns
        for day_idx in range(self.total_days):
            day_start = day_idx * self.samples_per_day
            day_end = (day_idx + 1) * self.samples_per_day
            day_timestamps = timestamps[day_start:day_end]
            
            # Daily capacity
            daily_capacity = profile['initial_capacity'] * capacity_retention[day_idx]
            
            # Generate driving/charging pattern for the day
            soc_profile = self._generate_daily_soc_profile(profile)
            
            for i, ts in enumerate(day_timestamps):
                # SOC from daily profile
                soc = soc_profile[i % len(soc_profile)]
                
                # Pack voltage (function of SOC and degradation)
                pack_voltage = self._calculate_pack_voltage(soc, capacity_retention[day_idx])
                
                # Cell voltages (with imbalance)
                cell_v_min, cell_v_max = self._calculate_cell_voltages(pack_voltage, day_idx)
                
                # Current (charging/discharging)
                current, charging_power, is_fast_charge = self._calculate_current(
                    soc, ts.hour, profile
                )
                
                # Temperature
                pack_temp, ambient_temp = self._calculate_temperatures(
                    current, ts, profile['climate']
                )
                
                # Vehicle speed
                vehicle_speed = self._calculate_speed(ts.hour, current)
                
                # Charge session ID
                charge_session_id = self._get_charge_session_id(current, ts)
                
                # Add sensor noise and edge cases
                if np.random.random() < 0.001:  # 0.1% sensor dropout
                    pack_voltage = np.nan
                
                data.append({
                    'vehicle_id': vehicle_id,
                    'timestamp': ts,
                    'pack_voltage': pack_voltage,
                    'cell_voltage_min': cell_v_min,
                    'cell_voltage_max': cell_v_max,
                    'pack_current': current,
                    'soc': soc,
                    'pack_temp': pack_temp,
                    'ambient_temp': ambient_temp,
                    'charging_power': charging_power,
                    'vehicle_speed': vehicle_speed,
                    'charge_session_id': charge_session_id,
                    'is_fast_charge': is_fast_charge,
                    'soh_percent': capacity_retention[day_idx] * 100,
                    'capacity_mAh': daily_capacity,
                    'rul_days': self._calculate_rul(capacity_retention[day_idx:])
                })
        
        return pd.DataFrame(data)
    
    def _generate_daily_soc_profile(self, profile: Dict) -> np.ndarray:
        """Generate realistic daily SOC profile"""
        # Start at random SOC
        soc = np.random.uniform(0.2, 0.9)
        soc_profile = []
        
        for hour in range(24):
            # Driving periods (discharge)
            if hour in [7, 8, 17, 18, 19]:  # Commute hours
                discharge_rate = profile['driving_aggressiveness'] * 0.05
                soc = max(0.1, soc - discharge_rate)
            
            # Charging periods
            elif hour in [22, 23, 0, 1, 2, 3, 4, 5, 6]:  # Overnight charging
                if soc < 0.8:
                    charge_rate = 0.1 if np.random.random() < profile['fast_charge_frequency'] else 0.05
                    soc = min(0.95, soc + charge_rate)
            
            # Repeat for each sample in the hour
            samples_per_hour = self.samples_per_day // 24
            soc_profile.extend([soc] * samples_per_hour)
        
        return np.array(soc_profile)
    
    def _calculate_pack_voltage(self, soc: float, degradation: float) -> float:
        """Calculate pack voltage based on SOC and degradation"""
        # Voltage curve (simplified)
        base_voltage = 300 + (soc * 100)  # 300-400V range
        degradation_factor = 0.95 + (degradation * 0.05)  # Slight voltage drop with age
        return base_voltage * degradation_factor + np.random.normal(0, 2)
    
    def _calculate_cell_voltages(self, pack_voltage: float, day: int) -> Tuple[float, float]:
        """Calculate min/max cell voltages with imbalance"""
        avg_cell_voltage = pack_voltage / self.cell_count
        imbalance = 0.01 + (day / (self.total_days * 365)) * 0.05  # Increasing imbalance
        
        cell_min = avg_cell_voltage * (1 - imbalance) + np.random.normal(0, 0.01)
        cell_max = avg_cell_voltage * (1 + imbalance) + np.random.normal(0, 0.01)
        
        return cell_min, cell_max
    
    def _calculate_current(self, soc: float, hour: int, profile: Dict) -> Tuple[float, float, bool]:
        """Calculate pack current and charging power"""
        # Charging periods
        if hour in [22, 23, 0, 1, 2, 3, 4, 5, 6] and soc < 0.9:
            is_fast_charge = np.random.random() < profile['fast_charge_frequency']
            if is_fast_charge:
                current = np.random.uniform(100, 200)  # Fast charging
                power = current * 400 / 1000  # kW
            else:
                current = np.random.uniform(20, 50)  # Slow charging
                power = current * 400 / 1000
        
        # Driving periods (discharge)
        elif hour in [7, 8, 17, 18, 19]:
            current = -np.random.uniform(50, 150) * profile['driving_aggressiveness']
            power = 0
            is_fast_charge = False
        
        # Idle
        else:
            current = np.random.normal(0, 5)  # Small parasitic load
            power = 0
            is_fast_charge = False
        
        return current, power, is_fast_charge
    
    def _calculate_temperatures(self, current: float, timestamp: datetime, climate: str) -> Tuple[float, float]:
        """Calculate pack and ambient temperatures"""
        # Ambient temperature (seasonal + daily variation)
        base_temp = {'hot': 30, 'cold': 5, 'moderate': 20}[climate]
        seasonal_var = 10 * np.sin(2 * np.pi * timestamp.timetuple().tm_yday / 365)
        daily_var = 5 * np.sin(2 * np.pi * timestamp.hour / 24)
        ambient_temp = base_temp + seasonal_var + daily_var + np.random.normal(0, 2)
        
        # Pack temperature (ambient + heating from current)
        heating = abs(current) * 0.1  # Heating from current
        pack_temp = ambient_temp + heating + np.random.normal(0, 1)
        
        return pack_temp, ambient_temp
    
    def _calculate_speed(self, hour: int, current: float) -> float:
        """Calculate vehicle speed"""
        if current < -10:  # Discharging (driving)
            if hour in [7, 8, 17, 18, 19]:  # Rush hour
                return np.random.uniform(20, 80)
            else:
                return np.random.uniform(40, 120)
        else:
            return 0  # Parked
    
    def _get_charge_session_id(self, current: float, timestamp: datetime) -> str:
        """Generate charge session ID"""
        if current > 10:  # Charging
            # Create session ID based on date and hour
            return f"session_{timestamp.strftime('%Y%m%d_%H')}"
        return ""
    
    def _calculate_rul(self, future_retention: np.ndarray) -> int:
        """Calculate remaining useful life (days until SOH < 80%)"""
        eol_indices = np.where(future_retention < 0.8)[0]
        if len(eol_indices) > 0:
            return eol_indices[0]
        else:
            return len(future_retention)  # Beyond simulation period
    
    def generate_all_vehicles(self, output_dir: str = "data/raw") -> None:
        """Generate telemetry for all vehicles and save to parquet"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = []
        
        for i in range(self.n_vehicles):
            vehicle_id = f"EV_{i:04d}"
            profile = self.generate_vehicle_profile(vehicle_id)
            
            # Generate telemetry
            vehicle_data = self.generate_telemetry(vehicle_id, profile)
            all_data.append(vehicle_data)
            
            # Save individual vehicle file
            vehicle_file = f"{output_dir}/vehicle_{vehicle_id}.parquet"
            vehicle_data.to_parquet(vehicle_file, index=False)
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{self.n_vehicles} vehicles")
        
        # Combine all data
        print("Combining all vehicle data...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Save partitioned by year-month
        combined_data['year_month'] = combined_data['timestamp'].dt.to_period('M')
        
        for period, group in combined_data.groupby('year_month'):
            period_dir = f"{output_dir}/year_month={period}"
            os.makedirs(period_dir, exist_ok=True)
            group.drop('year_month', axis=1).to_parquet(
                f"{period_dir}/data.parquet", index=False
            )
        
        print(f"Generated {len(combined_data):,} records for {self.n_vehicles} vehicles")
        print(f"Data saved to {output_dir}")
        
        # Generate summary statistics
        self._generate_summary_stats(combined_data, output_dir)
    
    def _generate_summary_stats(self, data: pd.DataFrame, output_dir: str) -> None:
        """Generate and save summary statistics"""
        stats = {
            'total_vehicles': int(data['vehicle_id'].nunique()),
            'total_records': int(len(data)),
            'date_range': {
                'start': data['timestamp'].min().isoformat(),
                'end': data['timestamp'].max().isoformat()
            },
            'soh_distribution': {
                'mean': float(data['soh_percent'].mean()),
                'std': float(data['soh_percent'].std()),
                'min': float(data['soh_percent'].min()),
                'max': float(data['soh_percent'].max())
            },
            'rul_distribution': {
                'mean': float(data['rul_days'].mean()),
                'std': float(data['rul_days'].std()),
                'min': float(data['rul_days'].min()),
                'max': float(data['rul_days'].max())
            },
            'edge_cases': {
                'sensor_dropouts': int(data['pack_voltage'].isna().sum()),
                'fast_charge_sessions': int(data['is_fast_charge'].sum()),
                'extreme_temps': int(((data['pack_temp'] > 50) | (data['pack_temp'] < -10)).sum())
            }
        }
        
        import json
        with open(f"{output_dir}/summary_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Summary statistics saved to {output_dir}/summary_stats.json")

def upload_to_s3(local_dir: str, bucket: str, prefix: str = "ev-battery-data"):
    """Upload generated data to S3"""
    try:
        s3 = boto3.client('s3')
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{prefix}/{relative_path}".replace("\\", "/")
                
                print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                s3.upload_file(local_path, bucket, s3_key)
        
        print(f"Data uploaded to S3 bucket: {bucket}")
        
    except Exception as e:
        print(f"S3 upload failed: {e}")
        print("Continuing with local data only...")

def main():
    parser = argparse.ArgumentParser(description='Generate EV battery telemetry data')
    parser.add_argument('--vehicles', type=int, default=100, help='Number of vehicles')
    parser.add_argument('--years', type=int, default=3, help='Years of data per vehicle')
    parser.add_argument('--resolution', type=int, default=5, help='Resolution in minutes')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory')
    parser.add_argument('--s3-bucket', help='S3 bucket for upload (optional)')
    
    args = parser.parse_args()
    
    print(f"Generating data for {args.vehicles} vehicles over {args.years} years")
    print(f"Resolution: {args.resolution} minutes")
    
    # Generate data
    simulator = EVBatterySimulator(
        n_vehicles=args.vehicles,
        years=args.years,
        resolution_minutes=args.resolution
    )
    
    simulator.generate_all_vehicles(args.output_dir)
    
    # Upload to S3 if specified
    if args.s3_bucket:
        upload_to_s3(args.output_dir, args.s3_bucket)
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()