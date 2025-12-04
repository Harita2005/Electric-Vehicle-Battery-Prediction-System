#!/usr/bin/env python3
"""
Data Drift Detection for EV Battery Prediction Models
Monitors feature distributions and model performance over time.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BatteryDriftDetector:
    def __init__(self, s3_bucket: str, reference_period_days: int = 30):
        self.s3_bucket = s3_bucket
        self.reference_period_days = reference_period_days
        self.s3_client = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'ks_test_pvalue': 0.05,      # Kolmogorov-Smirnov test
            'psi_threshold': 0.2,        # Population Stability Index
            'performance_degradation': 0.1  # 10% increase in MAE
        }
    
    def load_reference_data(self) -> pd.DataFrame:
        """Load reference dataset for comparison"""
        try:
            # Load recent data as reference
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.reference_period_days)
            
            # In practice, load from S3 partitioned data
            # For demo, generate mock reference data
            reference_data = self._generate_mock_reference_data()
            
            print(f"Loaded reference data: {len(reference_data)} samples")
            return reference_data
            
        except Exception as e:
            print(f"Error loading reference data: {e}")
            return pd.DataFrame()
    
    def load_current_data(self) -> pd.DataFrame:
        """Load current dataset for drift detection"""
        try:
            # Load last 7 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # In practice, load from S3 partitioned data
            # For demo, generate mock current data with some drift
            current_data = self._generate_mock_current_data()
            
            print(f"Loaded current data: {len(current_data)} samples")
            return current_data
            
        except Exception as e:
            print(f"Error loading current data: {e}")
            return pd.DataFrame()
    
    def _generate_mock_reference_data(self) -> pd.DataFrame:
        """Generate mock reference data"""
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'pack_temp_max_30d': np.random.normal(35, 8, n_samples),
            'fast_charge_sessions_30d': np.random.poisson(6, n_samples),
            'cumulative_cycles': np.random.normal(1000, 300, n_samples),
            'vehicle_age_years': np.random.uniform(0.5, 5, n_samples),
            'soc_std_30d': np.random.exponential(5, n_samples),
            'current_mean_7d': np.random.normal(-20, 15, n_samples),
            'soh_percent': np.random.normal(88, 5, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _generate_mock_current_data(self) -> pd.DataFrame:
        """Generate mock current data with some drift"""
        np.random.seed(123)  # Different seed for drift
        n_samples = 2000
        
        # Introduce drift in some features
        data = {
            'pack_temp_max_30d': np.random.normal(38, 9, n_samples),  # Higher temp (drift)
            'fast_charge_sessions_30d': np.random.poisson(8, n_samples),  # More fast charging (drift)
            'cumulative_cycles': np.random.normal(1200, 350, n_samples),  # More cycles
            'vehicle_age_years': np.random.uniform(1, 5.5, n_samples),  # Older vehicles
            'soc_std_30d': np.random.exponential(5.2, n_samples),  # Slight drift
            'current_mean_7d': np.random.normal(-22, 16, n_samples),  # Slight drift
            'soh_percent': np.random.normal(86, 5.5, n_samples)  # Lower SoH
        }
        
        return pd.DataFrame(data)
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / len(reference)
            cur_props = cur_counts / len(current)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            cur_props = np.where(cur_props == 0, 0.0001, cur_props)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            
            return psi
            
        except Exception as e:
            print(f"Error calculating PSI: {e}")
            return 0.0
    
    def detect_feature_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
        """Detect drift in individual features"""
        drift_results = {}
        
        # Get common features
        common_features = set(reference_data.columns) & set(current_data.columns)
        common_features.discard('soh_percent')  # Exclude target variable
        
        for feature in common_features:
            ref_values = reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)
            
            # Population Stability Index
            psi = self.calculate_psi(ref_values.values, cur_values.values)
            
            # Statistical measures
            ref_mean, ref_std = ref_values.mean(), ref_values.std()
            cur_mean, cur_std = cur_values.mean(), cur_values.std()
            
            mean_shift = abs(cur_mean - ref_mean) / ref_std if ref_std > 0 else 0
            std_ratio = cur_std / ref_std if ref_std > 0 else 1
            
            # Determine drift status
            has_drift = (
                ks_pvalue < self.drift_thresholds['ks_test_pvalue'] or
                psi > self.drift_thresholds['psi_threshold']
            )
            
            drift_results[feature] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'psi': psi,
                'mean_shift': mean_shift,
                'std_ratio': std_ratio,
                'ref_mean': ref_mean,
                'cur_mean': cur_mean,
                'ref_std': ref_std,
                'cur_std': cur_std,
                'has_drift': has_drift,
                'drift_severity': self._classify_drift_severity(psi, ks_pvalue)
            }
        
        return drift_results
    
    def _classify_drift_severity(self, psi: float, ks_pvalue: float) -> str:
        """Classify drift severity"""
        if psi > 0.5 or ks_pvalue < 0.001:
            return 'high'
        elif psi > 0.2 or ks_pvalue < 0.01:
            return 'medium'
        elif psi > 0.1 or ks_pvalue < 0.05:
            return 'low'
        else:
            return 'none'
    
    def detect_performance_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
        """Detect drift in model performance"""
        try:
            # Mock model predictions (in practice, load from model artifacts)
            ref_predictions = self._mock_model_predictions(reference_data)
            cur_predictions = self._mock_model_predictions(current_data)
            
            # Calculate performance metrics
            ref_mae = mean_absolute_error(reference_data['soh_percent'], ref_predictions)
            cur_mae = mean_absolute_error(current_data['soh_percent'], cur_predictions)
            
            ref_rmse = np.sqrt(mean_squared_error(reference_data['soh_percent'], ref_predictions))
            cur_rmse = np.sqrt(mean_squared_error(current_data['soh_percent'], cur_predictions))
            
            # Calculate performance degradation
            mae_degradation = (cur_mae - ref_mae) / ref_mae if ref_mae > 0 else 0
            rmse_degradation = (cur_rmse - ref_rmse) / ref_rmse if ref_rmse > 0 else 0
            
            # Determine if performance has degraded significantly
            has_performance_drift = mae_degradation > self.drift_thresholds['performance_degradation']
            
            return {
                'reference_mae': ref_mae,
                'current_mae': cur_mae,
                'reference_rmse': ref_rmse,
                'current_rmse': cur_rmse,
                'mae_degradation': mae_degradation,
                'rmse_degradation': rmse_degradation,
                'has_performance_drift': has_performance_drift,
                'performance_severity': 'high' if mae_degradation > 0.2 else 'medium' if mae_degradation > 0.1 else 'low'
            }
            
        except Exception as e:
            print(f"Error detecting performance drift: {e}")
            return {}
    
    def _mock_model_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """Mock model predictions for demonstration"""
        # Simple linear model for demo
        predictions = (
            85 + 
            data['vehicle_age_years'] * -2 +
            (data['pack_temp_max_30d'] - 35) * -0.5 +
            data['fast_charge_sessions_30d'] * -0.3 +
            np.random.normal(0, 1, len(data))
        )
        return predictions
    
    def generate_drift_report(self, feature_drift: Dict, performance_drift: Dict) -> Dict:
        """Generate comprehensive drift report"""
        # Count drift by severity
        drift_summary = {
            'high': 0,
            'medium': 0,
            'low': 0,
            'none': 0
        }
        
        drifted_features = []
        for feature, results in feature_drift.items():
            severity = results['drift_severity']
            drift_summary[severity] += 1
            
            if results['has_drift']:
                drifted_features.append({
                    'feature': feature,
                    'severity': severity,
                    'psi': results['psi'],
                    'ks_pvalue': results['ks_pvalue'],
                    'mean_shift': results['mean_shift']
                })
        
        # Overall drift status
        overall_drift = (
            drift_summary['high'] > 0 or 
            drift_summary['medium'] > 2 or
            performance_drift.get('has_performance_drift', False)
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': overall_drift,
            'drift_summary': drift_summary,
            'drifted_features': drifted_features,
            'performance_drift': performance_drift,
            'total_features_analyzed': len(feature_drift),
            'recommendation': self._get_recommendation(overall_drift, drift_summary, performance_drift)
        }
        
        return report
    
    def _get_recommendation(self, overall_drift: bool, drift_summary: Dict, performance_drift: Dict) -> str:
        """Get recommendation based on drift analysis"""
        if performance_drift.get('has_performance_drift', False):
            return "URGENT: Model retraining required due to performance degradation"
        elif drift_summary['high'] > 0:
            return "Model retraining recommended due to high feature drift"
        elif drift_summary['medium'] > 2:
            return "Monitor closely - multiple features showing medium drift"
        elif overall_drift:
            return "Continue monitoring - some drift detected but not critical"
        else:
            return "No action required - model performance stable"
    
    def send_cloudwatch_metrics(self, report: Dict) -> None:
        """Send drift metrics to CloudWatch"""
        try:
            metrics = []
            
            # Overall drift metric
            metrics.append({
                'MetricName': 'OverallDrift',
                'Value': 1 if report['overall_drift_detected'] else 0,
                'Unit': 'Count'
            })
            
            # Drift by severity
            for severity, count in report['drift_summary'].items():
                metrics.append({
                    'MetricName': f'DriftSeverity_{severity.title()}',
                    'Value': count,
                    'Unit': 'Count'
                })
            
            # Performance metrics
            if 'performance_drift' in report:
                perf = report['performance_drift']
                if 'mae_degradation' in perf:
                    metrics.append({
                        'MetricName': 'MAEDegradation',
                        'Value': perf['mae_degradation'],
                        'Unit': 'Percent'
                    })
            
            # Send metrics
            self.cloudwatch.put_metric_data(
                Namespace='EVBattery/DriftDetection',
                MetricData=metrics
            )
            
            print(f"Sent {len(metrics)} metrics to CloudWatch")
            
        except Exception as e:
            print(f"Error sending CloudWatch metrics: {e}")
    
    def save_report(self, report: Dict, s3_key: str = None) -> None:
        """Save drift report to S3"""
        try:
            if s3_key is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                s3_key = f"drift_reports/drift_report_{timestamp}.json"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(report, indent=2, default=str),
                ContentType='application/json'
            )
            
            print(f"Drift report saved to s3://{self.s3_bucket}/{s3_key}")
            
        except Exception as e:
            print(f"Error saving report to S3: {e}")
    
    def run_drift_detection(self) -> Dict:
        """Run complete drift detection pipeline"""
        print("Starting drift detection...")
        
        # Load data
        reference_data = self.load_reference_data()
        current_data = self.load_current_data()
        
        if reference_data.empty or current_data.empty:
            print("Insufficient data for drift detection")
            return {}
        
        # Detect feature drift
        print("Detecting feature drift...")
        feature_drift = self.detect_feature_drift(reference_data, current_data)
        
        # Detect performance drift
        print("Detecting performance drift...")
        performance_drift = self.detect_performance_drift(reference_data, current_data)
        
        # Generate report
        print("Generating drift report...")
        report = self.generate_drift_report(feature_drift, performance_drift)
        
        # Send metrics and save report
        self.send_cloudwatch_metrics(report)
        self.save_report(report)
        
        print(f"Drift detection complete. Overall drift: {report['overall_drift_detected']}")
        return report

def main():
    """Main function for standalone execution"""
    detector = BatteryDriftDetector(s3_bucket="ev-battery-processed-data")
    report = detector.run_drift_detection()
    
    # Print summary
    print("\n" + "="*50)
    print("DRIFT DETECTION SUMMARY")
    print("="*50)
    print(f"Overall Drift Detected: {report.get('overall_drift_detected', 'Unknown')}")
    print(f"Features Analyzed: {report.get('total_features_analyzed', 0)}")
    print(f"Recommendation: {report.get('recommendation', 'No recommendation')}")
    
    if 'drifted_features' in report:
        print(f"\nDrifted Features ({len(report['drifted_features'])}):")
        for feature in report['drifted_features']:
            print(f"  - {feature['feature']}: {feature['severity']} severity (PSI: {feature['psi']:.3f})")

if __name__ == "__main__":
    main()