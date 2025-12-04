#!/usr/bin/env python3
"""
Baseline XGBoost Model for Battery SoH Prediction
Includes hyperparameter tuning, uncertainty estimation, and model persistence.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.xgboost
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from data_pipeline.feature_engineering import BatteryFeatureEngineer

class BaselineModel:
    def __init__(self, model_name: str = "xgboost_baseline"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = BatteryFeatureEngineer()
        self.feature_columns = None
        
    def prepare_data(self, data_path: str) -> tuple:
        """Load and prepare data for training"""
        print("Loading data...")
        
        # Load multiple vehicle files
        data_files = list(Path(data_path).glob("vehicle_*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No vehicle data files found in {data_path}")
        
        # Load and combine data
        dfs = []
        for file in data_files[:50]:  # Limit for demo
            df = pd.read_parquet(file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df):,} records from {len(dfs)} vehicles")
        
        # Engineer features
        print("Engineering features...")
        df_features = self.feature_engineer.engineer_all_features(combined_df)
        
        # Prepare model data
        X, y = self.feature_engineer.prepare_model_data(df_features, target_col='soh_percent')
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled, y, df_features
    
    def create_time_splits(self, df: pd.DataFrame, n_splits: int = 5) -> list:
        """Create time-based splits ensuring no vehicle leakage"""
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Get unique vehicles
        vehicles = df_sorted['vehicle_id'].unique()
        np.random.shuffle(vehicles)
        
        # Split vehicles into folds
        fold_size = len(vehicles) // n_splits
        splits = []
        
        for i in range(n_splits):
            # Test vehicles for this fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else len(vehicles)
            test_vehicles = vehicles[test_start:test_end]
            
            # Train vehicles (all others)
            train_vehicles = vehicles[~np.isin(vehicles, test_vehicles)]
            
            # Get indices
            train_idx = df_sorted[df_sorted['vehicle_id'].isin(train_vehicles)].index
            test_idx = df_sorted[df_sorted['vehicle_id'].isin(test_vehicles)].index
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def train_with_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> dict:
        """Train model with hyperparameter tuning"""
        print("Starting hyperparameter tuning...")
        
        # Create custom splits
        custom_splits = self.create_time_splits(df)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        # Grid search with custom CV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=custom_splits,
            scoring='neg_mean_absolute_error',
            n_jobs=2,  # Limit parallel jobs
            verbose=1
        )
        
        # Fit
        grid_search.fit(X, y)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f} MAE")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_quantile_models(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> dict:
        """Train quantile regression models for uncertainty estimation"""
        print("Training quantile regression models...")
        
        quantiles = [0.05, 0.5, 0.95]  # 90% prediction interval
        quantile_models = {}
        
        for q in quantiles:
            print(f"Training quantile {q}...")
            
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Use simple train/test split for quantile models
            train_vehicles = df['vehicle_id'].unique()[:int(len(df['vehicle_id'].unique()) * 0.8)]
            train_mask = df['vehicle_id'].isin(train_vehicles)
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            model.fit(X_train, y_train)
            quantile_models[q] = model
        
        self.quantile_models = quantile_models
        return quantile_models
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> dict:
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Create train/test split by vehicle
        vehicles = df['vehicle_id'].unique()
        np.random.seed(42)
        np.random.shuffle(vehicles)
        
        train_vehicles = vehicles[:int(len(vehicles) * 0.8)]
        test_vehicles = vehicles[int(len(vehicles) * 0.8):]
        
        train_mask = df['vehicle_id'].isin(train_vehicles)
        test_mask = df['vehicle_id'].isin(test_vehicles)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Quantile predictions for uncertainty
        if hasattr(self, 'quantile_models'):
            q05_pred = self.quantile_models[0.05].predict(X_test)
            q95_pred = self.quantile_models[0.95].predict(X_test)
            
            # Coverage (percentage of true values within prediction interval)
            coverage = np.mean((y_test >= q05_pred) & (y_test <= q95_pred))
            interval_width = np.mean(q95_pred - q05_pred)
        else:
            coverage = None
            interval_width = None
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'coverage': coverage,
            'interval_width': interval_width,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print(f"Model Performance:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        if coverage:
            print(f"  90% Coverage: {coverage:.4f}")
            print(f"  Interval Width: {interval_width:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> dict:
        """Make predictions with uncertainty estimates"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Main prediction
        y_pred = self.model.predict(X)
        
        # Uncertainty estimates
        if hasattr(self, 'quantile_models'):
            q05_pred = self.quantile_models[0.05].predict(X)
            q95_pred = self.quantile_models[0.95].predict(X)
            
            return {
                'prediction': y_pred,
                'lower_bound': q05_pred,
                'upper_bound': q95_pred,
                'uncertainty': (q95_pred - q05_pred) / 2
            }
        else:
            return {'prediction': y_pred}
    
    def save_model(self, output_dir: str = "models/artifacts") -> None:
        """Save trained model and preprocessing components"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save main model
        model_path = f"{output_dir}/{self.model_name}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = f"{output_dir}/{self.model_name}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature engineer
        fe_path = f"{output_dir}/{self.model_name}_feature_engineer.joblib"
        joblib.dump(self.feature_engineer, fe_path)
        
        # Save quantile models
        if hasattr(self, 'quantile_models'):
            for q, model in self.quantile_models.items():
                q_path = f"{output_dir}/{self.model_name}_quantile_{q}.joblib"
                joblib.dump(model, q_path)
        
        # Save feature columns
        with open(f"{output_dir}/{self.model_name}_features.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str = "models/artifacts") -> None:
        """Load trained model and preprocessing components"""
        # Load main model
        model_path = f"{model_dir}/{self.model_name}.joblib"
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = f"{model_dir}/{self.model_name}_scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        
        # Load feature engineer
        fe_path = f"{model_dir}/{self.model_name}_feature_engineer.joblib"
        self.feature_engineer = joblib.load(fe_path)
        
        # Load feature columns
        with open(f"{model_dir}/{self.model_name}_features.json", 'r') as f:
            self.feature_columns = json.load(f)
        
        # Load quantile models if they exist
        quantile_files = list(Path(model_dir).glob(f"{self.model_name}_quantile_*.joblib"))
        if quantile_files:
            self.quantile_models = {}
            for file in quantile_files:
                q = float(file.stem.split('_')[-1])
                self.quantile_models[q] = joblib.load(file)
        
        print(f"Model loaded from {model_dir}")

def main():
    """Train baseline model"""
    # Initialize MLflow
    mlflow.set_experiment("ev_battery_baseline")
    
    with mlflow.start_run():
        # Initialize model
        model = BaselineModel()
        
        # Prepare data
        X, y, df = model.prepare_data("data/raw")
        
        # Log data info
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(X.columns))
        mlflow.log_param("n_vehicles", df['vehicle_id'].nunique())
        
        # Train with hyperparameter tuning
        tuning_results = model.train_with_hyperparameter_tuning(X, y, df)
        
        # Log best parameters
        for param, value in tuning_results['best_params'].items():
            mlflow.log_param(param, value)
        
        # Train quantile models
        model.train_quantile_models(X, y, df)
        
        # Evaluate model
        metrics = model.evaluate_model(X, y, df)
        
        # Log metrics
        for metric, value in metrics.items():
            if value is not None:
                mlflow.log_metric(metric, value)
        
        # Feature importance
        importance_df = model.get_feature_importance()
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Save feature importance plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances - XGBoost Baseline')
        plt.tight_layout()
        plt.savefig('models/artifacts/feature_importance.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/feature_importance.png')
        
        # Save model
        model.save_model()
        
        # Log model
        mlflow.xgboost.log_model(model.model, "model")
        
        print(f"\nTraining complete!")
        print(f"Model saved to models/artifacts/")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()