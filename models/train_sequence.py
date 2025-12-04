#!/usr/bin/env python3
"""
LSTM Sequence Model for Battery SoH Prediction
Handles variable-length sequences with uncertainty estimation.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.pytorch
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from data_pipeline.feature_engineering import BatteryFeatureEngineer

class BatterySequenceDataset(Dataset):
    def __init__(self, sequences, targets, sequence_length=90):
        self.sequences = sequences
        self.targets = targets
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Pad or truncate sequence
        if len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        else:
            padding = np.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([padding, sequence])
        
        return torch.FloatTensor(sequence), torch.FloatTensor([target])

class BatteryLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, 
                 learning_rate=0.001, uncertainty=True):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.uncertainty = uncertainty
        
        if uncertainty:
            # Output mean and log variance for uncertainty
            self.output_layer = nn.Linear(hidden_size, 2)
        else:
            self.output_layer = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Final prediction
        output = self.output_layer(last_output)
        
        if self.uncertainty:
            mean = output[:, 0]
            log_var = output[:, 1]
            return mean, log_var
        else:
            return output.squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        
        if self.uncertainty:
            mean, log_var = self(x)
            
            # Negative log likelihood loss with uncertainty
            var = torch.exp(log_var)
            loss = 0.5 * (torch.log(var) + (y - mean)**2 / var)
            loss = loss.mean()
            
            self.log('train_loss', loss)
            self.log('train_mae', torch.mean(torch.abs(y - mean)))
        else:
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred, y)
            self.log('train_loss', loss)
            self.log('train_mae', torch.mean(torch.abs(y - y_pred)))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        
        if self.uncertainty:
            mean, log_var = self(x)
            
            var = torch.exp(log_var)
            loss = 0.5 * (torch.log(var) + (y - mean)**2 / var)
            loss = loss.mean()
            
            self.log('val_loss', loss)
            self.log('val_mae', torch.mean(torch.abs(y - mean)))
        else:
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred, y)
            self.log('val_loss', loss)
            self.log('val_mae', torch.mean(torch.abs(y - y_pred)))
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class SequenceModel:
    def __init__(self, model_name: str = "lstm_sequence", sequence_length: int = 90):
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = BatteryFeatureEngineer()
        self.feature_columns = None
        
    def prepare_sequences(self, data_path: str) -> tuple:
        """Prepare sequence data for LSTM training"""
        print("Loading and preparing sequence data...")
        
        # Load data
        data_files = list(Path(data_path).glob("vehicle_*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No vehicle data files found in {data_path}")
        
        sequences = []
        targets = []
        vehicle_ids = []
        
        for file in data_files[:20]:  # Limit for demo
            df = pd.read_parquet(file)
            
            # Engineer features
            df_features = self.feature_engineer.engineer_all_features(df)
            
            # Prepare model data
            X, y = self.feature_engineer.prepare_model_data(df_features, target_col='soh_percent')
            
            if len(X) < self.sequence_length:
                continue  # Skip vehicles with insufficient data
            
            # Store feature columns from first vehicle
            if self.feature_columns is None:
                self.feature_columns = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X) if len(sequences) == 0 else self.scaler.transform(X)
            
            # Create sequences (sliding window)
            for i in range(self.sequence_length, len(X_scaled)):
                sequence = X_scaled[i-self.sequence_length:i]
                target = y.iloc[i]
                
                sequences.append(sequence)
                targets.append(target)
                vehicle_ids.append(df['vehicle_id'].iloc[0])
        
        print(f"Created {len(sequences)} sequences from {len(data_files[:20])} vehicles")
        
        return np.array(sequences), np.array(targets), vehicle_ids
    
    def create_data_loaders(self, sequences, targets, batch_size=32, train_split=0.8):
        """Create train/validation data loaders"""
        # Split data
        n_train = int(len(sequences) * train_split)
        
        train_sequences = sequences[:n_train]
        train_targets = targets[:n_train]
        val_sequences = sequences[n_train:]
        val_targets = targets[n_train:]
        
        # Create datasets
        train_dataset = BatterySequenceDataset(train_sequences, train_targets, self.sequence_length)
        val_dataset = BatterySequenceDataset(val_sequences, val_targets, self.sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def train_model(self, sequences, targets, max_epochs=50, batch_size=32):
        """Train LSTM model"""
        print("Training LSTM model...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(sequences, targets, batch_size)
        
        # Initialize model
        input_size = sequences.shape[2]  # Number of features
        self.model = BatteryLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            uncertainty=True
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min'
        )
        
        checkpoint = ModelCheckpoint(
            dirpath='models/artifacts',
            filename=f'{self.model_name}-{{epoch:02d}}-{{val_loss:.4f}}',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stopping, checkpoint],
            accelerator='auto',
            devices=1,
            log_every_n_steps=10
        )
        
        # Train
        trainer.fit(self.model, train_loader, val_loader)
        
        # Load best model
        self.model = BatteryLSTM.load_from_checkpoint(
            checkpoint.best_model_path,
            input_size=input_size
        )
        
        return trainer
    
    def evaluate_model(self, sequences, targets):
        """Evaluate model performance"""
        print("Evaluating LSTM model...")
        
        # Split data
        n_train = int(len(sequences) * 0.8)
        test_sequences = sequences[n_train:]
        test_targets = targets[n_train:]
        
        # Create test dataset
        test_dataset = BatterySequenceDataset(test_sequences, test_targets, self.sequence_length)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Predictions
        self.model.eval()
        predictions = []
        uncertainties = []
        true_values = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                
                if self.model.uncertainty:
                    mean, log_var = self.model(x)
                    var = torch.exp(log_var)
                    
                    predictions.extend(mean.cpu().numpy())
                    uncertainties.extend(torch.sqrt(var).cpu().numpy())
                else:
                    y_pred = self.model(x)
                    predictions.extend(y_pred.cpu().numpy())
                
                true_values.extend(y.squeeze().cpu().numpy())
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # Metrics
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'n_test': len(test_sequences)
        }
        
        if uncertainties:
            uncertainties = np.array(uncertainties)
            # Coverage (simplified - assumes normal distribution)
            lower_bound = predictions - 1.96 * uncertainties
            upper_bound = predictions + 1.96 * uncertainties
            coverage = np.mean((true_values >= lower_bound) & (true_values <= upper_bound))
            
            metrics['coverage'] = coverage
            metrics['mean_uncertainty'] = np.mean(uncertainties)
        
        print(f"LSTM Model Performance:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        if 'coverage' in metrics:
            print(f"  95% Coverage: {metrics['coverage']:.4f}")
            print(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
        
        return metrics
    
    def predict_with_uncertainty(self, sequences):
        """Make predictions with uncertainty estimates"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        predictions = []
        uncertainties = []
        
        # Convert to tensor if needed
        if isinstance(sequences, np.ndarray):
            sequences = torch.FloatTensor(sequences)
        
        with torch.no_grad():
            if self.model.uncertainty:
                mean, log_var = self.model(sequences)
                var = torch.exp(log_var)
                
                predictions = mean.cpu().numpy()
                uncertainties = torch.sqrt(var).cpu().numpy()
                
                return {
                    'prediction': predictions,
                    'uncertainty': uncertainties,
                    'lower_bound': predictions - 1.96 * uncertainties,
                    'upper_bound': predictions + 1.96 * uncertainties
                }
            else:
                predictions = self.model(sequences).cpu().numpy()
                return {'prediction': predictions}
    
    def save_model(self, output_dir: str = "models/artifacts"):
        """Save trained model and preprocessing components"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        model_path = f"{output_dir}/{self.model_name}.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save model hyperparameters
        hparams_path = f"{output_dir}/{self.model_name}_hparams.json"
        with open(hparams_path, 'w') as f:
            json.dump(dict(self.model.hparams), f)
        
        # Save scaler
        scaler_path = f"{output_dir}/{self.model_name}_scaler.joblib"
        import joblib
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature engineer
        fe_path = f"{output_dir}/{self.model_name}_feature_engineer.joblib"
        joblib.dump(self.feature_engineer, fe_path)
        
        # Save feature columns
        with open(f"{output_dir}/{self.model_name}_features.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        print(f"LSTM model saved to {output_dir}")

def main():
    """Train sequence model"""
    # Initialize MLflow
    mlflow.set_experiment("ev_battery_sequence")
    
    with mlflow.start_run():
        # Initialize model
        model = SequenceModel()
        
        # Prepare sequences
        sequences, targets, vehicle_ids = model.prepare_sequences("data/raw")
        
        # Log data info
        mlflow.log_param("n_sequences", len(sequences))
        mlflow.log_param("sequence_length", model.sequence_length)
        mlflow.log_param("n_features", sequences.shape[2])
        mlflow.log_param("n_vehicles", len(set(vehicle_ids)))
        
        # Train model
        trainer = model.train_model(sequences, targets, max_epochs=30)
        
        # Evaluate model
        metrics = model.evaluate_model(sequences, targets)
        
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Save model
        model.save_model()
        
        print(f"\nLSTM training complete!")
        print(f"Model saved to models/artifacts/")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()