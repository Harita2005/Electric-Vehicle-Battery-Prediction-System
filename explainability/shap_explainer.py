#!/usr/bin/env python3
"""
SHAP Explainability for Battery SoH Predictions
Provides global and local explanations with counterfactual analysis.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from models.train_baseline import BaselineModel

class BatteryExplainer:
    def __init__(self, model_path: str = "models/artifacts"):
        self.model_path = model_path
        self.baseline_model = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def load_model(self):
        """Load trained model and create SHAP explainer"""
        print("Loading model...")
        self.baseline_model = BaselineModel()
        self.baseline_model.load_model(self.model_path)
        
        # Load sample data for explainer background
        sample_data = pd.read_parquet('data/processed/features_sample.parquet')
        X_sample, _ = self.baseline_model.feature_engineer.prepare_model_data(sample_data)
        X_sample_scaled = pd.DataFrame(
            self.baseline_model.scaler.transform(X_sample),
            columns=X_sample.columns
        )
        
        # Create SHAP explainer
        print("Creating SHAP explainer...")
        self.explainer = shap.TreeExplainer(
            self.baseline_model.model,
            X_sample_scaled.sample(100)  # Background dataset
        )
        
        self.feature_names = X_sample_scaled.columns.tolist()
        print(f"Explainer ready with {len(self.feature_names)} features")
    
    def explain_predictions(self, X: pd.DataFrame, max_samples: int = 1000) -> np.ndarray:
        """Calculate SHAP values for predictions"""
        if self.explainer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Calculating SHAP values for {min(len(X), max_samples)} samples...")
        
        # Limit samples for computational efficiency
        X_explain = X.head(max_samples) if len(X) > max_samples else X
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_explain)
        
        return self.shap_values
    
    def plot_global_importance(self, save_path: str = None) -> None:
        """Plot global feature importance using SHAP values"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_predictions() first.")
        
        # Summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values, 
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.title('Global Feature Importance (SHAP Values)', fontsize=16, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance_bar(self, save_path: str = None) -> None:
        """Plot feature importance as bar chart"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_predictions() first.")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        # Plot top 20
        plt.figure(figsize=(12, 10))
        top_20 = importance_df.head(20)
        
        bars = plt.barh(range(len(top_20)), top_20['importance'])
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 20 Features by SHAP Importance')
        plt.gca().invert_yaxis()
        
        # Color bars by feature group
        feature_groups = self.baseline_model.feature_engineer.get_feature_importance_groups()
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_groups)))
        group_colors = {group: colors[i] for i, group in enumerate(feature_groups.keys())}\n        \n        for i, (_, row) in enumerate(top_20.iterrows()):\n            feature = row['feature']\n            for group, features in feature_groups.items():\n                if feature in features:\n                    bars[i].set_color(group_colors[group])\n                    break\n        \n        plt.tight_layout()\n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        return importance_df\n    \n    def explain_single_prediction(self, X: pd.DataFrame, sample_idx: int = 0, \n                                save_path: str = None) -> Dict:\n        \"\"\"Explain a single prediction with waterfall plot\"\"\"\n        if self.shap_values is None:\n            raise ValueError(\"SHAP values not calculated. Call explain_predictions() first.\")\n        \n        if sample_idx >= len(self.shap_values):\n            raise ValueError(f\"Sample index {sample_idx} out of range\")\n        \n        # Get prediction and SHAP values for this sample\n        sample_shap = self.shap_values[sample_idx]\n        sample_features = X.iloc[sample_idx]\n        prediction = self.baseline_model.model.predict(X.iloc[[sample_idx]])[0]\n        \n        # Create waterfall plot\n        plt.figure(figsize=(12, 8))\n        shap.waterfall_plot(\n            shap.Explanation(\n                values=sample_shap,\n                base_values=self.explainer.expected_value,\n                data=sample_features.values,\n                feature_names=self.feature_names\n            ),\n            show=False\n        )\n        plt.title(f'SHAP Explanation for Sample {sample_idx}\\nPredicted SoH: {prediction:.2f}%')\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        # Return top contributing features\n        feature_contributions = list(zip(self.feature_names, sample_shap))\n        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)\n        \n        return {\n            'prediction': prediction,\n            'base_value': self.explainer.expected_value,\n            'top_features': feature_contributions[:10],\n            'sample_values': sample_features.to_dict()\n        }\n    \n    def analyze_feature_interactions(self, feature1: str, feature2: str, \n                                   X: pd.DataFrame, save_path: str = None) -> None:\n        \"\"\"Analyze interaction between two features\"\"\"\n        if feature1 not in self.feature_names or feature2 not in self.feature_names:\n            raise ValueError(\"Features not found in model\")\n        \n        # Get feature indices\n        idx1 = self.feature_names.index(feature1)\n        idx2 = self.feature_names.index(feature2)\n        \n        # Create interaction plot\n        plt.figure(figsize=(10, 8))\n        shap.plots.scatter(\n            self.shap_values[:, idx1], \n            color=self.shap_values[:, idx2],\n            show=False\n        )\n        plt.xlabel(f'SHAP value for {feature1}')\n        plt.ylabel(f'Feature value for {feature1}')\n        plt.title(f'Feature Interaction: {feature1} vs {feature2}')\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        plt.show()\n    \n    def generate_counterfactuals(self, X: pd.DataFrame, sample_idx: int = 0, \n                               modifications: Dict[str, float] = None) -> Dict:\n        \"\"\"Generate counterfactual predictions by modifying features\"\"\"\n        if modifications is None:\n            # Default modifications for common scenarios\n            modifications = {\n                'fast_charge_sessions_30d': 0.5,  # Reduce fast charging by 50%\n                'pack_temp_max_30d': 0.9,         # Reduce max temperature by 10%\n                'thermal_stress_hours_30d': 0.7   # Reduce thermal stress by 30%\n            }\n        \n        # Get original sample\n        original_sample = X.iloc[[sample_idx]].copy()\n        original_pred = self.baseline_model.model.predict(original_sample)[0]\n        \n        counterfactuals = {\n            'original_prediction': original_pred,\n            'scenarios': {}\n        }\n        \n        # Test each modification\n        for feature, multiplier in modifications.items():\n            if feature in original_sample.columns:\n                # Create modified sample\n                modified_sample = original_sample.copy()\n                modified_sample[feature] = modified_sample[feature] * multiplier\n                \n                # Get new prediction\n                new_pred = self.baseline_model.model.predict(modified_sample)[0]\n                improvement = new_pred - original_pred\n                \n                counterfactuals['scenarios'][feature] = {\n                    'modification': f\"Multiply by {multiplier}\",\n                    'new_prediction': new_pred,\n                    'improvement': improvement,\n                    'original_value': original_sample[feature].iloc[0],\n                    'new_value': modified_sample[feature].iloc[0]\n                }\n        \n        return counterfactuals\n    \n    def plot_counterfactuals(self, counterfactuals: Dict, save_path: str = None) -> None:\n        \"\"\"Plot counterfactual analysis results\"\"\"\n        scenarios = counterfactuals['scenarios']\n        \n        if not scenarios:\n            print(\"No counterfactual scenarios to plot\")\n            return\n        \n        # Prepare data for plotting\n        features = list(scenarios.keys())\n        improvements = [scenarios[f]['improvement'] for f in features]\n        \n        # Create plot\n        plt.figure(figsize=(12, 6))\n        bars = plt.bar(range(len(features)), improvements)\n        plt.xticks(range(len(features)), [f.replace('_', '\\n') for f in features], rotation=45)\n        plt.ylabel('SoH Improvement (%)')\n        plt.title('Counterfactual Analysis: Potential SoH Improvements')\n        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)\n        \n        # Color bars by improvement\n        for bar, improvement in zip(bars, improvements):\n            if improvement > 0:\n                bar.set_color('green')\n            else:\n                bar.set_color('red')\n        \n        plt.tight_layout()\n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        # Print summary\n        print(f\"\\nCounterfactual Analysis Summary:\")\n        print(f\"Original Prediction: {counterfactuals['original_prediction']:.2f}%\")\n        for feature, scenario in scenarios.items():\n            print(f\"{feature}: {scenario['improvement']:+.2f}% improvement\")\n    \n    def create_explanation_dashboard(self, X: pd.DataFrame, sample_idx: int = 0, \n                                  output_dir: str = \"explainability/outputs\") -> None:\n        \"\"\"Create comprehensive explanation dashboard\"\"\"\n        Path(output_dir).mkdir(parents=True, exist_ok=True)\n        \n        print(\"Creating explanation dashboard...\")\n        \n        # Calculate SHAP values if not done\n        if self.shap_values is None:\n            self.explain_predictions(X)\n        \n        # 1. Global importance\n        self.plot_global_importance(f\"{output_dir}/global_importance.png\")\n        \n        # 2. Feature importance bar chart\n        importance_df = self.plot_feature_importance_bar(f\"{output_dir}/feature_importance_bar.png\")\n        \n        # 3. Single prediction explanation\n        explanation = self.explain_single_prediction(X, sample_idx, f\"{output_dir}/single_prediction.png\")\n        \n        # 4. Counterfactual analysis\n        counterfactuals = self.generate_counterfactuals(X, sample_idx)\n        self.plot_counterfactuals(counterfactuals, f\"{output_dir}/counterfactuals.png\")\n        \n        # 5. Save explanation data\n        explanation_data = {\n            'sample_explanation': explanation,\n            'counterfactuals': counterfactuals,\n            'feature_importance': importance_df.to_dict('records')\n        }\n        \n        with open(f\"{output_dir}/explanation_data.json\", 'w') as f:\n            json.dump(explanation_data, f, indent=2, default=str)\n        \n        print(f\"Dashboard created in {output_dir}\")\n        return explanation_data\n\ndef main():\n    \"\"\"Generate explanations for sample predictions\"\"\"\n    # Initialize explainer\n    explainer = BatteryExplainer()\n    explainer.load_model()\n    \n    # Load test data\n    test_data = pd.read_parquet('data/processed/features_sample.parquet')\n    X_test, y_test = explainer.baseline_model.feature_engineer.prepare_model_data(test_data)\n    X_test_scaled = pd.DataFrame(\n        explainer.baseline_model.scaler.transform(X_test),\n        columns=X_test.columns\n    )\n    \n    # Create explanation dashboard\n    explanation_data = explainer.create_explanation_dashboard(X_test_scaled, sample_idx=0)\n    \n    print(\"\\nExplanation generation complete!\")\n    print(\"Files saved to explainability/outputs/\")\n    \n    # Print sample explanation\n    print(\"\\nTop 5 Contributing Features:\")\n    for feature, contribution in explanation_data['sample_explanation']['top_features'][:5]:\n        print(f\"  {feature}: {contribution:+.4f}\")\n\nif __name__ == \"__main__\":\n    main()