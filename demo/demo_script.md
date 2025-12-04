# EV Battery Prediction System Demo Script

## Demo Overview (2-3 minutes)
This demo showcases a production-grade EV battery health prediction system with end-to-end ML pipeline, explainable AI, and real-time monitoring.

## Demo Flow

### 1. System Architecture (30 seconds)
**Show:** Architecture diagram
**Say:** "This system predicts battery State of Health and Remaining Useful Life for electric vehicle fleets using AWS services. We have data simulation, feature engineering, multiple ML models, explainable AI, and a React dashboard."

**Key Points:**
- End-to-end pipeline from data to deployment
- Production-ready with monitoring and retraining
- Explainable predictions for business decisions

### 2. Data Simulator (30 seconds)
**Show:** Run simulator command
```bash
python simulator/generate_data.py --vehicles 10 --years 2
```

**Say:** "Our simulator generates realistic battery telemetry with degradation patterns, thermal stress, charging behaviors, and edge cases like sensor failures."

**Key Points:**
- Realistic physics-based degradation
- 1000+ vehicles, 3+ years of data
- Configurable driving/charging profiles

### 3. Model Training & Evaluation (45 seconds)
**Show:** Training results and evaluation notebook
```bash
python models/train_baseline.py
jupyter notebook evaluation/model_evaluation.ipynb
```

**Say:** "We train both XGBoost baseline and LSTM sequence models with uncertainty estimation. The evaluation shows strong performance with MAE under 2% and well-calibrated prediction intervals."

**Key Points:**
- Multiple model types (GBDT + sequence)
- Uncertainty quantification
- Comprehensive evaluation with ablation studies

### 4. Explainable AI (30 seconds)
**Show:** SHAP explanations
```bash
python explainability/shap_explainer.py
```

**Say:** "Every prediction comes with explanations showing which factors contribute most to battery degradation - temperature, fast charging, and cycle count are typically the top drivers."

**Key Points:**
- SHAP explanations for every prediction
- Feature importance analysis
- Counterfactual 'what-if' scenarios

### 5. Dashboard Demo (45 seconds)
**Show:** React dashboard
```bash
cd dashboard && npm start
```

**Navigate through:**
1. **Fleet Overview:** "Here's our fleet dashboard showing 50 vehicles with health scores and risk levels"
2. **Vehicle Detail:** "Drilling into a specific vehicle shows timeline, current status, and prediction explanations"
3. **What-If Analysis:** "The what-if tool shows how reducing fast charging by 50% could extend battery life by 25 days"

**Key Points:**
- Real-time fleet monitoring
- Per-vehicle detailed analysis
- Interactive what-if scenarios

### 6. Production Deployment (20 seconds)
**Show:** AWS infrastructure
```bash
cd deployment && cdk deploy
```

**Say:** "The system deploys to AWS with SageMaker for training, ECS for API serving, S3 for data storage, and CloudWatch for monitoring. It includes automated drift detection and retraining triggers."

**Key Points:**
- Infrastructure as Code with CDK
- Auto-scaling and monitoring
- MLOps with drift detection

## Demo Closing (10 seconds)
**Say:** "This system provides actionable insights for fleet operators to optimize battery life, reduce maintenance costs, and improve customer experience through predictive maintenance."

## Technical Highlights to Mention
- **Scalability:** Handles 1000+ vehicles with real-time predictions
- **Accuracy:** <2% MAE on battery health predictions
- **Explainability:** Every prediction explained with SHAP
- **Production-Ready:** Full MLOps pipeline with monitoring
- **Cost-Effective:** Spot instances and auto-scaling reduce costs

## Demo Tips
1. **Prepare screenshots** for any steps that might fail
2. **Have backup data** ready if simulator takes too long
3. **Practice transitions** between components
4. **Emphasize business value** throughout
5. **Keep technical details brief** but available for Q&A

## Q&A Preparation
**Common Questions:**
- **"How accurate are the predictions?"** → Show evaluation metrics (MAE, R², coverage)
- **"How do you handle data drift?"** → Explain drift detection and retraining
- **"What's the business impact?"** → Mention cost savings and customer experience
- **"How does it scale?"** → Discuss AWS auto-scaling and batch processing
- **"Can it work with real data?"** → Explain data requirements and integration points

## Recording Setup
- **Screen Resolution:** 1920x1080
- **Browser Zoom:** 100%
- **Terminal Font:** Large enough to read
- **Audio:** Clear narration with minimal background noise
- **Length:** Target 2-3 minutes maximum