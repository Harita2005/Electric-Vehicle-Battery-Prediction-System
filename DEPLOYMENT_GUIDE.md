# EV Battery Prediction System - Deployment Guide

## Prerequisites

### Local Development
- Python 3.9+
- Node.js 16+
- Docker (optional)
- Git

### AWS Deployment
- AWS CLI configured with appropriate permissions
- AWS CDK CLI installed (`npm install -g aws-cdk`)
- Docker for container builds

## Step-by-Step Deployment

### Phase 1: Local Setup and Testing

#### 1. Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd Electric_Vechical

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Generate Sample Data
```bash
# Generate data for 100 vehicles over 3 years
python simulator/generate_data.py --vehicles 100 --years 3 --output-dir data/raw

# This creates ~50GB of realistic battery telemetry data
# Adjust --vehicles parameter based on your needs
```

#### 3. Train Models Locally
```bash
# Train baseline XGBoost model
python models/train_baseline.py

# Train LSTM sequence model (optional, requires more compute)
python models/train_sequence.py

# Models will be saved to models/artifacts/
```

#### 4. Test Dashboard Locally
```bash
# Install dashboard dependencies
cd dashboard
npm install

# Start development server
npm start

# In another terminal, start the API
cd ..
python models/api.py

# Dashboard will be available at http://localhost:3000
```

#### 5. Run Tests
```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Phase 2: AWS Infrastructure Deployment

#### 1. Configure AWS Credentials
```bash
# Configure AWS CLI
aws configure

# Verify access
aws sts get-caller-identity
```

#### 2. Bootstrap CDK (First Time Only)
```bash
cd deployment

# Install CDK dependencies
pip install -r requirements.txt

# Bootstrap CDK in your account/region
cdk bootstrap
```

#### 3. Deploy Infrastructure
```bash
# Deploy the main stack
cdk deploy EVBatteryStack

# This creates:
# - S3 buckets for data and models
# - SageMaker notebook instance
# - ECS cluster and service
# - Application Load Balancer
# - CloudWatch dashboards
# - Lambda functions for monitoring
```

#### 4. Upload Data and Models
```bash
# Upload sample data to S3
aws s3 sync ../data/raw/ s3://ev-battery-raw-data-<account>-<region>/

# Upload trained models
aws s3 sync ../models/artifacts/ s3://ev-battery-models-<account>-<region>/
```

#### 5. Deploy Application
```bash
# Build and push Docker image
docker build -t ev-battery-api ..
docker tag ev-battery-api:latest <account>.dkr.ecr.<region>.amazonaws.com/ev-battery-api:latest

# Get ECR login
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

# Push image
docker push <account>.dkr.ecr.<region>.amazonaws.com/ev-battery-api:latest

# Update ECS service
aws ecs update-service --cluster ev-battery-cluster --service ev-battery-service --force-new-deployment
```

### Phase 3: Dashboard Deployment

#### 1. Build React App
```bash
cd dashboard

# Build for production
npm run build

# Upload to S3
aws s3 sync build/ s3://ev-battery-dashboard-<account>-<region>/

# Enable website hosting
aws s3 website s3://ev-battery-dashboard-<account>-<region>/ --index-document index.html
```

#### 2. Configure CloudFront (Optional)
```bash
# Create CloudFront distribution for better performance
# This can be added to the CDK stack for automation
```

### Phase 4: Monitoring and Alerts

#### 1. Set Up CloudWatch Dashboards
```bash
# Dashboards are automatically created by CDK
# Access via AWS Console > CloudWatch > Dashboards > EV-Battery-Monitoring
```

#### 2. Configure Drift Detection
```bash
# Lambda function is deployed automatically
# Test drift detection
aws lambda invoke --function-name DriftDetectionFunction response.json
```

#### 3. Set Up Alerts
```bash
# SNS topics and alarms are created by CDK
# Subscribe to alerts via AWS Console > SNS
```

## Configuration

### Environment Variables

#### API Service
```bash
export MODEL_BUCKET=ev-battery-models-<account>-<region>
export DATA_BUCKET=ev-battery-processed-data-<account>-<region>
export AWS_DEFAULT_REGION=<region>
```

#### Dashboard
```bash
export REACT_APP_API_URL=https://<alb-dns-name>
```

### Model Configuration

#### Baseline Model Hyperparameters
```python
# In models/train_baseline.py
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    # Adjust based on your data size and compute budget
}
```

#### LSTM Model Configuration
```python
# In models/train_sequence.py
model_config = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'sequence_length': 90,  # Days of history
    # Adjust based on your temporal requirements
}
```

## Scaling Considerations

### Data Volume
- **Small Scale** (< 100 vehicles): Use single EC2 instance
- **Medium Scale** (100-1000 vehicles): Use ECS with auto-scaling
- **Large Scale** (1000+ vehicles): Consider SageMaker batch transform

### Model Training
- **Development**: Use ml.m5.large instances
- **Production**: Use ml.m5.xlarge or larger with spot instances
- **Large Datasets**: Consider distributed training with SageMaker

### API Scaling
```yaml
# ECS Service Configuration
desired_count: 2
min_capacity: 1
max_capacity: 10
target_cpu_utilization: 70
```

## Security Best Practices

### IAM Roles
- Use least privilege principle
- Separate roles for different services
- Enable CloudTrail for audit logging

### Data Encryption
- S3 buckets encrypted with KMS
- ECS tasks use encrypted EBS volumes
- API uses HTTPS only

### Network Security
- VPC with private subnets for compute
- Security groups with minimal required ports
- WAF for public-facing endpoints (optional)

## Monitoring and Maintenance

### Key Metrics to Monitor
- **Model Performance**: MAE, RMSE, prediction latency
- **Data Quality**: Missing values, drift detection scores
- **System Health**: API response time, error rates
- **Cost**: AWS resource utilization

### Automated Maintenance
- **Daily**: Drift detection runs
- **Weekly**: Model performance evaluation
- **Monthly**: Cost optimization review
- **Quarterly**: Model retraining (if drift detected)

## Troubleshooting

### Common Issues

#### 1. Model Training Fails
```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker

# Common causes:
# - Insufficient data
# - Memory/compute limits
# - IAM permissions
```

#### 2. API Returns 500 Errors
```bash
# Check ECS logs
aws logs describe-log-streams --log-group-name /ecs/ev-battery-api

# Common causes:
# - Model files not found in S3
# - Environment variables not set
# - Insufficient memory
```

#### 3. Dashboard Not Loading
```bash
# Check S3 website configuration
aws s3api get-bucket-website --bucket ev-battery-dashboard-<account>-<region>

# Check API connectivity
curl https://<alb-dns-name>/health
```

### Performance Optimization

#### 1. Model Inference
- Use model caching for repeated predictions
- Batch predictions when possible
- Consider model quantization for edge deployment

#### 2. Data Processing
- Use Parquet format for faster I/O
- Implement data partitioning by date/vehicle
- Cache frequently accessed features

#### 3. API Performance
- Implement response caching
- Use connection pooling
- Enable gzip compression

## Cost Optimization

### Development Environment
- Use spot instances for training
- Stop SageMaker notebooks when not in use
- Use S3 Intelligent Tiering

### Production Environment
- Right-size ECS tasks based on actual usage
- Use Reserved Instances for predictable workloads
- Implement auto-scaling policies

### Estimated Monthly Costs (1000 vehicles)
- **Compute**: $50-100 (ECS + SageMaker)
- **Storage**: $15-25 (S3 + EBS)
- **Networking**: $10-20 (ALB + data transfer)
- **Monitoring**: $5-10 (CloudWatch)
- **Total**: $80-155/month

## Next Steps

1. **Customize for Your Use Case**
   - Modify simulator parameters for your vehicle types
   - Adjust feature engineering for your data sources
   - Customize dashboard for your business metrics

2. **Integrate with Existing Systems**
   - Connect to your vehicle telemetry systems
   - Integrate with maintenance management systems
   - Set up automated reporting

3. **Advanced Features**
   - Implement A/B testing for model improvements
   - Add real-time streaming with Kinesis
   - Develop mobile app for field technicians

4. **Scale and Optimize**
   - Monitor performance and costs
   - Implement advanced MLOps practices
   - Consider multi-region deployment for global fleets