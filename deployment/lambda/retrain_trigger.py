import json
import os
import boto3
from datetime import datetime

def handler(event, context):
    """Lambda handler for triggering model retraining"""
    try:
        # Get environment variables
        sagemaker_role = os.environ.get('SAGEMAKER_ROLE')
        model_bucket = os.environ.get('MODEL_BUCKET')
        
        if not sagemaker_role or not model_bucket:
            raise ValueError("Required environment variables not set")
        
        # Initialize SageMaker client
        sagemaker = boto3.client('sagemaker')
        
        # Training job configuration
        training_job_name = f"ev-battery-retrain-{int(datetime.now().timestamp())}"
        
        training_params = {
            'TrainingJobName': training_job_name,
            'RoleArn': sagemaker_role,
            'AlgorithmSpecification': {
                'TrainingImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1',
                'TrainingInputMode': 'File'
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{model_bucket}/training-data/',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv',
                    'CompressionType': 'None'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{model_bucket}/model-output/'
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600
            },
            'HyperParameters': {
                'objective': 'reg:squarederror',
                'num_round': '100',
                'max_depth': '6',
                'eta': '0.1',
                'subsample': '0.8',
                'colsample_bytree': '0.8'
            }
        }
        
        # Start training job
        response = sagemaker.create_training_job(**training_params)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Model retraining started successfully',
                'training_job_name': training_job_name,
                'training_job_arn': response['TrainingJobArn']
            })
        }
        
    except Exception as e:
        print(f"Error starting retraining: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to start model retraining'
            })
        }