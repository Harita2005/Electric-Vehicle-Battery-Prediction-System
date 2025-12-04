import json
import os
import sys
sys.path.append('/opt/python')

# Import our drift detection module
from monitoring.drift_detection import BatteryDriftDetector

def handler(event, context):
    """Lambda handler for drift detection"""
    try:
        # Get environment variables
        data_bucket = os.environ.get('DATA_BUCKET')
        model_bucket = os.environ.get('MODEL_BUCKET')
        
        if not data_bucket:
            raise ValueError("DATA_BUCKET environment variable not set")
        
        # Initialize drift detector
        detector = BatteryDriftDetector(s3_bucket=data_bucket)
        
        # Run drift detection
        report = detector.run_drift_detection()
        
        # Return results
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Drift detection completed successfully',
                'drift_detected': report.get('overall_drift_detected', False),
                'recommendation': report.get('recommendation', 'No recommendation'),
                'timestamp': report.get('timestamp')
            })
        }
        
    except Exception as e:
        print(f"Error in drift detection: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Drift detection failed'
            })
        }