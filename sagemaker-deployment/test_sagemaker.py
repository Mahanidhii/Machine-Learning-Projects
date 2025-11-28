"""
Test Client for AWS SageMaker Heart Disease Prediction Endpoint
"""
import boto3
import json

# Configuration
ENDPOINT_NAME = 'heart-disease-endpoint'
REGION = 'us-east-1'  # Change to your AWS region


def test_sagemaker_endpoint():
    """Test the SageMaker endpoint with sample data"""
    print("=" * 60)
    print("Testing AWS SageMaker Endpoint")
    print(f"Endpoint: {ENDPOINT_NAME}")
    print(f"Region: {REGION}")
    print("=" * 60)
    
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name=REGION)
    
    # Test cases
    test_cases = [
        {
            "name": "Patient 1 - High Risk",
            "data": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        },
        {
            "name": "Patient 2 - Low Risk",
            "data": {
                "age": 45,
                "sex": 0,
                "cp": 0,
                "trestbps": 120,
                "chol": 180,
                "fbs": 0,
                "restecg": 0,
                "thalach": 160,
                "exang": 0,
                "oldpeak": 0.5,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        },
        {
            "name": "Patient 3 - Medium Risk",
            "data": {
                "age": 55,
                "sex": 1,
                "cp": 2,
                "trestbps": 135,
                "chol": 210,
                "fbs": 0,
                "restecg": 1,
                "thalach": 145,
                "exang": 1,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 1,
                "thal": 2
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"Input: {json.dumps(test_case['data'], indent=2)}")
        
        try:
            # Invoke endpoint
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps(test_case['data'])
            )
            
            # Parse result
            result = json.loads(response['Body'].read().decode())
            
            prediction = result['predictions'][0]
            probabilities = result['probabilities'][0]
            
            print(f"\nResult:")
            print(f"  Prediction: {'❤️  DISEASE DETECTED' if prediction == 1 else '✅ NO DISEASE'}")
            print(f"  Confidence (No Disease): {probabilities[0]:.2%}")
            print(f"  Confidence (Disease): {probabilities[1]:.2%}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


def check_endpoint_status():
    """Check if the endpoint is active"""
    print("\nChecking endpoint status...")
    
    sagemaker = boto3.client('sagemaker', region_name=REGION)
    
    try:
        response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response['EndpointStatus']
        
        print(f"Endpoint Status: {status}")
        
        if status == 'InService':
            print("✅ Endpoint is ready for predictions!")
            return True
        elif status == 'Creating':
            print("⏳ Endpoint is still being created...")
            return False
        elif status == 'Failed':
            print("❌ Endpoint creation failed!")
            print(f"Failure Reason: {response.get('FailureReason', 'Unknown')}")
            return False
        else:
            print(f"⚠️  Endpoint status: {status}")
            return False
            
    except sagemaker.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"❌ Endpoint '{ENDPOINT_NAME}' not found!")
            print("Please deploy the endpoint first using deploy_sagemaker.py")
        else:
            print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("\n❤️  Heart Disease Prediction - SageMaker Endpoint Test\n")
    
    # Check if endpoint is ready
    if check_endpoint_status():
        # Run tests
        test_sagemaker_endpoint()
    else:
        print("\nCannot run tests - endpoint is not available.")
