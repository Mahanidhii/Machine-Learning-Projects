"""
AWS SageMaker Deployment Script for Heart Disease Prediction Model
This script trains the model and deploys it to AWS SageMaker
"""
import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel
from sagemaker import get_execution_role
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# Train and save the model
def train_and_save_model():
    """Train the heart disease model and save it"""
    print("Training Heart Disease Prediction model...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # Load dataset
    df = pd.read_csv('../Heart Disease Predction/Dataset/heart_cleveland_upload.csv')
    
    # Separate features and target
    X = df.drop('condition', axis=1)
    y = df['condition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train AdaBoost model
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/heart_disease_model.pkl')
    
    # Save feature names for inference
    feature_info = {
        'feature_names': list(X.columns),
        'model_type': 'AdaBoostClassifier',
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'created_at': datetime.now().isoformat()
    }
    
    with open('model/model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Model saved successfully!")
    return model


def deploy_to_sagemaker(
    model_name='heart-disease-model',
    endpoint_name='heart-disease-endpoint',
    instance_type='ml.t2.medium',
    role_arn=None
):
    """
    Deploy the trained model to AWS SageMaker
    
    Args:
        model_name: Name for the SageMaker model
        endpoint_name: Name for the SageMaker endpoint
        instance_type: EC2 instance type for hosting
        role_arn: IAM Role ARN with SageMaker permissions
    """
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Get execution role
    if role_arn is None:
        try:
            role = get_execution_role()
        except:
            print("❌ Could not get execution role.")
            print("Please provide IAM role ARN with SageMaker permissions.")
            print("Example: arn:aws:iam::123456789012:role/SageMakerRole")
            return None
    else:
        role = role_arn
    
    print(f"Using IAM Role: {role}")
    
    # Upload model to S3
    bucket = sagemaker_session.default_bucket()
    prefix = 'heart-disease-model'
    
    print(f"Uploading model to S3 bucket: {bucket}")
    
    # Create model.tar.gz
    import tarfile
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('model/', arcname='.')
        tar.add('inference.py')
        tar.add('requirements.txt')
    
    # Upload to S3
    model_data = sagemaker_session.upload_data(
        path='model.tar.gz',
        bucket=bucket,
        key_prefix=prefix
    )
    
    print(f"Model uploaded to: {model_data}")
    
    # Create SageMaker model
    print("Creating SageMaker model...")
    
    sklearn_model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point='inference.py',
        framework_version='1.2-1',
        py_version='py3',
        name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Deploy the model
    print(f"Deploying model to endpoint: {endpoint_name}")
    print(f"Instance type: {instance_type}")
    print("⚠️  This may take 5-10 minutes...")
    
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    print("✅ Model deployed successfully!")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Region: {sagemaker_session.boto_region_name}")
    
    return predictor


def test_endpoint(endpoint_name, region='us-east-1'):
    """
    Test the deployed SageMaker endpoint
    
    Args:
        endpoint_name: Name of the SageMaker endpoint
        region: AWS region
    """
    print("\n" + "="*60)
    print("Testing SageMaker Endpoint")
    print("="*60)
    
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Sample test data
    test_data = {
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
    
    print(f"\nTest Input: {test_data}")
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(test_data)
    )
    
    result = json.loads(response['Body'].read().decode())
    print(f"\nPrediction Result: {result}")
    print("="*60)
    
    return result


def delete_endpoint(endpoint_name):
    """Delete SageMaker endpoint to stop charges"""
    sagemaker_client = boto3.client('sagemaker')
    
    print(f"Deleting endpoint: {endpoint_name}")
    
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print("✅ Endpoint deleted successfully!")
    except Exception as e:
        print(f"❌ Error deleting endpoint: {e}")


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("AWS SageMaker Deployment - Heart Disease Prediction")
    print("="*60)
    
    # Step 1: Train model
    print("\n📊 Step 1: Training model...")
    model = train_and_save_model()
    
    # Step 2: Deploy to SageMaker
    print("\n☁️  Step 2: Deploying to AWS SageMaker...")
    print("\nPlease ensure you have:")
    print("1. AWS credentials configured (aws configure)")
    print("2. IAM role with SageMaker permissions")
    print("3. Sufficient permissions to create S3 buckets and SageMaker endpoints")
    
    proceed = input("\nProceed with deployment? (yes/no): ")
    
    if proceed.lower() in ['yes', 'y']:
        # Optional: Provide custom IAM role ARN
        role_arn = input("Enter IAM Role ARN (or press Enter to auto-detect): ").strip()
        role_arn = role_arn if role_arn else None
        
        endpoint_name = input("Enter endpoint name (or press Enter for 'heart-disease-endpoint'): ").strip()
        endpoint_name = endpoint_name if endpoint_name else 'heart-disease-endpoint'
        
        try:
            predictor = deploy_to_sagemaker(
                endpoint_name=endpoint_name,
                role_arn=role_arn
            )
            
            if predictor:
                # Test the endpoint
                test = input("\nTest the endpoint? (yes/no): ")
                if test.lower() in ['yes', 'y']:
                    test_endpoint(endpoint_name)
                
                print("\n" + "="*60)
                print("Deployment Summary")
                print("="*60)
                print(f"Endpoint: {endpoint_name}")
                print(f"Status: ACTIVE")
                print(f"\nTo delete this endpoint and stop charges:")
                print(f"  python deploy_sagemaker.py --delete {endpoint_name}")
                print("="*60)
        
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            print("\nCommon issues:")
            print("1. AWS credentials not configured")
            print("2. Insufficient IAM permissions")
            print("3. Region not supported")
    else:
        print("Deployment cancelled.")
