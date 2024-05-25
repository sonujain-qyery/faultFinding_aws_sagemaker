
# Import the required libraries
import xgboost as xgb
import os
import pickle
import boto3
import argparse
import sagemaker
from sagemaker.analytics import TrainingJobAnalytics

model_file_name = "pipeline_model"

# Main Function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args, _ = parser.parse_known_args()

    # Specify the AWS region
    region_name = 'us-east-1'  # Change this to your desired region
    # Set up the default session with the specified region
    boto3.setup_default_session(region_name=region_name)
    
    # Create a SageMaker session with the specified region
    session = boto3.Session(region_name=region_name)
    sagemaker_session = sagemaker.Session(boto_session=session)
    bucket = sagemaker_session.default_bucket()
    
    # Initialize the S3 client
    s3 = boto3.client('s3')
    
    # Specify the bucket name and key (path) of the pickle file
    bucket_name = bucket
    X_train_file_key = "datasets/resnet_X_train.pkl"
    y_train_file_key = "datasets/resnety_train.pkl"
    X_test_file_key = "datasets/resnetX_test.pkl"
    y_test_file_key = "datasets/resnety_test.pkl"

    def loadData(file_key):
        # Read the pickle file from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        pickle_bytes = response['Body'].read()
        # Load the pickle file from bytes
        data = pickle.loads(pickle_bytes)
    
        return data

    X_train = loadData(X_train_file_key)
    y_train =  loadData(y_train_file_key)
    X_test =  loadData(X_test_file_key)
    y_test =  loadData(y_test_file_key)
    
    # Replace all occurrences defective as 0 and good as 1
    for i in range(len(y_train)):
        if y_train[i] == 'defective': 
            y_train[i] = 0
        else:
            y_train[i] = 1
    # Convert it into string to int
    y_train = y_train.astype(int)
    
    # Model selection
    xgb_model = xgb.XGBClassifier()
    
    # Train the model
    xgb_model.fit(X_train, y_train)

    # Replace all occurrences defective as 0 and good as 1
    for i in range(len(y_test)):
        if y_test[i] == 'defective':
            y_test[i] = 0
        else:
            y_test[i] = 1
    # Convert it into string to int
    y_test = y_test.astype(int)
    
    # Calculate test accuracy
    test_accuracy = xgb_model.score(X_test, y_test)
    
    # Save Model
    model_save_path = os.path.join(args.model_dir, model_file_name)
    with open(model_save_path,'wb') as f:
        pickle.dump(xgb_model, f)
    
    print(f"Model saved at path: {model_save_path}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    

# Check if the script is being executed as the main module
if __name__ == "__main__":
    # Call the main function
    main()
