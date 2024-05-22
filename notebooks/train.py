
# Import the required library 
import xgboost as xgb
import os
import pickle

model_file_name = "pipeline_model"

# Main Function
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir",type = str , default = os.environ.get("SM_MODEL_DIR"))

    args, _ = parser.parse_known_args()
    
    # Get the current directory
    current_dir = os.getcwd()

    # Get the parent directory (one level up)
    parent_dir = os.path.dirname(current_dir)
    
    # Print the parent directory
    print("Parent Directory:", parent_dir)
    
    preprocessed_data_dir = parent_dir+'/datasets/'
    model_dir = parent_dir+'/models/'
    
    # Load X_test from file
    with open(os.path.join(preprocessed_data_dir,'resnet_X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
        
    # Load y_test from file
    with open(os.path.join(preprocessed_data_dir,'resnety_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    
    # Creating a copy of labels
    y_train_1 = y_train.copy()
    
    # Replace all occurrences defective as 0 and good as 1
    for i in range(len(y_train)):
        if y_train[i]=='defective' : 
            y_train[i] = 0
        else:
            y_train[i] = 1
    
    # Conert it into string to int
    y_train=y_train.astype(int)
    
    # Model selection
    xgb_model = xgb.XGBClassifier()
    
    # Train the model
    xgb_model.fit(X_train, y_train)

    # Load X_test from file
    with open(os.path.join(preprocessed_data_dir,'resnetX_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
        
    # Load y_test from file
    with open(os.path.join(preprocessed_data_dir,'resnety_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    
    # Creating a copy of labels
    y_test_1 = y_test.copy()
    
    # Replace all occurrences defective as 0 and good as 1
    for i in range(len(y_test)):
        if y_test[i]=='defective' : 
            y_test[i] = 0
        else:
            y_test[i] = 1
    
    # Conert it into string to int
    y_test=y_test.astype(int)

    #train accuracy
    train_accuracy = xgb_model.score(X_train,y_train)
    print(f"Training accuracy: {train_accuracy: .4f}")
    
    #test accuracy
    test_accuracy = xgb_model.score(X_test,y_test)
    print(f"Testing accuracy: {test_accuracy: .4f}")

    model_save_path = os.path.join(args.model_dir,model_file_name)
    pickle.dump(xgb_model, model_save_path)
    print(f"Model save at path: {model_save_path}")

if __name__=="main":
    main()
