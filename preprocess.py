import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-ratio', 
                        type=str, 
                        help='Split of train-test data (0-1) e.g. "0.7"', default="0.7")
    parser.add_argument('--output-dir',
                        type=str, 
                        help='Local directory (or in container) where processed data is saved as parquet', 
                        default='/opt/ml/processing/output')
    parser.add_argument('--input-dir',
                        type=str, 
                        help='Local directory (or in container) from where raw data is read',
                        default='/opt/ml/processing/input')

    args, _ = parser.parse_known_args()
    
    # Parse arguments
    train_test_ratio = eval(args.train_test_ratio)
    output_dir = args.output_dir
    input_dir = args.input_dir
    
    # Read raw data from previous step
    df = pd.read_parquet(os.path.join(input_dir, "raw_data.parquet"))
    
    # Drop nans
    df.dropna(axis=0, inplace=True)

    # Split train and test data
    predictors = ['x1', 'x2']
    target = ['is_blue']
    
    X = df[predictors].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_test_ratio), random_state=42)
    
    # Preprocessing steps
    scaler = StandardScaler()
    preprocessing_pipeline = Pipeline([('scaler', scaler)])
    
    X_train = preprocessing_pipeline.fit_transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)
    
    # Save preprocessor to use for inference
    joblib.dump(preprocessing_pipeline, os.path.join(output_dir, 'preprocessing_pipeline.mdl'))

    # Save data to be used for training and model evaluation
    train_data_filepath = os.path.join(output_dir, 'train.parquet')
    test_data_filepath = os.path.join(output_dir, 'test.parquet')
    
    df_train = pd.DataFrame(np.c_[X_train, y_train], columns=predictors+target)
    df_test = pd.DataFrame(np.c_[X_test, y_test], columns=predictors+target)

    df_train.to_parquet(train_data_filepath)
    df_test.to_parquet(test_data_filepath)
    