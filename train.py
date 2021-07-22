import numpy as np
import pandas as pd
import boto3
import sagemaker
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import argparse
from utils import get_data_from_dynamodb
import joblib
import os
from datetime import datetime, timedelta


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="Directory where to store model artefacts")
    parser.add_argument("--training", type=str)

    args, _ = parser.parse_known_args()
    
    training_dir = args.training
    model_dir = args.model_dir
    
    df = pd.read_parquet(os.path.join(model_dir, "train.parquet"))
    
    predictors = ['x1', 'x2']
    target = ['is_blue']
    
    X = df[predictors]
    y = df[target].values.ravel()
    
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    
    model = KNeighborsClassifier(n_neighbors=5)
    
    training_pipe = Pipeline([('imputer', imputer), ('scaler', scaler), ('model',model)])
    
    training_pipe.fit(X,y)
    
    joblib.dump(training_pipe, os.path.join(model_dir, "model.mdl"))
    

def model_fn(model_dir):
    mdl = joblib.load(os.path.join(model_dir, "model.mdl"))

