import pandas as pd
import boto3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import argparse
import joblib
import os
from logger import logger
from sklearn.metrics import f1_score
import json


def model_fn(model_dir):
    """Required model loading for Sagemaker framework"""
    model = joblib.load(os.path.join(model_dir, "model.mdl"))
    return model

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory where to store model artefacts",
        default=os.environ["SM_MODEL_DIR"],
    )
    parser.add_argument(
        "--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    
    args, _ = parser.parse_known_args()

    training_dir = args.training
    model_dir = args.model_dir
    
    # Needed for hyperparameter tuning (can be used for both custom and sagemaker algorithms)
    with open('/opt/ml/input/config/hyperparameters.json', 'r') as tc:
        hyperparams = json.load(tc) # All hyperparameters parsed as string
        
    # Allocate hyperparameters
    C = float(hyperparams['C'])  # float

    logger.info("Reading training data")
    df = pd.read_parquet(os.path.join(training_dir, "train.parquet"))

    predictors = ["x1", "x2"]
    target = ["is_blue"]

    X = df[predictors]
    y = df[target].values.ravel()

    model = SVC(C=C, kernel='rbf', class_weight='balanced')
    
    logger.info("Fitting model")
    model.fit(X, y)

    # Emit the required custom metrics (for hyperparameter tuning).
    f1 = f1_score(y, model.predict(X))
    print(f"f1_score = {f1};") # Mind the spaces to match Regex definition in Tuner

    joblib.dump(model, os.path.join(model_dir, "model.mdl"))

    logger.info("Uploading model to s3")
    bucket = boto3.resource("s3", region_name="eu-west-1").Bucket("hastie")
    bucket.upload_file(os.path.join(model_dir, "model.mdl"), Key="model/model.mdl")

