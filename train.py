import pandas as pd
import boto3
from sklearn.neighbors import KNeighborsClassifier
import argparse
import joblib
import os
from logger import logger

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

    logger.info("Reading training data")
    df = pd.read_parquet(os.path.join(training_dir, "train.parquet"))

    predictors = ["x1", "x2"]
    target = ["is_blue"]

    X = df[predictors]
    y = df[target].values.ravel()

    model = KNeighborsClassifier(n_neighbors=5)

    logger.info("Fitting model")
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_dir, "model.mdl"))

    logger.info("Uploading model to s3")
    bucket = boto3.resource("s3", region_name="eu-west-1").Bucket("hastie")
    bucket.upload_file(os.path.join(model_dir, "model.mdl"), Key="model/model.mdl")


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.mdl"))
