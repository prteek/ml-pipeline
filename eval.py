import sys
sys.path.append('/opt/ml/processing/input') # To read helper files
import numpy as np
import pandas as pd
import boto3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse
import joblib
import os
from logger import logger
import json



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data-dir", type=str, help="Location where test data is available (Local or inside the container)", default="/opt/ml/processing/input/data/")
    parser.add_argument("--model-dir", type=str, help="Location where model file (.mdl) is available (Local or inside the container)", default="/opt/ml/processing/input/model")
    parser.add_argument("--report-dir", type=str, help="Location where to save evaluation report (Local or inside the container)", default="/opt/ml/processing/evaluation")    
    
    
    args, _ = parser.parse_known_args()
    
    
    test_data_dir = args.test_data_dir
    model_dir = args.model_dir
    report_dir = args.report_dir
    

    logger.info('Reading Test data')
    df = pd.read_parquet(os.path.join(test_data_dir, "test.parquet"))

    predictors = ['x1', 'x2']
    target = ['is_blue']
        
    X = df[predictors]
    y = df[target].values.ravel()

    
    logger.info('Loading model')
    model = joblib.load(os.path.join(model_dir, "model.mdl"))
    
    y_pred = model.predict(X)
    
    report = dict()
    report['score'] = dict()
    report['score']['accuracy'] = dict()
    report['score']['accuracy']['value'] = accuracy_score(y, y_pred)
    report['score']['test_size'] = len(y)
    
    report_path = f"{report_dir}/evaluation.json"
    with open(report_path, "w") as f:
        f.write(json.dumps(report))
    
    
    
    
    