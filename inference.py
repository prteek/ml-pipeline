from sklearn.neighbors import KNeighborsClassifier
import boto3
import json
import joblib
import os
import numpy as np
from utils import dict_to_dynamodb

def inference_handler(event, context=None):
    
    bucket_name = 'hastie'
    model_key = 'model/model.mdl'
    artefacts_directory = '/tmp'
    model_filename = 'model.mdl'
    preprocessor_key = 'preprocess/data/preprocessing_pipeline.mdl'
    preprocessor_filename = 'preprocessing_pipeline.mdl'

    bucket = boto3.resource('s3').Bucket(bucket_name)
    bucket.download_file(Key=model_key, Filename=os.path.join(artefacts_directory, model_filename))
    model = joblib.load(os.path.join(artefacts_directory, model_filename))
    
    bucket.download_file(Key=preprocessor_key, Filename=os.path.join(artefacts_directory, preprocessor_filename))    
    preprocessor = joblib.load(os.path.join(artefacts_directory, preprocessor_filename))
    
    payload = event['params']['header']
    print(payload)
    
    data = json.loads(payload['data'])
    time = payload['time']
    
    results = dict()
    results['date'] = time.split("T")[0]
    results['time'] = time.split("T")[1]
    results['is_blue'] = payload['is_blue']
    predictors = ['x1', 'x2']
    
    for i, p in enumerate(predictors):
        results[p] = data[i]
    
    X = np.reshape(data, (1, len(predictors)))
    Xt = preprocessor.transform(X)
    prediction = model.predict(Xt)[0]
    
    results['prediction'] = prediction
    
    results_table = 'hastie'
    table = boto3.resource('dynamodb').Table(results_table)
    status = table.put_item(Item=dict_to_dynamodb(results))
    
    return {'prediction': json.dumps(prediction)}

