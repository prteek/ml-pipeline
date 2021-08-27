import os
import sagemaker
import argparse
from sagemaker.sklearn import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from orchestrate import create_and_upload_training_code_package
from dotenv import load_dotenv
load_dotenv("./local_credentials.env")


session = sagemaker.Session()
role = os.environ['SAGEMAKER_EXECUTION_ROLE']


if __name__ == '__main__':
    bucket = 'hastie'
    file_list = ["train.py", "logger.py"]
    source_code_location = create_and_upload_training_code_package(
            file_list, source_code_package="sourcedir.tar.gz", bucket=bucket
        )

    output_path = "s3://hastie/model/artifacts"
    training_data_location = "s3://hastie/preprocess/data"


    estimator = SKLearn(
            base_job_name='hastie-classification',
            role=role,
            entry_point="train.py",
            framework_version="0.23-1",
            instance_count=1,
            instance_type="ml.m5.large",
            source_dir=source_code_location,
            output_path=output_path,
            use_spot_instances=True,
            max_wait=1800,
            max_run=300,
        )


    # Define custom metric: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-metrics.html

    # This must also be emitted by estimator (in train.py): print(f"f1_score = {f1};")
    objective_metric_name = 'f1_score'
    metric_definitions = [{'Name': 'f1_score', 'Regex': "f1_score = (.*?);"}]

    # Hyperparameters to tune
    hyperparams = {'n_neighbors':IntegerParameter(1,20)}
    
    # Setup tuner
    tuner = HyperparameterTuner(estimator, 
                                objective_metric_name,
                                hyperparams,
                                metric_definitions=metric_definitions,
                                objective_type='Maximize',
                                base_tuning_job_name='hastie-knn-tuning',
                                max_jobs=8,
                                max_parallel_jobs=4,
                                early_stopping_type='Auto')

    tuner.fit(inputs={'training':training_data_location}) # Same channel name as expected in train.py


    # Analysing the runs (alternately look in Sagemaker studio experiments and trials)
    exp = HyperparameterTuningJobAnalytics(hyperparameter_tuning_job_name=tuner.latest_tuning_job.name)
    od_jobs = exp.dataframe()
    od_jobs.sort_values('FinalObjectiveValue', ascending=0)
