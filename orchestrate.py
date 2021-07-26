import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.inputs import TrainingInput
import os

session = sagemaker.Session()
role = sagemaker.get_execution_role()

image_uri ='434616802091.dkr.ecr.eu-west-1.amazonaws.com/hastie:latest'

def upload_code_helpers(filepath_list: list, bucket: str, prefix: str) -> str:
    for filepath in filepath_list:
        s3_location = session.upload_data(filepath, bucket, key_prefix=prefix)

    return f"s3://{bucket}/{prefix}/"


def create_processor(image_uri, job_name, local_mode=False):
    """Create script processor for either local mode or with ml.t3.medium"""
    if local_mode:
        instance_type = "local"
    else:
        instance_type = "ml.t3.medium"

    processor = ScriptProcessor(
        base_job_name=job_name,
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=instance_type
    )
    
    return processor

bucket = 'hastie'
helpers = upload_code_helpers(['utils.py'], bucket, prefix='helpers')

# ------------------ Fetch data step ------------------- #
date = ParameterString(name='date', default_value='2021-07-21')
fetch_data_processor = create_processor(image_uri, 'fetch_data', local_mode=False)
fetch_data_code_location = session.upload_data('fetch_data.py', bucket=bucket, key_prefix='fetch-data/code')
fetch_data_output_location = 's3://hastie/fetch-data/data'
fetch_data_step = ProcessingStep(name='fetch-data', 
                                processor=fetch_data_processor,
                                code=fetch_data_code_location,
                                job_arguments=['--date', date],
                                inputs=[ProcessingInput(input_name='helpers', 
                                                        source=helpers,
                                                       destination='/opt/ml/processing/input')],
                                outputs=[ProcessingOutput(output_name='raw_data',
                                                         source='/opt/ml/processing/output',
                                                         destination=fetch_data_output_location)])

# ----------------- Preprocessing step ------------------- #
preproc_processor = create_processor(image_uri, 'preprocess', local_mode=False)
preprocess_code_location = session.upload_data('preprocess.py', bucket=bucket, key_prefix='preprocess/code')
raw_data_location = fetch_data_step.properties.ProcessingOutputConfig.Outputs['raw_data'].S3Output.S3Uri
preprocess_output_location = 's3://hastie/preprocess/data'
preprocess_step = ProcessingStep(name='preprocess', 
                                processor=preproc_processor,
                                code=preprocess_code_location,
                                job_arguments=['--train-test-ratio', '0.8'],
                                inputs=[ProcessingInput(input_name='raw_data', 
                                                        source=raw_data_location,
                                                       destination='/opt/ml/processing/input')],
                                outputs=[ProcessingOutput(output_name='processed_data',
                                                         source='/opt/ml/processing/output',
                                                         destination=preprocess_output_location)])

# ------------------- Training step --------------------- #
import os
import tarfile
def create_tar(file, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        tar_handle.add(file)
               
            
source_code_package = 'sourcedir.tar.gz'
create_tar('train.py', source_code_package)
source_code_location = session.upload_data(source_code_package, bucket, key_prefix='train')
output_path = 's3://hastie/model/artifacts'
training_data_location = preprocess_step.properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri
estimator = SKLearn(role=role,
                    entry_point='train.py',
                    framework_version='0.23-1',
                    instance_count=1,
                    instance_type='ml.m5.large',
                    source_dir=source_code_location,
                    output_path=output_path)


training_step = TrainingStep(name='training', 
                             estimator=estimator,
                             inputs={'training':TrainingInput(training_data_location)}
                            )


# -------------------------- Pipeline ------------------------- #

training_pipeline = Pipeline(name='training-pipeline',
                            parameters=[date],
                            steps=[fetch_data_step, preprocess_step, training_step])


training_pipeline.upsert(role_arn=role)

training_pipeline.start(parameters={'date':'2021-07-21'})
