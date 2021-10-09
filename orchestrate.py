import os
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
import argparse
from utils import create_tar_from_files
from dotenv import load_dotenv
load_dotenv("./local_credentials.env")


session = sagemaker.Session()
role = os.environ['SAGEMAKER_EXECUTION_ROLE']

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
        instance_type=instance_type,
    )

    return processor


def create_and_upload_training_code_package(
    file_list, source_code_package="sourcedir.tar.gz", bucket="hastie"
):
    """Create tarfile package and upload to s3 for use in training"""
    create_tar_from_files(file_list, source_code_package)
    source_code_location = session.upload_data(
        source_code_package, bucket, key_prefix="train"
    )
    os.system(f"rm {source_code_package}")  # delete tarfile after upload
    return source_code_location


def orchestrate_training_pipeline(image_uri, bucket="hastie"):

    helpers = upload_code_helpers(["utils.py", "logger.py"], bucket, prefix="helpers")
    error_alert_code_location = session.upload_data(
        "error_alert.py", bucket=bucket, key_prefix="notify/code"
    )

    # ------------------ Fetch data step ------------------- #
    end_date = ParameterString(name="end-date", default_value="2021-07-27")
    number_of_days = ParameterString(name="number-of-days", default_value="1")
    fetch_data_processor = create_processor(image_uri, "fetch_data", local_mode=False)
    fetch_data_code_location = session.upload_data(
        "fetch_data.py", bucket=bucket, key_prefix="fetch-data/code"
    )
    fetch_data_output_location = "s3://hastie/fetch-data/data"
    fetch_data_step = ProcessingStep(
        name="fetch-data",
        processor=fetch_data_processor,
        code=fetch_data_code_location,
        job_arguments=["--end-date", end_date, "--number-of-days", number_of_days],
        inputs=[
            ProcessingInput(
                input_name="helpers",
                source=helpers,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="raw_data",
                source="/opt/ml/processing/output",
                destination=fetch_data_output_location,
            )
        ],
    )

    # ----------------- Preprocessing step ------------------- #
    preproc_processor = create_processor(image_uri, "preprocess", local_mode=False)
    preprocess_code_location = session.upload_data(
        "preprocess.py", bucket=bucket, key_prefix="preprocess/code"
    )
    raw_data_location = fetch_data_step.properties.ProcessingOutputConfig.Outputs[
        "raw_data"
    ].S3Output.S3Uri
    preprocess_output_location = "s3://hastie/preprocess/data"
    preprocess_step = ProcessingStep(
        name="preprocess",
        processor=preproc_processor,
        code=preprocess_code_location,
        job_arguments=["--train-test-ratio", "0.8"],
        inputs=[
            ProcessingInput(
                input_name="raw_data",
                source=raw_data_location,
                destination="/opt/ml/processing/input/data",
            ),
            ProcessingInput(
                input_name="helpers",
                source=helpers,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=preprocess_output_location,
            )
        ],
    )

    # ------------------- Training step --------------------- #

    file_list = ["train.py", "logger.py"]
    source_code_location = create_and_upload_training_code_package(
        file_list, source_code_package="sourcedir.tar.gz", bucket=bucket
    )
    output_path = "s3://hastie/model/artifacts"
    training_data_location = preprocess_step.properties.ProcessingOutputConfig.Outputs[
        "processed_data"
    ].S3Output.S3Uri
    estimator = SKLearn(
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
        hyperparameters={'C':1}
    )

    training_step = TrainingStep(
        name="training",
        estimator=estimator,
        inputs={"training": TrainingInput(training_data_location)},
    )

    # -------------------------- Eval ----------------------------- #
    eval_processor = create_processor(image_uri, "model-evaluation", local_mode=False)
    eval_code_location = session.upload_data(
        "eval.py", bucket=bucket, key_prefix="eval/code"
    )
    test_data_location = preprocess_step.properties.ProcessingOutputConfig.Outputs[
        "processed_data"
    ].S3Output.S3Uri
    model_file_location = "s3://hastie/model"
    eval_output_location = "s3://hastie/eval/data"
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    evaluation_step = ProcessingStep(
        name="model-evaluation",
        processor=eval_processor,
        code=eval_code_location,
        inputs=[
            ProcessingInput(
                input_name="helpers",
                source=helpers,
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                input_name="test_data",
                source=test_data_location,
                destination="/opt/ml/processing/input/data",
            ),
            ProcessingInput(
                input_name="model",
                source=model_file_location,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                input_name="model_artifacts",
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model/artifacts",
            ),
        ],  # Redundant input only makes graphs connect more logically
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=eval_output_location,
            )
        ],
        property_files=[evaluation_report],
    )

    # -------------------------- Register model ------------------------- #
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(eval_output_location),
            content_type="application/json",
        )
    )

    model_data = training_step.properties.ModelArtifacts.S3ModelArtifacts

    step_register = RegisterModel(
        name="register-model",
        estimator=estimator,
        model_data=model_data,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_metrics=model_metrics,
    )

    # -------------------------- Notify bad model ------------------------- #
    notify_bad_model_processor = create_processor(
        image_uri, "notify-bad-model", local_mode=False
    )

    notify_bad_model = ProcessingStep(
        name="notify-bad-model",
        processor=notify_bad_model_processor,
        code=error_alert_code_location,
        inputs=[
            ProcessingInput(
                input_name="helpers",
                source=helpers,
                destination="/opt/ml/processing/input",
            )
        ],
        job_arguments=["--message", "Model performance inadequate"],
    )

    # -------------------------- Check model performance ------------------------- #

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step=evaluation_step,
            property_file=evaluation_report,
            json_path="score.accuracy.value",
        ),
        right=0.8,
    )

    step_cond = ConditionStep(
        name="model-check",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[notify_bad_model],
    )

    # -------------------------- Pipeline ------------------------- #

    training_pipeline = Pipeline(
        name="training-pipeline",
        parameters=[end_date, number_of_days],
        steps=[
            fetch_data_step,
            preprocess_step,
            training_step,
            evaluation_step,
            step_cond,
        ],
    )

    training_pipeline.upsert(role_arn=role)

    return training_pipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-uri",
        type=str,
        help='ECR image uri that would be used for processing jobs e.g. "434616802091.dkr.ecr.eu-west-1.amazonaws.com/hastie:latest"',
        default="434616802091.dkr.ecr.eu-west-1.amazonaws.com/hastie:latest",
    )

    # Parse arguments
    args, _ = parser.parse_known_args()
    image_uri = args.image_uri

    training_pipeline = orchestrate_training_pipeline(image_uri)
#     training_pipeline.start(parameters={'end-date':'2021-07-27', 'number-of-days':4})
