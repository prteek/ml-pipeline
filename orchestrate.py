import sagemaker
from sagemaker.sklearn import SKLearn


session = sagemaker.Session()

role = sagemaker.get_execution_role()

source_dir = "file://"
output_path = "file://"

sk = SKLearn(role=role,
            entry_point='train.py',
            framework_version='0.23-1',
            instance_count=1,
            instance_type='local',
            source_dir=source_dir,
            output_path=output_path)

training = 'file://'
sk.fit({'training':training})

training = 'file://'
output = 'file://'

role = sagemaker.get_execution_role()
sk = SKLearn(entry_point='train.py',
             framework_version='0.23-1',
             role=role,
             instance_count=1, 
             instance_type='local',
             output_path=output,
)
