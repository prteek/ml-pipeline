FROM python:3.7

COPY ./requirements.txt /opt/ml/processing/input/
RUN python3 -m pip install -r /opt/ml/processing/input/requirements.txt
