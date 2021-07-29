FROM public.ecr.aws/lambda/python:3.8


COPY ./requirements.txt ./
COPY ./inference.py ./
COPY ./utils.py ./
COPY ./local_credentials.env ./

RUN python3 -m pip install -r requirements.txt
