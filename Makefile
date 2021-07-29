REPO=hastie
REGION=eu-west-1
ACCOUNT_ID=$(shell aws sts get-caller-identity --query Account --output text)
IMAGE=$(REPO):latest
IMAGE_ID=$(shell docker images -q $(IMAGE))

REPO_LAMBDA=hastie-lambda
IMAGE_LAMBDA=$(REPO_LAMBDA):latest
IMAGE_ID_LAMBDA=$(shell docker images -q $(IMAGE_LAMBDA))


help:
	@echo " - setup         	: Install requirements and dl4 library"
	@echo " - create-ecr-repo	: Creates repo in ECR with name specified in Makefile"
	@echo " - build-container	: Build container using Dockerfile for Sagemaker processing"
	@echo " - tag-image     	: Tag the last built image to latest version"
	@echo " - push-image     	: Push latest tagged image to ECR"
	@echo " - all           	: build-tag-push"
	@echo " - orchestrate		: orchestrate the workflow in Sagemaker pipelines"
	@echo " - clean         	: Clean cache and checkpoint folders"
	@echo " - black         	: Format .py files using black tool"


create-ecr-repo:
	aws ecr create-repository --repository-name $(REPO) --region $(REGION)


build-container:
	docker build -t $(IMAGE) -f Dockerfile .


build-container-lambda:
	docker build -t $(IMAGE_LAMBDA) -f lambda.Dockerfile .


tag-image:
	docker tag $(IMAGE_ID) $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE)


tag-image-lambda:
	docker tag $(IMAGE_ID_LAMBDA) $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_LAMBDA)


# First get login id password for AWS account and then push image
push-image:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE);docker push $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE)


push-image-lambda:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_LAMBDA);docker push $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_LAMBDA)


all: build-container tag-image push-image


setup:
	pip install -r requirements.txt

    
clean:
	rm -rf __pycache__/
	rm -rf .ipynb_checkpoints/


black:
	black *.py


orchestrate:
	python3 orchestrate.py --image-uri $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE)


orchestrate-inference: build-container-lambda tag-image-lambda push-image-lambda
