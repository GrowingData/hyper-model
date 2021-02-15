# Hyper Model

Hyper Model helps take Machine Learning to production, created by [Growing Data](https://growingdata.com.au).

Hyper Model takes an opinionated and simple approach to the deployment, testing and monitoring of Machine Learning models, leveraging Kubernetes and Kubeflow to do all the important work.

# API Documentation

API / SDK documentation for Hyper Model is available at https://docs.hypermodel.com/en/latest/

# Getting started

## 1. Install conda

## 2. Create a conda environment & install the development package

```sh
conda create --name hml-dev python=3.8
conda install -n hml-dev mypy scikit-learn pandas joblib flask waitress click tqdm
conda install -c conda-forge xgboost
conda activate hml-dev

```

## 3. Install the HyperModel pip package for local development

```sh
cd src/hyper-model/
python -m pip install --upgrade setuptools wheel
pip install -e .

```

# Example Pipeline

```python
  @hml.pipeline(app.pipelines, cron="0 0 * * *", experiment="demos")
  def crashed_pipeline():
      """
      This is where we define the workflow for this pipeline purely
      with method invocations.
      """
      create_training_op = pipeline.create_training()
      create_test_op = pipeline.create_test()
      train_model_op = pipeline.train_model()

      # Set up the dependencies for this model
      (
          train_model_op
          .after(create_training_op)
          .after(create_test_op)
      )

```

# Anatomy of an ML Project in Hyper Model

Hyper Model Projects provide are self-contained Python Packages, providing all the code required for both Training and Inferences phases. Hyper Model Projects are executable locally as console scripts.

The code in the walk through is found in this repository here: https://github.com/GrowingData/hyper-model/tree/master/demo/car-crashes

Project layout:

- demo_pkg/
  - demo
    - pipeline.py
    - inference.py
    - shared.py
    - app.Dockerfile
    - start.py
  - setup.py
  - Readme.md

Lets run through the purpose of each file:

## `setup.py`

This file is reponsible for defining the Python Package that this application will be run as, as per
any other python package.

```python
from setuptools import setup, find_packages

NAME = "crashed"
VERSION = "0.0.80"
REQUIRES = [
    "click",
    "kfp",
    "xgboost",
    "pandas",
    "sklearn",
    "xgboost",
    "google-cloud",
    "google-cloud-bigquery",
    "hypermodel",
]

setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRES,
    packages=find_packages(),
    python_requires=">=3.5.3",
    include_package_data=True,
    entry_points={"console_scripts": ["crashed = crashed.start:main"]},
)

```

Of note is that we define a console_script entrypoint which enables us to launch the project via the command line. The examples, and Kubeflow Workflow definitions rely on an entrypoint being defined in this way. E.g. `entry_points={"console_scripts": ["crashed = crashed.start:main"]}` is important

## `start.py`

This is our entrypoint into the application (HmlApp), along with its two central components the HmlPipelineApp (used for data processing and training) and the HmlInferenceApp (used for generating inferences / predictions via Json API).

### Main, wrapper method (start.py)

```python
import logging
from typing import Dict, List
from hypermodel import hml
from flask import request

# Import my local modules
from crashed import shared, pipeline, inference

def main():
    # Create a reference to our "App" object which maintains state
    # about both the Inference and Pipeline phases of the model
    app = hml.HmlApp(
        name="car-crashes",
        platform="GCP",
        image_url=os.environ["DOCKERHUB_IMAGE"] + ":" + os.environ["CI_COMMIT_SHA"],
        package_entrypoint="crashed",
        inference_port=8000,
        k8s_namespace="kubeflow")

  # ... Code from the rest of the walkthrough ...

  app.start()
```

An HmlApp is responsible for managing both the Pipeline and Inference phases of the application, helping to manage shared functionality, such as the CLI and Kubeflow integration. The HmlApp is also responsible for deploying the HmlPipelineApp to Kubeflow, and deploying the HmlInferenceApp to Kubernetes as a Deployment / Service.

An `HmlApp` may have multiple `HmlPipelines` and `ModelContainers`, but only a single `HmlInferenceApp`.

The `image_url` field here is the address of the Docker Image containing a fully functional container with the package installed. When this container is loaded, it should be possible to execute the `package_entrypoint` script within the container and have this code be exectured.

Note: In the future, the HmlInferenceApp will utilise KFServing.

### Define & Register a reference for the ML Model

HyperModel maintains meta data about the Machine Learning model, including information about features, the "target" (e.g. what we are trying to predict). This helps to build re-usable pipeline operations, as we can always rely on having metadata in a certain format.

The actual metadata for this example is defined in `shared.py`, and can be registered with our HmlApp using the code below:

```python
    crashed_model = shared.crashed_model_container(app)
    app.register_model(shared.MODEL_NAME, crashed_model)
```

The `HmlModelContainer` object is also responsible for building the ModelReference during the training / pipeline phase. This `ModelReference` is then loaded during Initialization of the Inference Application. The `ModelReference` contains information about the distribution of numeric features, distinct values of categorical features and the all important `.joblib` file containing the actual model.

### Define your Pipeline

```python
    @hml.pipeline(app.pipelines, cron="0 0 * * *", experiment="demos")
    def crashed_pipeline():
        """
        This is where we define the workflow for this pipeline purely
        with method invocations.
        """
        create_training_op = pipeline.create_training()
        create_test_op = pipeline.create_test()
        train_model_op = pipeline.train_model()

        # Set up the dependencies for this model
        (
            train_model_op
            .after(create_training_op)
            .after(create_test_op)
        )
```

This method defines the Kubeflow Pipeline `crashed_pipeline` which will be deployed using the `demos` experiment within Kubeflow. Each function invocation within the `crashed_pipeline()` method defines Kubeflow ContainerOps which execute the `script_name` defined above in `config` with the correct CLI parameters.

The `@hml.pipeline` decorator is essentially a wrapper for the Kubeflow SDK's `@dsl.pipeline` decorator but with additional functionality to enable each `ContainerOp` to be executed via the command line.

### Configure Pipeline Operations for Deployment

Within Kubeflow, all Ops within a Pipeline are executed as invocations of Docker Images managed by Kubernetes. This means that we often need to apply additional configuration for the deployment of these containers.

The `@hml.deploy_op` decorator lets us configure the Container within the Kubeflow Pipelines Workflow so that we can bind secrets, bind environment variables, mount volumes, etc.

```python
    @hml.deploy_op(app.pipelines)
    def op_configurator(op):
        """
        Configure our Pipeline Operation Pods with the right secrets and
        environment variables so that it can work with our cloud
        provider's services
        """
        (op
            # Service account for authentication / authorisation
            .with_gcp_auth("svcacc-tez-kf")
            .with_env("GCP_PROJECT", "grwdt-dev")
            .with_env("GCP_ZONE", "australia-southeast1-a")
            .with_env("K8S_NAMESPACE", "kubeflow")
            .with_env("K8S_CLUSTER", "kf-crashed")
        )
        return op
```

#### Define your Inference API

With the model built, we need to think about how we can use the model to make predictions. We do this by defining a method to define routes and initialize our model, decorated with the `@hml.inference` decorator.

```python
    @hml.inference(app.inference)
    def crashed_inference(inference_app: hml.HmlInferenceApp):
        # Get a reference to the current version of my model
        model_container = inference_app.get_model(shared.MODEL_NAME)
        model_container.load()

        # Define our routes here, which can then call other functions with more
        # context
        @inference_app.flask.route("/predict", methods=["GET"])
        def predict():
            logging.info("api: /predict")

            feature_params = request.args.to_dict()
            return inference.predict_alcohol(inference_app, model_container, feature_params)
```

When the inference application is executed (e.g. with `crashed inference run-prod`), this function will be executed prior to the Flask application starting. This provides us with an opportunity to load the required model into memory (e.g. from the DataLake) using `model_container.load()`.

With the model loaded into memory, we can also define our routes to actually make predictions. In this example we are simple passing the execution context to the method defined in `inference.predict_alcohol()`.

At present, the InferenceAPI's are deployed using standard Kubernetes Deployments / Services / Ingress. In the future, we plan on migrating this to KFServing.

### Configure Inference API for deployment

Again, containers deployed to Kubernetes will require configuration prior to deployment to bind Service Accounts, mount volumes, set resource limits, etc. This is achieved by using the `@hml.deploy_inference` decorator which enables us to configure / override the default `Deployment` and `Service` used to run the Inference API in Kubernetes.

```python
    @hml.deploy_inference(app.inference)
    def deploy_inference(deployment: hml.HmlInferenceDeployment):
        logging.info(f"Preparing deploying: {deployment.deployment_name} ({deployment.k8s_container.image} -> {deployment.k8s_container.args} )")

        (
            deployment
            .with_gcp_auth("svcacc-tez-kf")
            .with_empty_dir("tmp", "/temp")
            .with_empty_dir("artifacts", "/artifacts")
        )
```

## pipeline.py

This module defines our different pipeline operations, with functions decorated with `@hml.op()` representing Kubeflow Operations to be run within a Pipeline. Importantly, Kubeflow Operations are designed to be re-used and thus are only bound to a Pipeline at run time, via the function decorated with @hml.pipeline().

It is importand that the pipeline build a "ModelContainer" object, serialized as JSON which contains details about the newly trained model, including a reference to the joblib file defining the final model. This `ModelContainer` object will be loaded by the `InferencesApp`

## shared.py

Both the Training and Inference phases of the project will share functionality, especially regarding key elements of pre-processing such as encoding & normalisation. Methods relating to shared functionality live in this file by way of example, but obviously may span multiple modules.

## inference.py

This module provides functionality to create inferences based on data, without all fuss of dealing with Flask, HyperModel or other libraries.

# Command Line Interface

All Hyper Model applications can be run from the command line using intuitive commands:

## Execute a Pipeline Step

```sh
<your_app> pipelines <your_pipeline_name> run <your_pipeline_step>
```

## Run all steps in a Pipeline

```sh
<your_app> pipelines <your_pipeline_name> run-all
```

## Deploy your Pipeline to Kubeflow (Development)

```sh
<your_app> pipelines <your_pipeline_name> deploy-dev
```

## Deploy your Pipeline to Kubeflow (Production)

```sh
<your_app> pipelines <your_pipeline_name> deploy-prod
```

## Serve Inference Requests (dev)

Run using the Flask based development environment

```sh
<your_app> inference run-dev
```

## Serve Inference Requests (prod)

Run using the Waitress based serving engine

```sh
<your_app> inference run-prod
```

## Deploy the Inference Application

```sh
<your_app> inference deploy
```

# Development setup

```sh
conda create --name hml-dev python=3.7
activate hml-dev
conda install -n hml-dev mypy pandas joblib flask waitress click tqdm


cd src/hyper-model/
pip install -e ,

```

Set Visual Studio Code to use the newly created `hml-dev` environment

## About Growing Data

Growing Data is a Melbourne based consultancy, focussed on [AI & Machine Learning](https://growingdata.com.au/data-services/ai-and-machine-learning/), [Data Engineering](https://growingdata.com.au/data-services/data-engineering/) and [Data Strategy & Governance](https://growingdata.com.au/data-services/data-strategy/).
