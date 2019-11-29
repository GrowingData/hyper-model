import logging
import joblib
import json
import click
import os
import uuid
from typing import Dict, List
from xgboost import XGBClassifier
from hypermodel import hml
from hypermodel.features import one_hot_encode
from hypermodel.features import get_unique_feature_values, describe_features
from crashed.shared import MODEL_NAME
from crashed.shared import build_feature_matrix
from crashed.shared import FEATURES_NUMERIC, FEATURES_CATEGORICAL, TARGET


@hml.op()
def select_into(sql: str, output_dataset: str, output_table: str):
    pkg = hml.get_package()

    services = pkg.services
    services.warehouse.select_into(sql, output_dataset, output_table)
    return output_table


@hml.op()
def export_csv(bucket: str, dataset_name: str, table_name: str, filename: str):
    pkg = hml.get_package()

    services = pkg.services
    bucket_path = pkg.artifact_path(filename)
    bucket_url = services.warehouse.export_csv(bucket, bucket_path, dataset_name, table_name)

    return bucket_path


@hml.op()
def analyze_categorical_features(bucket: str, csv_path: str, artifact_name: str, columns: List[str]):
    pkg = hml.get_package()
    services = pkg.services

    training_df = services.lake.download_csv(bucket, csv_path)
    unique_feature_values = get_unique_feature_values(training_df, columns)

    # Add the artifact we have just produced
    return pkg.add_artifact_json(artifact_name, unique_feature_values)


@hml.op()
def analyze_numeric_features(bucket: str, csv_path: str, artifact_name: str, columns: List[str]
                             ):
    pkg = hml.get_package()
    services = pkg.services

    training_df = services.lake.download_csv(bucket, csv_path)
    unique_feature_values = describe_features(training_df, columns)

    # Add the artifact we have just produced
    return pkg.add_artifact_json(artifact_name, unique_feature_values)


@hml.op()
def build_matrix(bucket: str, csv_path: str, analysis_path_categorical: str, numeric_features: List[str], artifact_name: str):
    pkg = hml.get_package()
    services = pkg.services
    json_features = services.lake.download_string(bucket, analysis_path_categorical)
    unique_feature_values = json.loads(json_features)
    training_df = services.lake.download_csv(bucket, csv_path)
    encoded_df = one_hot_encode(training_df, unique_feature_values, throw_on_missing=True)

    for nf in numeric_features:
        encoded_df[nf] = training_df[nf]

    return pkg.add_artifact_dataframe(artifact_name, encoded_df)


@hml.op()
def train_model(bucket: str, matrix_path: str, target: str, artifact_name: str):
    pkg = hml.get_package()
    config = pkg.services.config
    services = pkg.services
    final_df = services.lake.download_csv(bucket, matrix_path)
    targets = final_df[target]
    feature_matrix = final_df.values

    classifier = XGBClassifier()
    model = classifier.fit(feature_matrix, targets, verbose=True)

    filename = uuid.uuid4()
    tmp_path = os.path.join(config.temp_path, f"{filename}.joblib")
    joblib.dump(model, tmp_path)

    artifact_path = pkg.add_artifact_file(artifact_name, tmp_path)
    os.remove(tmp_path)

    return artifact_path


# @hml.op()
# @hml.pass_context
# @hml.option("-m", "--message")
# def create_training(pkg: hml.HmlPackage, message=None):
#     logging.info(f"create_training: {message}")
#     services = pkg.services

#     column_string = ",".join((FEATURES_NUMERIC + FEATURES_CATEGORICAL))

#     query = f"""
#         SELECT {column_string}, {TARGET}
#         FROM crashed.crashes_raw
#         WHERE accident_date BETWEEN '2013-01-01' AND '2017-01-01'
#     """
#     services.warehouse.select_into(query, services.config.warehouse_dataset, BQ_TABLE_TRAINING)
#     logging.info(f"Wrote training set to {BQ_TABLE_TRAINING}.  Success!")

#     return f"Decorated message: {message}"


# @hml.op()
# @hml.pass_context
# @hml.option("-m", "--adjusted_message")
# def create_test(pkg: hml.HmlPackage, adjusted_message):
#     logging.info(f"create_test: My message is: {adjusted_message}")
#     services = pkg.services
#     column_string = ",".join((FEATURES_NUMERIC + FEATURES_CATEGORICAL))

#     query = f"""
#         SELECT {column_string}, {TARGET}
#         FROM crashed.crashes_raw
#         WHERE accident_date > '2018-01-01'
#     """
#     services.warehouse.select_into(query, services.config.warehouse_dataset, BQ_TABLE_TEST)

#     logging.info(f"Wrote test set to {BQ_TABLE_TEST}.  Success!")


# @hml.op()
# @hml.pass_context
# def train_model(pkg: hml.HmlPackage):
#     services = pkg.services

#     logging.info("Got Training DataFrame!")
#     training_df = services.warehouse.dataframe_from_table(services.config.warehouse_dataset, BQ_TABLE_TRAINING)

#     unique_feature_values = get_unique_feature_values(training_df, FEATURES_CATEGORICAL)
#     pkg.add_artifact_json("feature-summary-categorical.json", unique_feature_values)

#     # Train the model
#     model = train(model_container, training_df)

#     # Let out container know about the trained model
#     model_container.bind_model(model)

#     # Get the testing dataframe so that we can evaluate the model
#     test_df = services.warehouse.dataframe_from_table(services.config.warehouse_dataset, BQ_TABLE_TEST)
#     logging.info("Got Test DataFrame!")
#     # Run some evaluation against the model
#     evaluate_model(model_container, test_df)

#     # Publish this version of the model & data analysis
#     ref = model_container.publish()

#     model_container.dump_reference(ref)

#     # Create a merge request for this model to be deployed (don't do it here
#     # because we don't want to polute the repository with merge requests relating
#     # to test runs)
#     model_container.create_merge_request(reference=ref)
#     return


# def get_model_container(ctx):
#     models: Dict[str, ModelContainer] = ctx.obj["models"]
#     model_container = models[MODEL_NAME]
#     return model_container


# def train(model_container, data_frame):
#     logging.info(f"training: {model_container.name}: train")
#     feature_matrix = build_feature_matrix(model_container, data_frame)
#     targets = data_frame[model_container.target]

#     classifier = XGBClassifier()
#     model = classifier.fit(feature_matrix, targets, verbose=True)

#     return model


# def evaluate_model(model_container, data_frame):
#     logging.info(f"training: {model_container.name}: evaluate_model")

#     test_feature_matrix = build_feature_matrix(model_container, data_frame)
#     test_targets = data_frame[model_container.target]

#     # Evaluate the model against the training data to get an idea of where we are at
#     test_predictions = [v for v in model_container.model.predict(test_feature_matrix)]
#     correct = 0
#     for i in range(0, len(test_predictions)):
#         if test_predictions[i] == test_targets[i]:
#             correct += 1

#     pc_correct = int(100 * correct / len(test_predictions))
#     logging.info(f"Got {correct} out of {len(test_predictions)} ({pc_correct}%)")
