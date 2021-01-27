# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
import logging

from typing import Dict, List, Any
import numpy as np
import fsspec
import pandas as pd
import pickle
import tarfile

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sagemaker.sklearn.estimator import SKLearn


def split_data(master_table: pd.DataFrame, parameters: Dict) -> List:
    """Splits data into training and test sets.

        Args:
            data: Source data.
            parameters: Parameters defined in parameters.yml.

        Returns:
            A list containing split data.

    """
    X = master_table[
        [
            "engines",
            "passenger_capacity",
            "crew",
            "d_check_complete",
            "moon_clearance_complete",
        ]
    ].values
    y = master_table["price"].values
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"]
    )

    return [Xtrain, Xtest, Ytrain, Ytest]


def train_model_sagemaker(
    X_train_path: str, sklearn_estimator_kwargs: Dict[str, Any]
) -> str:
    """Train the linear regression model on SageMaker.

        Args:
            X_train_path: Full S3 path to `Xtrain` dataset.
            sklearn_estimator_kwargs: Keyword arguments that will be used
                to instantiate SKLearn estimator.

        Returns:
            Full S3 path to `model.tar.gz` file containing the model artifact.

    """
    sklearn_estimator = SKLearn(**sklearn_estimator_kwargs)

    # we need a path to the directory containing both
    # X_train (feature table) and y_train (target variable)
    inputs_dir = X_train_path.rsplit("/", 1)[0]
    inputs = {"train": inputs_dir}

    # wait=True ensures that the execution is blocked
    # until the job finishes on SageMaker
    sklearn_estimator.fit(inputs=inputs, wait=True)

    training_job = sklearn_estimator.latest_training_job
    job_description = training_job.describe()
    model_path = job_description["ModelArtifacts"]["S3ModelArtifacts"]
    return model_path


def untar_model(model_path: str) -> LinearRegression:
    """Unarchive the linear regression model artifact produced
    by the training job on SageMaker.

        Args:
            model_path: Full S3 path to `model.tar.gz` file containing
                the model artifact.

        Returns:
            Trained model.

    """
    with fsspec.open(model_path) as s3_file, tarfile.open(
        fileobj=s3_file, mode="r:gz"
    ) as tar:
        # we expect to have only one file inside the `model.tar.gz` archive
        filename = tar.getnames()[0]
        model_obj = tar.extractfile(filename)
        return pickle.load(model_obj)


def evaluate_model(
        Xtest: np.ndarray, Ytest: np.ndarray,
        regression_model: LinearRegression):
    Ypredict = regression_model.predict(Xtest)
    score = r2_score(Ytest, Ypredict)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f.", score)
