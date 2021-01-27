import argparse
import pickle
from os import getenv
from pathlib import Path
from typing import Any

from sklearn.linear_model import LinearRegression


def _pickle(path: Path, data: Any) -> None:
    """Pickle the object and save it to disk"""
    with path.open("wb") as f:
        pickle.dump(data, f)


def _unpickle(path: Path) -> Any:
    """Unpickle the object from a given file"""
    with path.open("rb") as f:
        return pickle.load(f)


def _get_arg_parser() -> argparse.ArgumentParser:
    """Instantiate the command line argument parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-data-dir", type=str, default=getenv("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--model-dir", type=str, default=getenv("SM_MODEL_DIR"))
    parser.add_argument(
        "--train", type=str, default=getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=getenv("SM_CHANNEL_TEST"))

    return parser


def main():
    """The main script entry point which will be called by SageMaker
    when running the training job
    """
    parser = _get_arg_parser()
    args = parser.parse_args()

    data_path = Path(args.train)
    X_train = _unpickle(data_path / "Xtrain.pkl")
    y_train = _unpickle(data_path / "Ytrain.pkl")

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    model_dir = Path(args.model_dir)
    _pickle(model_dir / "pickle_model.pkl", regressor)


if __name__ == "__main__":
    # SageMaker will run this script as the main program
    main()
