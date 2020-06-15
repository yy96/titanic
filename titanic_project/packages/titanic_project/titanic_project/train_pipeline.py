import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from titanic_project.pipeline import titanic_pipe
from titanic_project.config import config


def save_pipeline(*, pipline_to_persist) -> None:
    save_file_name = "logistic_regression.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipline_to_persist, save_path)

    print("saved pipeline")


def run_training() -> None:
    """Train the model."""

    # read training data
    data = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),  # predictors
        data[config.TARGET],  # target
        test_size=0.2,  # percentage of obs in test set
        random_state=0)

    # fit pipeline
    titanic_pipe.fit(X_train, y_train)

    # save pipeline
    save_pipeline(pipline_to_persist=titanic_pipe)


if __name__ == '__main__':
    run_training()
