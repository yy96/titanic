import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from titanic_project.pipeline import titanic_pipe
from titanic_project.config import config
from titanic_project.processing.data_management import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name = config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),  # predictors
        data[config.TARGET],  # target
        test_size=0.2,  # percentage of obs in test set
        random_state=0)

    # fit pipeline
    titanic_pipe.fit(X_train, y_train)

    # save pipeline
    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == '__main__':
    run_training()
