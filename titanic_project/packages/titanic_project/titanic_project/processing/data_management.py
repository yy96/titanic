import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from titanic_project.config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data

def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = "logistic_regression.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print("saved pipeline")

def load_pipeline(file_name: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)

    return saved_pipeline