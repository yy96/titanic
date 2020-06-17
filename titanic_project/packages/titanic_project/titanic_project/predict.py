import pandas as pd

from titanic_project.config import config
from titanic_project.processing.data_management import load_pipeline
from titanic_project.processing.validation import validate_inputs

pipeline_file_name = "logistic_regression.pkl"
_titanic_pipeline = load_pipeline(pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    
    data = pd.read_json(input_data)
    #validated_data = validate_inputs(input_data = data)
    prediction = _titanic_pipeline.predict(data.drop(config.TARGET, axis=1))
    response = {"prediction": prediction}

    return response

