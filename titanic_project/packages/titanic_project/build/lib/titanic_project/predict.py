import pandas as pd

from titanic_project.config import config
from titanic_project.processing.data_management import load_pipeline
from titanic_project.processing.validation import validate_inputs
from titanic_project import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_titanic_pipeline = load_pipeline(pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    
    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data = data)
    prediction = _titanic_pipeline.predict(validated_data.drop(config.TARGET, axis=1))
    response = {"prediction": prediction}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {response}"
    )

    return response

