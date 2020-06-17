import numpy as np

from titanic_project.predict import make_prediction
from titanic_project.processing.data_management import load_dataset

def test_make_single_prediction():

    test_data = load_dataset(file_name = "titanic.csv")
    single_test_json = test_data[0:1].to_json(orient = "records")

    subject = make_prediction(input_data = single_test_json)

    #print (subject.get('prediction')[0])

    assert subject is not None
    assert isinstance(subject.get('prediction')[0], np.int64)
    #assert math.ceil(subject.get('prediction')[0]) == 112476

def test_make_multiple_predictions():

    test_data = load_dataset(file_name = "titanic.csv")
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient = "records")

    subject = make_prediction(input_data = multiple_test_json)

    assert subject is not None
    # assume nothing is dropped
    assert len(subject.get('prediction').tolist()) == original_data_length