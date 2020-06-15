import pathlib
import titanic_project

PACKAGE_ROOT = pathlib.Path(titanic_project.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"


# ====   PATHS ===================

TRAINING_DATA_FILE = "titanic.csv"
PIPELINE_NAME = 'logistic_regression.pkl'


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

CABIN = 'cabin'