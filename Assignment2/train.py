import preprocessing_functions as pf
import config
import pandas as pd

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df, config.TARGET)

# get first letter from cabin variable
X_train = pf.extract_cabin_letter(X_train, 'cabin')
X_test = pf.extract_cabin_letter(X_test, 'cabin')

# impute categorical variables
X_train = pf.impute_na(X_train, config.CATEGORICAL_VARS)
X_test = pf.impute_na(X_test, config.CATEGORICAL_VARS)

# impute numerical variable
for var in config.IMPUTATION_DICT.keys():
    X_train = pf.add_missing_indicator(X_train, var)
    X_test = pf.add_missing_indicator(X_test, var)

    X_train = pf.impute_na(X_train, var, config.IMPUTATION_DICT[var])
    X_test = pf.impute_na(X_test, var, config.IMPUTATION_DICT[var])

# Group rare labels
for var in config.FREQUENT_LABELS.keys():
    X_train = pf.remove_rare_labels(X_train, var, config.FREQUENT_LABELS[var])
    X_test = pf.remove_rare_labels(X_test, var, config.FREQUENT_LABELS[var])

# encode categorical variables
for var in config.CATEGORICAL_VARS:
    X_train = pf.encode_categorical(X_train, var)
    X_test = pf.encode_categorical(X_test, var)

# check all dummies were added
X_train = pf.check_dummy_variables(X_train, config.DUMMY_VARIABLES)
X_test = pf.check_dummy_variables(X_test, config.DUMMY_VARIABLES)

# train scaler and save
pf.train_scaler(X_train, config.OUTPUT_SCALER_PATH)

# scale train set
X_train = pf.scale_features(X_train, config.OUTPUT_SCALER_PATH)
X_test = pf.scale_features(X_test, config.OUTPUT_SCALER_PATH)

# train model and save
pf.train_model(X_train, y_train, config.OUTPUT_MODEL_PATH)

print('Finished training')