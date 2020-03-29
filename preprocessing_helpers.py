import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_info import *
from datetime import datetime

LSTM_STEP = 6
LSTM_FUTURE_TARGET = 6
LSTM_HISTORY = 12


def extract_data(train_file_path, columns, categorical_columns=CATEGORICAL_COLUMNS, categories_desc=CATEGORIES):
    # Read csv file and return
    all_data = pd.read_csv(train_file_path, usecols=columns)
    if categorical_columns is not None:
        # map categorical to columns
        for feature_name in categorical_columns:
            mapping_dict = {categories_desc[feature_name][i]: categories_desc[feature_name][i] for i in
                            range(0, len(categories_desc[feature_name]))}
            all_data[feature_name] = all_data[feature_name].map(mapping_dict)

        # Change mapped categorical data to 0/1 columns
        all_data = pd.get_dummies(all_data, prefix='', prefix_sep='')

    # fix missing data
    all_data = all_data.interpolate(method='linear', limit_direction='forward')

    return all_data


def preproc_data(data, norm_cols=cols_to_norm, scale_cols=cols_to_scale):
    # Make a copy, not to modify oryginal data
    new_data = data.copy()
    # Normalize temp and percipation
    new_data[norm_cols] = StandardScaler().fit_transform(new_data[norm_cols])

    # Scale year and week no but within (0,1)
    new_data[scale_cols] = MinMaxScaler(feature_range=(0, 1)).fit_transform(new_data[scale_cols])

def create_features_columns(data: pd.DataFrame, categorical_columns=CATEGORICAL_COLUMNS,
                            numerical_columns=NUMERIC_COLUMNS):
    feature_columns = []

    for feature_name in categorical_columns:
        vocabulary = data[feature_name].unique()
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

    for feature_name in numerical_columns:
        feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                                                dtype=tf.float32))

    return feature_columns


def make_input_fn(X, y, n_epochs=1, shuffle=True):
    num_examples = len(y)

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(num_examples)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(num_examples)
        return dataset

    return input_fn


def split_data(data, train_frac=TRAIN_DATASET_FRAC, label_column=LABEL_COLUMN, filter_cols=None):
    train_data = data.sample(frac=train_frac, random_state=0)
    test_data = data.drop(train_data.index)

    train_y = train_data.pop(label_column) if label_column is not None else []
    test_y = test_data.pop(label_column) if label_column is not None else []

    if filter_cols is not None:
        train_data = train_data[filter_cols]
        test_data = test_data[filter_cols]

    return (train_data, train_y), (test_data, test_y)


def export_test_to_csv(predictions=None, path=test_file):
    org_test_data = pd.read_csv(path)
    org_test_data['total_cases'] = predictions
    org_test_data['total_cases'] = org_test_data['total_cases'].apply(lambda x: int(x) if x > 0 else 0)
    org_test_data[['city', 'year', 'weekofyear', 'total_cases']].to_csv(
        './out/out' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv", index=False)
