from tpot import TPOTRegressor
import pandas as pd
from data_info import *
from sklearn.model_selection import train_test_split


def extract_data(train_file_path, columns, categorical_columns=CATEGORICAL_COLUMNS, categories_desc=CATEGORIES,
                 interpolate=True):
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
    if interpolate:
        all_data = all_data.interpolate(method='linear', limit_direction='forward')

    return all_data


train_x = extract_data(train_file, columns=CSV_COLUMNS)
train_y = train_x.pop(LABEL_COLUMN)
print(len(train_x.to_numpy()))
print(len(train_y.to_numpy()))
X_train, X_test, y_train, y_test = train_test_split(train_x.to_numpy(), train_y.to_numpy(),
                                                    train_size=0.8, test_size=0.2, random_state=42)

tpot = TPOTRegressor(scoring="neg_mean_absolute_error", generations=200, population_size=10, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_denga_pipeline2.py')