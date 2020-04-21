import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from datetime import datetime
from data_info import *


def export_test_to_csv(predictions=None, path=test_file):
    print(len(predictions))
    print('asas')

    org_test_data = pd.read_csv(path)
    org_test_data['total_cases'] = predictions
    org_test_data['total_cases'] = org_test_data['total_cases'].apply(lambda x: int(x) if x > 0 else 0)
    org_test_data[['city', 'year', 'weekofyear', 'total_cases']].to_csv(
        './out/out' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv", index=False)


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


tpot_data = extract_data(train_file, columns=CSV_COLUMNS)

test_data = extract_data(test_file, columns=CSV_COLUMNS_NO_LABEL)
print(tpot_data)
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('dengue_features_train_with_out.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('total_cases', axis=1)
print(features)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['total_cases'], random_state=42)

print(training_features)
print(testing_features)
print(training_target)
print(testing_target)

# Average CV score on the training set was: -6.981145679172217
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            FunctionTransformer(copy),
            make_pipeline(
                make_union(
                    FunctionTransformer(copy),
                    FunctionTransformer(copy)
                ),
                FeatureAgglomeration(affinity="euclidean", linkage="average"),
                SelectPercentile(score_func=f_regression, percentile=15),
                RBFSampler(gamma=0.8500000000000001)
            )
        ),
        SelectFwe(score_func=f_regression, alpha=0.005)
    ),
    StackingEstimator(
        estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.7000000000000001, min_samples_leaf=7,
                                      min_samples_split=11, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=2, min_samples_split=5,
                        n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(test_data)
export_test_to_csv(predictions=results)

print(len(results))
