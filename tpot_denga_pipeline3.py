import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from data_info import *
from preprocessing_helpers import *


new_sj_norm = [
                'precipitation_amt_mm',
                'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k',
                'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k',
                'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm',
                'station_avg_temp_c',
                'station_max_temp_c',
                'station_min_temp_c',
                'station_precip_mm'
]
new_sj_scale = [
                   'weekofyear',
]

extra_sj_cols = [
]

new_sj_cols = [LABEL_COLUMN] + CATEGORICAL_COLUMNS + new_sj_norm + new_sj_scale + extra_sj_cols + [DATETIME_COLUMN]
new_sj_cols_no_label = CATEGORICAL_COLUMNS + new_sj_norm + new_sj_scale + extra_sj_cols + [DATETIME_COLUMN]
sj_datasets, sj_norm_scale = generate_lstm_data(
    test_file,
    single_step=True,
    history_size=24,
    cols=new_sj_cols_no_label,
    norm_cols=new_sj_norm,
    scale_cols=new_sj_scale,
    extra_columns=extra_sj_cols,
    prepend_with_file=train_file,
    train_frac=1.0
)
sj_train_x, sj_train_y = sj_datasets[0][0]
sj_train_x = sj_train_x.reshape(sj_train_x.shape[0], sj_train_x.shape[1] * sj_train_x.shape[2])
print(np.size(sj_train_x))
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(sj_train_x, sj_train_y, random_state=42)
print(np.size(training_features))

# Average CV score on the training set was: -18.091824133868496
exported_pipeline = make_pipeline(
    Normalizer(norm="l2"),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.25, tol=0.01)),
    StackingEstimator(estimator=RidgeCV()),
    RandomForestRegressor(bootstrap=True, max_features=0.4, min_samples_leaf=6, min_samples_split=14, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


preds = np.concatenate((results, np.zeros(156)), axis=None)
export_test_to_csv(predictions=results,path=test_file, prefix='rf')