from tpot import TPOTRegressor
import pandas as pd
from data_info import *
from preprocessing_helpers import *
from sklearn.model_selection import train_test_split

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
    train_file,
    single_step=True,
    history_size=24,
    cols=new_sj_cols,
    norm_cols=new_sj_norm,
    scale_cols=new_sj_scale,
    extra_columns=extra_sj_cols,
    prepend_with_file=train_file,
    train_frac=1.0
)
sj_train_x, sj_train_y = sj_datasets[0][0]
sj_train_x = sj_train_x.reshape(sj_train_x.shape[0], sj_train_x.shape[1] * sj_train_x.shape[2])

X_train, X_test, y_train, y_test = train_test_split(sj_train_x, sj_train_y,
                                                    train_size=0.8, test_size=0.2, random_state=42)

tpot = TPOTRegressor(scoring="neg_mean_absolute_error", generations=10, population_size=80, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_denga_pipeline3.py')