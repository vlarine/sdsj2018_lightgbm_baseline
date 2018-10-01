import os
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from sklearn import model_selection

from utils import transform_datetime_features

ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024

def transform_categorical_features(df, categorical_values={}):
    # categorical encoding
    for col_name in list(df.columns):
        if col_name not in categorical_values:
            if col_name.startswith('id') or col_name.startswith('string'):
                categorical_values[col_name] = df[col_name].value_counts().to_dict()

        if col_name in categorical_values:
            col_unique_values = df[col_name].unique()
            for unique_value in col_unique_values:
                df.loc[df[col_name] == unique_value, col_name] = categorical_values[col_name].get(unique_value, -1)

    return df, categorical_values

def check_column_name(name):
    if name == 'line_id':
        return False
    if name.startswith('datetime'):
        return False
    if name.startswith('string'):
        return False
    if name.startswith('id'):
        return False

    return True


def load_data(filename, datatype='train', cfg={}):

    model_config = cfg
    model_config['missing'] = True

    # read dataset
    df = pd.read_csv(filename, low_memory=False)
    if datatype == 'train':
        y = df.target
        df = df.drop('target', axis=1)
        if df.memory_usage().sum() > BIG_DATASET_SIZE:
            model_config['is_big'] = True
    else:
        y = None
    print('Dataset read, shape {}'.format(df.shape))

    # features from datetime
    df = transform_datetime_features(df)
    print('Transform datetime done, shape {}'.format(df.shape))

    # categorical encoding
    if datatype == 'train':
        df, categorical_values = transform_categorical_features(df)
        model_config['categorical_values'] = categorical_values
    else:
        df, categorical_values = transform_categorical_features(df, model_config['categorical_values'])
    print('Transform categorical done, shape {}'.format(df.shape))

    # drop constant features
    if datatype == 'train':
        constant_columns = [
            col_name
            for col_name in df.columns
            if df[col_name].nunique() == 1
            ]
        df.drop(constant_columns, axis=1, inplace=True)

    # filter columns
    if datatype == 'train':
        model_config['used_columns'] = [c for c in df.columns if check_column_name(c) or c in categorical_values]
    used_columns = model_config['used_columns']
    print('Used {} columns'.format(len(used_columns)))

    line_id = df[['line_id', ]]
    df = df[used_columns]

    # missing values
    if model_config['missing']:
        df.fillna(-1, inplace=True)

    return df.values.astype(np.float16) if 'is_big' in model_config else df, y, model_config, line_id





