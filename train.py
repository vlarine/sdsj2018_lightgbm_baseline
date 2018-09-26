import argparse
import os
import pandas as pd
import pickle
import time

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from sklearn import model_selection

from utils import transform_datetime_features
from sdsj_feat import load_data

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

ONEHOT_MAX_UNIQUE_VALUES = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()

    df_X, df_y, model_config, _ = load_data(args.train_csv)

    model_config['mode'] = args.mode

    params =  {
	'task': 'train',
	'boosting_type': 'gbdt',
	'objective': 'regression' if args.mode == 'regression' else 'binary',
	'metric': 'rmse',
	"learning_rate": 0.01,
	"num_leaves": 200,
	"feature_fraction": 0.70,
	"bagging_fraction": 0.70,
	'bagging_freq': 4,
	"max_depth": -1,
        "verbosity" : -1,
	"reg_alpha": 0.3,
	"reg_lambda": 0.1,
	#"min_split_gain":0.2,
	"min_child_weight":10,
	'zero_as_missing':True,
        'num_threads': 4,
    }

    params['seed'] = 1
    model = lgb.train(params, lgb.Dataset(df_X, label=df_y), 600)

    model_config['model'] = model
    model_config['params'] = params

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))
