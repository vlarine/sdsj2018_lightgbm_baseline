import argparse
import os
import pandas as pd
import pickle
import time

from utils import transform_datetime_features
from sdsj_feat import load_data

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    X_scaled, _, _, df = load_data(args.test_csv, datatype='test', cfg=model_config)

    model = model_config['model']
    #df = pd.read_csv(args.test_csv, usecols=['line_id',])
    #print(args.test_csv)
    #df = pd.read_csv(args.test_csv)
    if model_config['mode'] == 'regression':
        df['prediction'] = model.predict(X_scaled)
    elif model_config['mode'] == 'classification':
        #df['prediction'] = model.predict_proba(X_scaled)[:, 1]
        df['prediction'] = model.predict(X_scaled)

    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))
