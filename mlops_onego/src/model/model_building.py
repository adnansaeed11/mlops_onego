import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb

# -----------------------------------------------------------------------------

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# -----------------------------------------------------------------------------

def load_params(path: str) -> dict:
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('SUCCESSFULLY paramters load from params.yaml')
        return params

    except Exception as e:
        logger.error('There is some problem with load_params file')

# -----------------------------------------------------------------------------

def data_loading(path1: str) -> pd.DataFrame:
    try:
        train_transformed_countvect = pd.read_csv(path1)
        logger.debug('SUCCESSFULLY data loaded')
        return train_transformed_countvect
    
    except Exception as e:
        logger.error('Something error in data loading file', exc_info=True)

# -----------------------------------------------------------------------------

def training_spliting(train_transformed_countvect: pd.DataFrame) -> np.ndarray:
    try:
        X_train_countvect = train_transformed_countvect.iloc[:, 0:-1]
        y_train_countvect = train_transformed_countvect.iloc[:, -1]

        logger.debug('SUCCESSFULLY training data split into two parts')
        return X_train_countvect, y_train_countvect

    except Exception as e:
        logger.error('There is some issue in training_spliting function')

# -----------------------------------------------------------------------------

def training_time(X_train_countvect: np.ndarray, y_train: np.ndarray, max_depth: int, learning_rate: int, n_estimators: int) -> xgb:
    try:
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators
        )
        xgb_model.fit(X_train_countvect, y_train)

        logger.debug('SUCCESSFULLY Model Fit')
        return xgb_model

    except Exception as e:
        logger.error('There is an issue in training_time function', exc_info=True)


# -----------------------------------------------------------------------------

def model_store(model: xgb, path: str) -> None:
    try:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('SUCCESSFULLY Model Stored')
    except Exception as e:
        logger.error('There is something issue in model_store file', exc_info=True)

# -----------------------------------------------------------------------------

def main():
    try:
        params = load_params('params.yaml')
        train_transformed = data_loading(r'./data/processed/train_transformed_countvect.csv')
        X_train_countvect, y_train_countvect = training_spliting(train_transformed)

        max_depth = params['model_building']['max_depth']
        learning_rate = params['model_building']['learning_rate']
        n_estimators = params['model_building']['n_estimators']

        xgb_model = training_time(X_train_countvect, y_train_countvect, max_depth, learning_rate, n_estimators)
        model_store(xgb_model, 'mlops_onego/models/model.pkl')
        logger.debug('SUCCESSFULLY run main() file')

    except Exception as e:
        logger.error('Error occur in main() function', exc_info=True)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------