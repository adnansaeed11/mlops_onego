import json
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# -----------------------------------------------------------------------------

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# -----------------------------------------------------------------------------

def data_loads(path: str) -> pd.DataFrame:
    try:
        x_test_bow = pd.read_csv(path)

        logger.debug('SUCCESSFULLY data load')
        return x_test_bow

    except Exception as e:
        logger.error('There is some problem in data loading', exc_info=True)

# -----------------------------------------------------------------------------

def model_loads(path: str) -> xgb:
    try:
        with open(path, 'rb') as file:
            xgb = pickle.load(file)

            logger.debug('SUCCESSFULLY model load')
            return xgb
        
    except Exception as e:
        logger.error('There is some error in model loading', exc_info=True)


# -----------------------------------------------------------------------------

def data_spliting(x_test_bow: pd.DataFrame) -> np.ndarray:
    try:
        x_test = x_test_bow.iloc[:, 0:-1]
        y_test = x_test_bow.iloc[:, -1]

        logger.debug('SUCCESSFULLY data split')
        return x_test, y_test
    
    except Exception as e:
        logger.error('There is some problem in data spliting', exc_info=True)

# -----------------------------------------------------------------------------

def predictions(xgb: xgb, x_test: np.ndarray) -> float:
    try:
        y_pred = xgb.predict(x_test)
        y_pred_proba = xgb.predict_proba(x_test)[:, 1]

        logger.debug('SUCCESSFULLY prediction')
        return y_pred, y_pred_proba
    
    except Exception as e:
        logger.debug('Error in prediction phase', exc_info=True)

# -----------------------------------------------------------------------------

def evaluation(y_test: np.ndarray, y_pred: float, y_pred_proba: float) -> dict:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.debug('SUCCESSFULLY evaluated')
        return metrics
    
    except Exception as e:
        logger.debug('There is some problem during evaluation', exc_info=True)

# -----------------------------------------------------------------------------

def result_store(metrics: dict, path: str) -> None:
    try:
        with open(path, 'w') as file:
            json.dump(metrics, file, indent=4)

        logger.debug('SUCCESSFULLY metrics are stored')

    except Exception as e:
        logger.debug('There is some error in result storing phase')

# -----------------------------------------------------------------------------

def main():
    try:
        x_test_bow = data_loads(r'./data/processed/train_transformed.csv')
        xgb = model_loads('mlops_onego/models/model.pkl')
        x_test, y_test = data_spliting(x_test_bow)
        y_pred, y_pred_proba = predictions(xgb, x_test)
        metrics = evaluation(y_test, y_pred, y_pred_proba)
        result_store(metrics, 'mlops_onego/reports/metrics.json')

        logger.debug('SUCCESSFULLY run main() function')

    
    except Exception as e:
        logger.debug('There is some problem during evaluation', exc_info=True)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------