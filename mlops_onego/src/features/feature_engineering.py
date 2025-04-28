import os
import yaml
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------------- #

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# ----------------------------------------------------------------------------- #

def import_params(path: str) -> dict:
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('SUCCESSFULLY imported parameters from params.yaml')
        return params
    except Exception as e:
        logger.error('Something went wrong with params.yaml', exc_info=True)

# ----------------------------------------------------------------------------- #

def data_loading(path1: str, path2: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(path1)
        test_data = pd.read_csv(path2)

        logger.debug("SUCCESSFULLY loaded data")
        return train_data, test_data
    except Exception as e:
        logger.error('Problem occurred during data loading', exc_info=True)

# ----------------------------------------------------------------------------- #

def data_splitting(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    try:
        x_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        x_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        logger.debug('SUCCESSFULLY split data')
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error('Error occurred during data splitting', exc_info=True)

# ----------------------------------------------------------------------------- #

def tfidf_vector(x_train, x_test, y_train, y_test, max_feature: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        vectorizer = TfidfVectorizer(max_features=max_feature)

        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('SUCCESSFULLY applied TF-IDF and transformed data')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during TF-IDF transformation', exc_info=True)

# ----------------------------------------------------------------------------- #

def storing(x_train_tfidf: pd.DataFrame, x_test_tfidf: pd.DataFrame, data_path: str) -> None:
    try:
        x_train_tfidf.to_csv(os.path.join(data_path, 'train_transformed.csv'), index=False)
        x_test_tfidf.to_csv(os.path.join(data_path, 'test_transformed.csv'), index=False)

        logger.debug('SUCCESSFULLY stored data')
    except Exception as e:
        logger.error('Error occurred during storing process', exc_info=True)

# ----------------------------------------------------------------------------- #

def main():
    try:
        params = import_params('params.yaml')
        train_data, test_data = data_loading('./data/raw/train.csv', './data/raw/test.csv')

        x_train, x_test, y_train, y_test = data_splitting(train_data, test_data)

        max_feature = params['feature_engineering']['max_features']
        x_train_tfidf, x_test_tfidf = tfidf_vector(x_train, x_test, y_train, y_test, max_feature)

        data_path = os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)

        storing(x_train_tfidf, x_test_tfidf, data_path)

        logger.debug('main() ran SUCCESSFULLY')
    except Exception as e:
        logger.error('Error occurred in main() function', exc_info=True)

# ----------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------- #