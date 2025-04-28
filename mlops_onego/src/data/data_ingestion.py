from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import yaml
import os

# -----------------------------------------------------------------------------

logger = logging.getLogger("data_ingestion_logger")
logger.setLevel('DEBUG')

console_handle = logging.StreamHandler()
console_handle.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handle.setFormatter(formatter)

logger.addHandler(console_handle)

# -----------------------------------------------------------------------------

def add_yaml(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('SUCCESSFULLY parameters import from params.yaml')
        return params

    except Exception as e:
        logger.error('something wrong with params.yaml', exe_info=True)

# -----------------------------------------------------------------------------

def data_load(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)

        logger.debug("SUCCESSFULLY Data load from url")
        return df
        
    except Exception as e:
        logger.error("Problem in Data Loading", exc_info=True)

# -----------------------------------------------------------------------------

def process(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'],inplace=True)
        final_df = df[df['sentiment'].isin(['happiness','sadness'])]
        final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)

        logger.debug("SUCCESSFULLY Process got completed")
        return final_df

    except Exception as e:
        logger.error("Error in during process time", exc_info=True)

# -----------------------------------------------------------------------------

def data_store(train_data: pd.DataFrame, test_data: pd.DataFrame, raw_data_path: str) -> None:
    try:
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug("SUCCESSFULLY Data store in given path")

    except Exception as e:
        logger.error("Error occure during the data storing phase", exc_info=True)

# -----------------------------------------------------------------------------

def main():
    try:
        params = add_yaml('params.yaml')
        df = data_load('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process(df)
        test_size = params.get('test_size', 0.2)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        raw_data_path = os.path.join('data', 'raw')
        data_store(train_data, test_data, raw_data_path)


    except Exception as e:
        logger.error('Failed to complete the data_ingestion process', exc_info=True)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------