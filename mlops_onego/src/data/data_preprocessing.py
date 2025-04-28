import os
import re
import nltk
import logging
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------------------------------

nltk.download('wordnet')
nltk.download('stopwords')

# -----------------------------------------------------------------------------

logger = logging.getLogger("preprocessing")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# -----------------------------------------------------------------------------

def lemmatization(text: str) -> str:
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)

# -----------------------------------------------------------------------------

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

# -----------------------------------------------------------------------------

def removing_numbers(text: str) -> str:
    text=''.join([i for i in text if not i.isdigit()])
    return text

# -----------------------------------------------------------------------------

def lower_case(text: str) -> str:

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

# -----------------------------------------------------------------------------

def removing_punctuations(text: str) -> str:
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

# -----------------------------------------------------------------------------

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# -----------------------------------------------------------------------------

def remove_small_sentences(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

# -----------------------------------------------------------------------------

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:

    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        logger.debug('SUCCESSFULLY convert into lower case')
    except Exception as e:
        logger.error('Failed in converting lower_case')

    try:
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        logger.debug('SUCCESSFULLY stop words removed')
    except Exception as e:
        logger.error('Failed in removing stop words')

    try:
        df.content=df.content.apply(lambda content : removing_numbers(content))
        logger.debug('SUCCESSFULLY numbers removed')
    except Exception as e:
        logger.error('Failed in numbers removing', exc_info=True)

    try:
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        logger.debug('SUCCESSFULLY punctuations removed')
    except Exception as e:
        logger.error('Failed in punctuations removing', exc_info=True)

    try:
        df.content=df.content.apply(lambda content : removing_urls(content))
        logger.debug('SUCCESSFULLY URLs removed')
    except Exception as e:
        logger.error('Failed URLs removing', exc_info=True)
    
    try:
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.debug('SUCCESSFULLY in lemmatization')
    except Exception as e:
        logger.error('Failed in lemmatization', exc_info=True)

    return df

# -----------------------------------------------------------------------------

def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('data loaded properly')

        # Transform the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)

    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', exc_info=True)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------