
# Import required libraries.
import tensorflow as tf
import nltk
from nltk.corpus import stopwords

# Download stopwords and load them for preprocessing.
nltk.download('stopwords')
stopwords = list(stopwords.words('indonesian'))

# Define feature and label keys.
LABEL_KEY = 'sentiment'
FEATURE_KEY = 'text'

def transformed_name(key):
    '''Rename transformed features.'''
    return key + '_xf'

def preprocessing_fn(inputs):
    '''
    Preprocess input features into transformed features.

    Args:
        inputs: Map from feature keys to raw features.

    Returns:
        outputs: Map from feature keys to transformed features.
    '''
    outputs = {}

    # Standardize and clean text data.
    text = tf.strings.lower(inputs[FEATURE_KEY])
    text = tf.strings.regex_replace(text, r'[^a-z\s]', ' ')
    text = tf.strings.regex_replace(text, r'\b(' + r'|'.join(stopwords) + r')\b\s*', ' ')
    text = tf.strings.strip(text)
    
    # Assign transformed features to outputs.
    outputs[transformed_name(FEATURE_KEY)] = text
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
