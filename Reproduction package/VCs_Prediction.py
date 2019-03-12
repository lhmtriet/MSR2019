# Import libraries
import time
import numpy as np
import pickle

from scipy.sparse import hstack

model_folder = "Models/"

from sklearn.feature_extraction import text
from nltk.corpus import stopwords

my_stop_words = text.ENGLISH_STOP_WORDS
my_stop_words = list(my_stop_words) + list(stopwords.words('english'))
my_stop_words = list(set(my_stop_words))
my_stop_words.extend(['possibly', 'aka'])

from nltk.stem import PorterStemmer

# Text preprocessing module
def preprocess_text(s):

    ps = PorterStemmer()

    s = s.replace('\'', '').replace(', ', ' ').replace('; ', ' ').replace('. ', ' ').replace('(', '').\
        replace(')', '').strip().lower()

    if not s[-1].isalnum():
        s = s[:-1]
    words = s.split()
    s = ' '.join([ps.stem(w) for w in words if w not in my_stop_words])

    return np.array([s])

# Load the word and character feature models for each VC to transform the description to feature vectors
def transform_feature(description, label):

    word_model_name = model_folder + label + "_word.model"
    char_model_name = model_folder + label + "_char.model"

    word_model = pickle.load(open(word_model_name, 'rb'))
    char_model = pickle.load(open(char_model_name, 'rb'))

    X_word = word_model.transform(description)
    X_char = char_model.transform(description)

    X_transformed = hstack([X_word, X_char])

    return X_transformed

labels = ['CVSS2_Conf', 'CVSS2_Integrity', 'CVSS2_Avail', 'CVSS2_AccessVect',
          'CVSS2_AccessComp', 'CVSS2_Auth', 'CVSS2_Severity']

description = input("Please input a description:")

description = preprocess_text(description)

t_time = time.clock()

# Predict value of each VC using saved classification model
for label in labels:

    print(label, "is:", end='\t')

    X_transformed = transform_feature(description, label)

    ml_model_name = model_folder + label + "_clf.model"
    clf = pickle.load(open(ml_model_name, 'rb'))

    y_pred = clf.predict(X_transformed)[0]
    print(y_pred)

print("Prediction time:", time.clock() - t_time)
print("##############################################")
