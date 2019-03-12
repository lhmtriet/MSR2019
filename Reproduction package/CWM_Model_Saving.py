# Import libraries

import pandas as pd
import numpy as np
import time

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from scipy.sparse import hstack

# Load datasets

df = pd.read_csv('Data/all_cves_cleaned.csv')

# Extract year

import re


def extractYearFromId(id):
    return re.match(r'CVE-(\d+)-\d+', id).group(1)

# Year in CVE-ID
df['Year'] = df.ID.map(extractYearFromId).astype(np.int64)

# Extract non-null CVSS2 training & testing sets

notnull_indices = np.where(df.CVSS2_Avail.notnull())[0]

# Remove null values
df_notnull = df.iloc[notnull_indices]

# Year to split dataset
split_year = 2019

# Training set: Vuln. before 2016
train_indices = np.where(df_notnull.Year < split_year)[0]

X = df_notnull.Cleaned_Desc
X_train = X.iloc[train_indices]

model_folder = "Models/"

# Extract NLP features

# config 1: Bag-of-word without tf-idf
# config 2: Bag-of-word with tf-idf
# config 3: N-gram without tf-idf
# config 4: N-gram with tf-idf

def extract_features(config, start_word_ngram, end_word_ngram, vocabulary=None):
    if config == 1:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=False, min_df=0.001,
                               norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)
    elif config == 2:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=True, min_df=0.001,
                               norm='l2', token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)
    elif config < 6:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=False,
                               min_df=0.001, norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)

    return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=True,
                           min_df=0.001, norm='l2', token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)

def feature_model(X_train, config, start_word_ngram, end_word_ngram, label):

    word_vocab_generator = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram)

    X_train = X_train.astype(str)

    # Generate the word vocabulary
    word_vocab_generator.fit(X_train)
    word_vocab = word_vocab_generator.vocabulary_

    start_char_ngram = 3
    end_char_ngram = 6

    use_idf = False
    norm = None

    if config == 2 or config == 6 or config == 7 or config == 8:
        use_idf = True
        norm = 'l2'

    # Generate the character vocabulary
    char_vocab_generator = TfidfVectorizer(stop_words=['aka'], ngram_range=(start_char_ngram, end_char_ngram), use_idf=use_idf, min_df=0.1,
                             analyzer='char', norm=norm)

    char_vocab_generator.fit(X_train)
    char_vocabs = char_vocab_generator.vocabulary_

    # Filter the character vocabulary
    slt_char_vocabs = []
    for w in char_vocabs.keys():
        toks = w.split()
        if len(toks) == 1 and len(toks[0]) > 1:
            slt_char_vocabs.append(w.strip())

    slt_char_vocabs = set(slt_char_vocabs)

    # Remove word n-grams that are duplicate
    word_vocab = set(word_vocab) - slt_char_vocabs

    char_vectorizer = TfidfVectorizer(stop_words=['aka'], use_idf=use_idf, analyzer='char', norm=norm, vocabulary=slt_char_vocabs)
    X_train_char = char_vectorizer.fit_transform(X_train)

    word_vectorizer = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram,
                                  vocabulary=word_vocab)

    X_train_word = word_vectorizer.fit_transform(X_train)

    X_train_transformed = hstack([X_train_word, X_train_char])

    X_train_transformed = X_train_transformed.astype(np.float64)

    # Save the word and character feature models for future prediction
    word_model_name = model_folder + label + "_word.model"
    char_model_name = model_folder + label + "_char.model"

    pickle.dump(word_vectorizer, open(word_model_name, "wb"))
    pickle.dump(char_vectorizer, open(char_model_name, "wb"))

    return X_train_transformed

# Test the data

def evaluate(clf, X_train_transformed, y, label):

    y_train = y[train_indices]

    clf.fit(X_train_transformed, y_train)

    # Save the classification model
    ml_model_name = model_folder + label + "_clf.model"
    pickle.dump(clf, open(ml_model_name, 'wb'))


labels = ['CVSS2_Conf', 'CVSS2_Integrity', 'CVSS2_Avail', 'CVSS2_AccessVect',
          'CVSS2_AccessComp', 'CVSS2_Auth', 'CVSS2_Severity']


clfs = {'CVSS2_Conf': {'LGBM': LGBMClassifier(num_leaves=100, max_depth=-1, objective='multiclass', n_jobs=-1, random_state=42)},
        'CVSS2_Integrity': {'XGB': XGBClassifier(objective='multiclass', max_depth=0, max_leaves=100, grow_policy='lossguide',
                                 n_jobs=-1, random_state=42, tree_method='hist')},
        'CVSS2_Avail': {'LGBM': LGBMClassifier(num_leaves=100, max_depth=-1, objective='multiclass', n_jobs=-1, random_state=42)},
        'CVSS2_AccessVect': {'XGB': XGBClassifier(objective='multiclass', max_depth=0, max_leaves=100, grow_policy='lossguide',
                                 n_jobs=-1, random_state=42, tree_method='hist')},
        'CVSS2_AccessComp': {'LGBM': LGBMClassifier(num_leaves=100, max_depth=-1, objective='multiclass', n_jobs=-1, random_state=42)},
        'CVSS2_Auth': {'LR': LogisticRegression(C=0.1, multi_class='ovr', n_jobs=-1, solver='lbfgs', max_iter=1000,
                                     random_state=42)},
        'CVSS2_Severity': {'LGBM': LGBMClassifier(num_leaves=100, max_depth=-1, objective='multiclass', n_jobs=-1, random_state=42)}}

configs = {'CVSS2_Conf': 1, 'CVSS2_Integrity': 4, 'CVSS2_Avail': 1, 'CVSS2_AccessVect': 7, 'CVSS2_AccessComp': 1, 'CVSS2_Auth': 3, 'CVSS2_Severity': 5}

def get_config(config):
    start_word_ngram = 1
    end_word_ngram = 1

    if config == 1:
        print("Bag-of-word without tf-idf")
        start_word_ngram = 1
        end_word_ngram = 1
    elif config == 2:
        print("Bag-of-word with tf-idf")
        start_word_ngram = 1
        end_word_ngram = 1
    elif config <= 5:
        print("N-gram without tf-idf")
        if config == 3:
            start_word_ngram = 1
            end_word_ngram = 2
        elif config == 4:
            start_word_ngram = 1
            end_word_ngram = 3
        elif config == 5:
            start_word_ngram = 1
            end_word_ngram = 4
    else:
        print("N-gram with tf-idf")
        if config == 6:
            start_word_ngram = 1
            end_word_ngram = 2
        elif config == 7:
            start_word_ngram = 1
            end_word_ngram = 3
        elif config == 8:
            start_word_ngram = 1
            end_word_ngram = 4

    return start_word_ngram, end_word_ngram

for label in labels:
    print("Current output:", label, "\n")

    t_time = time.clock()

    config = configs[label]
    start_word_ngram, end_word_ngram = get_config(config)

    cur_clf = clfs[label]

    X_train_transformed = feature_model(X_train, config, start_word_ngram, end_word_ngram, label)
    print("Building vocab time:", time.clock() - t_time)

    print("Classifier\tAccuracy\tMacro F-Score\tWeighted F-Score\tTrain time\tPredict time\n")

    for clf_name, clf in cur_clf.items():
        print(clf_name + "\t" + "", end='')

        y = df_notnull[label].values
        print(evaluate(clf, X_train_transformed, y, label))

    print("##############################################")
