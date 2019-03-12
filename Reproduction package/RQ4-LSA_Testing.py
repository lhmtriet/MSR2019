# Import libraries

import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from scipy.sparse import hstack

from sklearn.decomposition import TruncatedSVD

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
split_year = 2016

# Training set: Vuln. before 2016
train_indices = np.where(df_notnull.Year < split_year)[0]

# Testing set: Vuln. from 2016
test_indices = np.where(df_notnull.Year >= split_year)[0]

X = df_notnull.Cleaned_Desc
X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]

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

# Extract the word and character n-grams as vocabularies and transform n-grams into the feature vectors

def feature_model(X_train, X_test, config, start_word_ngram, end_word_ngram):

    vectorizer = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram)

    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    vectorizer.fit(X_train)
    word_vocab = vectorizer.vocabulary_

    start_char_ngram = 3
    end_char_ngram = 6

    use_idf = False
    norm = None

    if config == 2 or config == 6 or config == 7 or config == 8:
        use_idf = True
        norm = 'l2'

    tf_idf = TfidfVectorizer(stop_words=['aka'], ngram_range=(start_char_ngram, end_char_ngram), use_idf=use_idf, min_df=0.1,
                             analyzer='char', norm=norm)

    tf_idf.fit(X_train)
    char_vocabs = tf_idf.vocabulary_

    slt_char_vocabs = []
    for w in char_vocabs.keys():
        toks = w.split()
        if len(toks) == 1 and len(toks[0]) > 1:
            slt_char_vocabs.append(w.strip())

    slt_char_vocabs = set(slt_char_vocabs)

    word_vocab = set(word_vocab) - slt_char_vocabs

    tf_idf_char = TfidfVectorizer(stop_words=['aka'], ngram_range=(start_char_ngram - 1, end_char_ngram), use_idf=use_idf, min_df=0,
                                  analyzer='char', norm=norm, vocabulary=slt_char_vocabs)
    X_train_char = tf_idf_char.fit_transform(X_train)
    X_test_char = tf_idf_char.transform(X_test)

    vectorizer = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram,
                                  vocabulary=word_vocab)

    X_train_word = vectorizer.fit_transform(X_train)
    X_test_word = vectorizer.transform(X_test)

    X_train_transformed = hstack([X_train_word, X_train_char])
    X_test_transformed = hstack([X_test_word, X_test_char])

    X_train_transformed = X_train_transformed.astype(np.float64)
    X_test_transformed = X_test_transformed.astype(np.float64)

    return X_train_transformed, X_test_transformed

# Evaluate the models with Accuracy, Macro and Weighted F-Scores

def evaluate(clf, X_train_transformed, X_test_transformed, y):

    y_train, y_test = y[train_indices], y[test_indices]

    t_start = time.clock()

    clf.fit(X_train_transformed, y_train)

    train_time = time.clock() - t_start

    p_start = time.clock()

    y_pred = clf.predict(X_test_transformed)

    pred_time = time.clock() - p_start

    return "{:.3f}".format(accuracy_score(y_test, y_pred)) + "\t" + "{:.3f}".format(
        f1_score(y_test, y_pred, average='macro')) + "\t" + "{:.3f}".format(
        f1_score(y_test, y_pred, average='weighted')) + "\t" + "{:.3f}".format(train_time) + "\t" + "{:.3f}".format(pred_time)

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

reduced_dimension = 300

for label in labels:
    print("Current output:", label, "\n")

    t_time = time.clock()

    config = configs[label]
    start_word_ngram, end_word_ngram = get_config(config)

    cur_clf = clfs[label]

    X_train_transformed, X_test_transformed = feature_model(X_train, X_test, config, start_word_ngram, end_word_ngram)
    lsa = TruncatedSVD(n_components=reduced_dimension, algorithm='randomized', n_iter=10, random_state=42, tol=0.0)

    X_train_transformed = lsa.fit_transform(X_train_transformed)
    X_test_transformed = lsa.transform(X_test_transformed)

    print("Building vocab time:", time.clock() - t_time)

    print("Classifier\tAccuracy\tMacro F-Score\tWeighted F-Score\tTrain time\tPredict time\n")

    for clf_name, clf in cur_clf.items():
        print(clf_name + "\t" + "", end='')

        y = df_notnull[label].values
        print(evaluate(clf, X_train_transformed, X_test_transformed, y))

    print("##############################################")
