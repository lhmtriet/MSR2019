# Import libraries

import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

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

# Extract the character n-grams as vocabulary and transform n-grams into the feature vectors

def feature_model(X_train, X_test, config):

    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    start_char_ngram = 3
    end_char_ngram = 6

    use_idf = False
    norm = None

    if config == 2 or config == 6 or config == 7 or config == 8:
        use_idf = True
        norm = 'l2'

    vocab_generator = TfidfVectorizer(stop_words=['aka'], ngram_range=(start_char_ngram, end_char_ngram), use_idf=use_idf, min_df=0.1,
                             analyzer='char', norm=norm)

    # Generate character vocabulary
    vocab_generator.fit(X_train)
    char_vocabs = vocab_generator.vocabulary_

    # Filter character vocabulary
    slt_char_vocabs = []
    for w in char_vocabs.keys():
        toks = w.split()
        if len(toks) == 1 and len(toks[0]) > 1:
            slt_char_vocabs.append(w.strip())

    slt_char_vocabs = set(slt_char_vocabs)

    # Feature transformation

    char_vectorizer = TfidfVectorizer(stop_words=['aka'], ngram_range=(start_char_ngram - 1, end_char_ngram), use_idf=use_idf, min_df=0,
                                  analyzer='char', norm=norm, vocabulary=slt_char_vocabs)
    X_train_transformed = char_vectorizer.fit_transform(X_train)
    X_test_transformed = char_vectorizer.transform(X_test)

    X_train_transformed = X_train_transformed.astype(np.float64)
    X_test_transformed = X_test_transformed.astype(np.float64)

    return X_train_transformed, X_test_transformed

# Test the data

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

    if config == 1:
        print("Bag-of-word without tf-idf")
    elif config == 2:
        print("Bag-of-word with tf-idf")
    elif config <= 5:
        print("N-gram without tf-idf")
    else:
        print("N-gram with tf-idf")

# Iterate over each label and corresponding optimal classifier and NLP representation for testing
for label in labels:
    print("Current output:", label, "\n")

    t_time = time.clock()

    config = configs[label]
    get_config(config)

    cur_clf = clfs[label]

    X_train_transformed, X_test_transformed = feature_model(X_train, X_test, config)
    print("Building vocab time:", time.clock() - t_time)

    print("Classifier\tAccuracy\tMacro F-Score\tWeighted F-Score\tTrain time\tPredict time\n")

    for clf_name, clf in cur_clf.items():
        print(clf_name + "\t" + "", end='')

        y = df_notnull[label].values
        print(evaluate(clf, X_train_transformed, X_test_transformed, y))

    print("##############################################")
