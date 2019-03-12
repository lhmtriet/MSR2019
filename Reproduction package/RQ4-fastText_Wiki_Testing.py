# Import libraries

import pandas as pd
import numpy as np
import time

from gensim.models import FastText

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

split_year = 2016

# Training set: Vuln. before 2016
train_indices = np.where(df_notnull.Year < split_year)[0]

# Testing set: Vuln. from 2016
test_indices = np.where(df_notnull.Year >= split_year)[0]

X = df_notnull.Cleaned_Desc

X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]

ft_model = None

#  Convert a sentence to a feature vector using fastText embeddings
def sen_to_vec(sen):

    words = sen.split()

    sen_vec = np.array([0.0] * 300)
    cnt = 0

    for w in words:
        try:
            sen_vec = sen_vec + ft_model[w]
            cnt += 1
        except:
            pass

    if cnt == 0:
        return np.random.rand(300)

    return sen_vec / (cnt * 1.0)

# Validate the data

def feature_model(X_train, X_test):

    global ft_model

    fastText_pretrained = 'crawl-300d-2M-subword.bin'
    ft_model = FastText.load_fasttext_format(fastText_pretrained, encoding='latin1')

    X_train_transformed = np.asarray(X_train.map(sen_to_vec).values.tolist())
    X_test_transformed = np.asarray(X_test.map(sen_to_vec).values.tolist())

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
        f1_score(y_test, y_pred, average='weighted')) + "\t" + "{:.3f}".format(train_time) + "\t" + "{:.3f}".format(
        pred_time)


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

t_time = time.clock()
X_train_transformed, X_test_transformed = feature_model(X_train, X_test)
print("Building vocab:", time.clock() - t_time)

for label in labels:
    print("Current output:", label, "\n")

    print("Classifier\tAccuracy\tAccuracy Std\tMacro F-Score\tMacro F-Score Std\tWeighted F-Score\tWeighted F-Score Std\tVal Time\n")

    cur_clf = clfs[label]

    for clf_name, clf in cur_clf.items():
        print(clf_name + "\t" + "", end='')

        y = df_notnull[label].values

        print(evaluate(clf, X_train_transformed, X_test_transformed, y))

    print("------------------------------------------------\n")

print("##############################################\n")
