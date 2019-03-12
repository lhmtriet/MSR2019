# Import libraries

import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
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
all_train_indices = np.where(df_notnull.Year < split_year)[0]

X = df_notnull.Cleaned_Desc.iloc[all_train_indices].values

# Extract NLP features

# config 1: Bag-of-word without tf-idf
# config 2: Bag-of-word with tf-idf
# config 3: N-gram without tf-idf
# config 4: N-gram with tf-idf

def extract_features(config, start_word_ngram, end_word_ngram):
    if config == 1:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=False, min_df=0.001,
                               norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')
    elif config == 2:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=True, min_df=0.001,
                               norm='l2', token_pattern=r'\S*[A-Za-z]\S+')
    elif config < 6:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=False,
                               min_df=0.001, norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')

    return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=True,
                           min_df=0.001, norm='l2', token_pattern=r'\S*[A-Za-z]\S+')

# Build the Classifiers

def build_classifiers(config):

    clfs = {'NB': MultinomialNB(),
            'SVM': OneVsRestClassifier(LinearSVC(random_state=42, C=0.1, max_iter=1000), n_jobs=-1)}

    if config == 2 or config == 6 or config == 7 or config == 8:
        clfs['LR'] = LogisticRegression(C=10, multi_class='ovr', n_jobs=-1, solver='lbfgs', max_iter=1000,
                                        random_state=42)
    else:
        clfs['LR'] = LogisticRegression(C=0.1, multi_class='ovr', n_jobs=-1, solver='lbfgs', max_iter=1000,
                                        random_state=42)

    clfs['RF'] = RandomForestClassifier(n_estimators=100, max_depth=None, max_leaf_nodes=None, random_state=42,
                                        n_jobs=-1)
    clfs['XGB'] = XGBClassifier(objective='multiclass', max_depth=0, max_leaves=100, grow_policy='lossguide',
                                n_jobs=-1, random_state=42, tree_method='hist')
    clfs['LGBM'] = LGBMClassifier(num_leaves=100, max_depth=-1, objective='multiclass', n_jobs=-1, random_state=42)
	
    return clfs

# Extract the n-grams and transform n-grams into the feature vectors

def feature_model(X_train, X_test, y_test, config, start_word_ngram, end_word_ngram):

    # Create vectorizer
    vectorizer = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram)

    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    # Remove rows with all zero values
    test_df = pd.DataFrame(X_test_transformed.todense())
    results = test_df.apply(lambda x: x.value_counts().get(0.0, 0), axis=1)
    non_zero_indices = np.where(results < len(test_df.columns))[0]

    X_train_transformed = X_train_transformed.astype(np.float64)
    X_test_transformed = X_test_transformed.astype(np.float64)

    return X_train_transformed, X_test_transformed[non_zero_indices], y_test[non_zero_indices]

# Evaluate the models with Accuracy, Macro and Weighted F-Scores

def evaluate(clf, X_train_transformed, X_test_transformed, y_train, y_test):

    clf.fit(X_train_transformed, y_train)

    y_pred = clf.predict(X_test_transformed)

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'),\
           f1_score(y_test, y_pred, average='weighted')

#  Data splitting in time-based k-fold cross-validation

def validate_data(clf, y, config, start_word_ngram, end_word_ngram):

    accs = []
    f_mac = []
    f_weighted = []

    t_start = time.clock()

    k_fold = 5
    start_year = split_year - k_fold

    # 5-fold cross-validation with time order
    for year in range(start_year, split_year):

        # Split training and testing sets
        train_indices = np.where(df_notnull.iloc[all_train_indices].Year < year)[0]
        test_indices = np.where(df_notnull.iloc[all_train_indices].Year == year)[0]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # word n-gram generation & feature transformation
        X_train_transformed, X_test_transformed, y_test = feature_model(X_train, X_test, y_test, config, start_word_ngram, end_word_ngram)

        # training and evaluation
        results = evaluate(clf, X_train_transformed, X_test_transformed, y_train, y_test)

        # store temporary results
        accs.append(results[0])
        f_mac.append(results[1])
        f_weighted.append(results[2])

    # List of results in each fold
    accs = np.asarray(accs)
    f_mac = np.asarray(f_mac)
    f_weighted = np.asarray(f_weighted)

    val_time = time.clock() - t_start

    return "{:.3f}".format(np.mean(accs)) + "\t" + "{:.3f}".format(np.std(accs)) + "\t" + "{:.3f}".format(
        np.mean(f_mac)) + "\t" + "{:.3f}".format(np.std(f_mac)) + "\t" + "{:.3f}".format(
        np.mean(f_weighted)) + "\t" + "{:.3f}".format(np.std(f_weighted)) + "\t" + "{:.3f}".format(val_time)


labels = ['CVSS2_Conf', 'CVSS2_Integrity', 'CVSS2_Avail', 'CVSS2_AccessVect',
          'CVSS2_AccessComp', 'CVSS2_Auth', 'CVSS2_Severity']

# config 1: Bag-of-word without tf-idf
# config 2: Bag-of-word with tf-idf
# config 3: N-gram without tf-idf
# config 4: N-gram with tf-idf

result_file = 'val_with_time.txt'

configs = list(range(1, 9))

with open(result_file, 'w') as fout:
    for config in configs:

        print("Current config:", config)
        fout.write("Current config:" + str(config) + "\n")

        start_word_ngram = 1
        end_word_ngram = 1

        if config == 1:
            print("Bag-of-word without tf-idf")
            fout.write("Bag-of-word without tf-idf\n")
            start_word_ngram = 1
            end_word_ngram = 1
        elif config == 2:
            print("Bag-of-word with tf-idf")
            fout.write("Bag-of-word with tf-idf\n")
            start_word_ngram = 1
            end_word_ngram = 1
        elif config <= 5:
            print("N-gram without tf-idf")
            fout.write("N-gram without tf-idf\n")
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
            fout.write("N-gram with tf-idf\n")
            if config == 6:
                start_word_ngram = 1
                end_word_ngram = 2
            elif config == 7:
                start_word_ngram = 1
                end_word_ngram = 3
            elif config == 8:
                start_word_ngram = 1
                end_word_ngram = 4

        clfs = build_classifiers(config=config)

        for label in labels:
            print("Current output:", label, "\n")

            fout.write("Current output:" + label + "\n")

            print("Classifier\tAccuracy\tAccuracy Std\tMacro F-Score\tMacro F-Score Std\tWeighted F-Score\tWeighted F-Score Std\tVal Time\n")
            fout.write("Classifier\tAccuracy\tAccuracy Std\tMacro F-Score\tMacro F-Score Std\tWeighted F-Score\tWeighted F-Score Std\tVal Time\n\n")

            for clf_name, clf in clfs.items():
                print(clf_name + "\t" + "", end='')

                fout.write(clf_name + "\t")

                y = df_notnull[label].iloc[all_train_indices].values

                val_res = validate_data(clf, y, config, start_word_ngram, end_word_ngram)
                print(val_res)

                fout.write(val_res + "\n")

            print("------------------------------------------------\n")
            fout.write("------------------------------------------------\n\n")

        print("##############################################\n")
        fout.write("##############################################\n\n")
