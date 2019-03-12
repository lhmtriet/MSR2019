# Import libraries
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Load datasets
df = pd.read_csv('Data/all_cves_cleaned.csv')

# Extract year from ID
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

def extract_features(config, start_word_ngram, end_word_ngram):
    if config == 1:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=False, min_df=0.01, max_df=1.0,
                               norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')
    elif config == 2:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=True, min_df=0.01,
                               norm='l2', token_pattern=r'\S*[A-Za-z]\S+')
    elif config < 6:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=False,
                               min_df=0.01, norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')

    return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=True,
                           min_df=0.01, norm='l2', token_pattern=r'\S*[A-Za-z]\S+')

# Train model

def feature_model(X_train, X_test, config, start_word_ngram, end_word_ngram):

    # Generate the vectorizer that matches the current config
    word_vectorizer = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram)

    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    X_train_transformed = word_vectorizer.fit_transform(X_train)
    X_test_transformed = word_vectorizer.transform(X_test)

    X_train_transformed = X_train_transformed.astype(np.float64)
    X_test_transformed = X_test_transformed.astype(np.float64)

    return X_train_transformed, X_test_transformed

# Test the data

def evaluate(clf, X_train_transformed, X_test_transformed, y):

    y_train, y_test = y[train_indices], y[test_indices]

    # Training
    t_start = time.clock()
    clf.fit(X_train_transformed, y_train)
    train_time = time.clock() - t_start

    # Prediction
    p_start = time.clock()
    y_pred = clf.predict(X_test_transformed)
    pred_time = time.clock() - p_start

    return "{:.3f}".format(accuracy_score(y_test, y_pred)) + "\t" + "{:.3f}".format(
        f1_score(y_test, y_pred, average='macro')) + "\t" + "{:.3f}".format(
        f1_score(y_test, y_pred, average='weighted')) + "\t" + "{:.3f}".format(train_time) + "\t" + "{:.3f}".format(pred_time)


labels = ['CVSS2_Conf', 'CVSS2_Integrity', 'CVSS2_Avail', 'CVSS2_AccessVect',
          'CVSS2_AccessComp', 'CVSS2_Auth', 'CVSS2_Severity']

# Iterate over each label and corresponding optimal classifier and NLP representation for testing
for label in labels:
    print("Current output:", label, "\n")

    config = 1

    print("Bag-of-word without tf-idf")
    start_word_ngram = 1
    end_word_ngram = 1

    clf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1, max_features=40)

    t_time = time.clock()
    X_train_transformed, X_test_transformed = feature_model(X_train, X_test, config,
                                                                            start_word_ngram, end_word_ngram)
    print("Building vocab time:", time.clock() - t_time)

    print("Classifier\tAccuracy\tMacro F-Score\tWeighted F-Score\tTrain time\tPredict time\n")

    print("Random Forest\t")

    y = df_notnull[label].values

    print(evaluate(clf, X_train_transformed, X_test_transformed, y))

    print("##############################################")
