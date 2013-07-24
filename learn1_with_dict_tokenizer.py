#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First simple sklearn classifier"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import os
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib import pyplot as plt
from matplotlib import cm
from nltk.corpus import stopwords
import unicodecsv
import sql_convenience
from tweet_tokenizer import TweetTokenizer

############
# NOTE
# this is a basic LogisticRegression classifier, using 5-fold cross validation
# and a cross entropy error measure (which should nicely fit this binary
# decision classification problem).
# do not trust this code to do anything useful in the real world!
############


def reader(class_name):
    class_reader = unicodecsv.reader(open(class_name), encoding='utf-8')
    row0 = next(class_reader)
    assert row0 == ["tweet_id", "tweet_text"]
    lines = []
    for tweet_id, tweet_text in class_reader:
        txt = tweet_text.strip()
        if len(txt) > 0:
            lines.append(txt)
    return lines


def label_learned_set(vectorizer, clfl, threshold, validation_table):
    for row in sql_convenience.extract_classifications_and_tweets(validation_table):
        cls, tweet_id, tweet_text = row
        #spd = vectorizer.transform([tweet_text]).todense()
        spd = vectorizer.transform(splitForDictVectorizer([tweet_text])).todense()

        predicted_cls = clfl.predict(spd)
        predicted_class = predicted_cls[0]  # turn 1D array of 1 item into 1 item
        predicted_proba = clfl.predict_proba(spd)[0][predicted_class]
        if predicted_proba < threshold and predicted_class == 1:
            predicted_class = 0  # force to out-of-class if we don't trust our answer
        sql_convenience.update_class(tweet_id, validation_table, predicted_class)

def splitForDictVectorizer(arr):
    return [Counter(s.split()) for s in arr]

def check_classification(vectorizer, clfl):
    #spd0 = vectorizer.transform([u'really enjoying how the apple\'s iphone makes my ipad look small']).todense()
    spd0 = vectorizer.transform(splitForDictVectorizer([u'really enjoying how the apple\'s iphone makes my ipad look small'])).todense()
    print("1?", clfl.predict(spd0), clfl.predict_proba(spd0))  # -> 1 which is set 1 (is brand)
    spd1 = vectorizer.transform(splitForDictVectorizer([u'i like my apple, eating it makes me happy'])).todense()
    print("0?", clfl.predict(spd1), clfl.predict_proba(spd1))  # -> 0 which is set 0 (not brand)


def cross_entropy_error(Y, probas_):
    # compute Cross Entropy using the Natural Log:
    # ( -tln(y) ) − ( (1−t)ln(1−y) )
    probas_class1 = probas_[:, 1]  # get the class 1 probabilities
    cross_entropy_errors = ((-Y) * (np.log(probas_class1))) - ((1 - Y) * (np.log(1 - probas_class1)))
    return cross_entropy_errors


def show_cross_validation_errors(cross_entropy_errors_by_fold):
    print("Cross validation cross entropy errors:" + str(cross_entropy_errors_by_fold))
    print("Cross entropy (lower is better): %0.3f (+/- %0.3f)" % (cross_entropy_errors_by_fold.mean(), cross_entropy_errors_by_fold.std() / 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple sklearn implementation, example usage "learn1.py scikit_testtrain_apple --validation_table=learn1_validation_apple"')
    parser.add_argument('table', help='Name of in and out of class data to read (e.g. scikit_validation_app)')
    parser.add_argument('--validation_table', help='Table of validation data - get tweets and write predicted class labels back (e.g. learn1_validation_apple)')
    parser.add_argument('--roc', default=False, action="store_true", help='Plot a Receiver Operating Characterics graph for the learning results')
    parser.add_argument('--pr', default=False, action="store_true", help='Plot a Precision/Recall graph for the learning results')
    parser.add_argument('--termmatrix', default=False, action="store_true", help='Draw a 2D matrix of tokens vs binary presence (or absence) using all training documents')
    args = parser.parse_args()

    data_dir = "data"
    in_class_name = os.path.join(data_dir, args.table + '_in_class.csv')
    out_class_name = os.path.join(data_dir, args.table + '_out_class.csv')

    in_class_lines = reader(in_class_name)
    out_class_lines = reader(out_class_name)

    # put all items into the training set
    train_set = out_class_lines + in_class_lines
    target = np.array([0] * len(out_class_lines) + [1] * len(in_class_lines))

    # choose a vectorizer to turn the tokens in tweets into a matrix of
    # examples (we can plot this further below using --termmatrix)
    stopWords = stopwords.words('english')
    MIN_DF = 2
    vectorizer_binary = CountVectorizer(stop_words=stopWords, min_df=MIN_DF, binary=True)
    vectorizer_tfidf = TfidfVectorizer(stop_words=stopWords, min_df=MIN_DF)
    vectorizer_dict = TweetTokenizer("Apple")
    #vectorizer = vectorizer_tfidf
    #vectorizer = vectorizer_binary
    vectorizer = vectorizer_dict
    
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    print("Feature names (first 20):", vectorizer.get_feature_names()[:20], "...")
    print("Vectorized %d features" % (len(vectorizer.get_feature_names())))

    clf = linear_model.LogisticRegression()

    kf = cross_validation.KFold(n=len(target), n_folds=5, shuffle=True)
    # using a score isn't so helpful here (I think) as I want to know the
    # distance from the desired categories and a >0.5 threshold isn't
    # necessaryily the right thing to measure (I care about precision when
    # classifying, not recall, so the threshold matters)
    #cross_val_scores = cross_validation.cross_val_score(clf, trainVectorizerArray, target, cv=kf, n_jobs=-1)
    #print("Cross validation in/out of class test scores:" + str(cross_val_scores))
    #print("Accuracy: %0.3f (+/- %0.3f)" % (cross_val_scores.mean(), cross_val_scores.std() / 2))

    # try the idea of calculating a cross entropy score per fold
    cross_entropy_errors_test_by_fold = np.zeros(len(kf))
    cross_entropy_errors_train_by_fold = np.zeros(len(kf))
    for i, (train_rows, test_rows) in enumerate(kf):
        Y_train = target[train_rows]
        X_train = trainVectorizerArray[train_rows]
        X_test = trainVectorizerArray[test_rows]
        probas_test_ = clf.fit(X_train, Y_train).predict_proba(X_test)
        probas_train_ = clf.fit(X_train, Y_train).predict_proba(X_train)
        # compute cross entropy for all trained and tested items in this fold
        Y_test = target[test_rows]

        cross_entropy_errors_test = cross_entropy_error(Y_test, probas_test_)
        cross_entropy_errors_train = cross_entropy_error(Y_train, probas_train_)
        cross_entropy_errors_test_by_fold[i] = np.average(cross_entropy_errors_test)
        cross_entropy_errors_train_by_fold[i] = np.average(cross_entropy_errors_train)
    #import pdb; pdb.set_trace()
    print("Training:")
    show_cross_validation_errors(cross_entropy_errors_train_by_fold)
    print("Testing:")
    show_cross_validation_errors(cross_entropy_errors_test_by_fold)

    if args.termmatrix:
        fig = plt.figure()
        # to plot the word vector on the training data use:
        plt.title("{} matrix of features per sample for {}".format(str(vectorizer.__class__).split('.')[-1][:-2], args.table))
        plt.imshow(trainVectorizerArray, cmap=cm.gray, interpolation='nearest', origin='lower')
        nbr_features = trainVectorizerArray.shape[1]
        plt.xlabel("{} Features".format(nbr_features))
        last_class_0_index = len(out_class_lines) - 1
        plt.ylabel("Samples (Class 0: 0-{}, Class 1: {}-{})".format(last_class_0_index, last_class_0_index + 1, trainVectorizerArray.shape[0] - 1))
        plt.hlines([last_class_0_index], 0, nbr_features, colors='r', alpha=0.8)
        plt.show()

    # plot a Receiver Operating Characteristics plot from the cross validation
    # sets
    if args.roc:
        fig = plt.figure()
        for i, (train, test) in enumerate(kf):
            probas_ = clf.fit(trainVectorizerArray[train], target[train]).predict_proba(trainVectorizerArray[test])
            fpr, tpr, thresholds = roc_curve(target[test], probas_[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristics')  # , Mean ROC (area = %0.2f)' % (mean_auc))
        plt.legend(loc="lower right")
        plt.show()

    # plot a Precision/Recall line chart from the cross validation sets
    if args.pr:
        fig = plt.figure()
        for i, (train, test) in enumerate(kf):
            probas_ = clf.fit(trainVectorizerArray[train], target[train]).predict_proba(trainVectorizerArray[test])
            precision, recall, thresholds = precision_recall_curve(target[test], probas_[:, 1])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label='Precision-Recall curve %d (area = %0.2f)' % (i, pr_auc))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.title('Precision-Recall curves')
        plt.legend(loc="lower left")
        plt.show()

    # write validation results to specified table
    if args.validation_table:
        # make sparse training set using all of the test/train data (combined into
        # one set)
        train_set_sparse = vectorizer.transform(train_set)
        # instantiate a local classifier
        clfl = clf.fit(train_set_sparse.todense(), target)

        # check and print out two classifications as sanity checks
        check_classification(vectorizer, clfl)
        # use a threshold (arbitrarily chosen at present), test against the
        # validation set and write classifications to DB for reporting
        chosen_threshold = 0.92
        label_learned_set(vectorizer, clfl, chosen_threshold, args.validation_table)
