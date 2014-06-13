#!/usr/bin/python

import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm

# One v/s All Classification
# Logistic, SVM
class Solver:
    def __init__(self, cache_size=1000):
        self.model = list()
        self.predictions = 0
        self.prob = 0
        self.acc = list()
        self.C = 1.0
        self.gamma = 0.1
        self.tol = 0.0001
        self.cache_size = cache_size

    def train(self, trainMat, labels, model="LR"):
        print "Traing with model", model
        del self.model[:]
        s = labels.shape
        if (len(s) > 1):
            for i in range(s[1]):
                if (model == "SVM"):
                    clf = svm.SVC(C=self.C, gamma=self.gamma, cache_size=self.cache_size)
                else:
                    clf = LR(C=self.C, penalty='l2', tol=self.tol)
                clf.fit(trainMat, labels[:,i]) # lookup clf.coef_
                self.model.append(clf)
        else:
            if (model == "SVM"):
                clf = svm.SVC(C=self.C, gamma=self.gamma, cache_size=self.cache_size)
            else:
                clf = LR(C=self.C, penalty='l2', tol=self.tol)
            clf.fit(trainMat, labels) # lookup clf.coef_
            self.model.append(clf)

    def predict(self, testMat):
        l = len(self.model)
        preds = list()
        for i in range(l):
            pred = self.model[i].predict(testMat)
            preds.append(pred)
        self.predictions = (np.array(preds, dtype=np.int)).T

    def predict_proba(self, testMat):
        l = len(self.model)
        preds = list()
        for i in range(l):
            pred = self.model[i].predict_proba(testMat)
            preds.append(pred)
        self.prob = (np.array(preds, dtype=np.int)).T

    def validate(self, reference):
        del self.acc[:]
        s = reference.shape
        if (len(s) > 1):
            for i in range(s[1]):
                a = self.predictions[:,i] == reference[:, i]
                a = float(np.sum(a)) / s[0]
                self.acc.append(a)
            # do all
            combined = (self.predictions == reference)
            a_c_rows = np.prod(combined, axis=1)
            a_c = float(np.sum(a_c_rows)) / s[0]
            self.acc.append(a_c)
        else:
            a = self.predictions.T == reference
            a = float(np.sum(a)) / s[0]
            self.acc.append(a)

