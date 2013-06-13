#!/usr/bin/python

#import numpy as np
#import types

class NaiveBayesHelper:

    def __init__(self, model):
        self.model = model

    def inspect(self, feature, top=10):
        p0 = sorted(self.model[0][feature], key=self.model[0][feature].get, reverse=True)
        p1 = sorted(self.model[1][feature], key=self.model[1][feature].get, reverse=True)

        for i in range(0, min(len(p0), len(p1), top)):
            print feature, " y0: ", p0[i], self.model[0][feature][p0[i]], " y1:", p1[i], self.model[1][feature][p1[i]]

    def inspectRaw(self, feature, top=10):
        p0 = sorted(self.model['c0'][feature], key=self.model['c0'][feature].get, reverse=True)
        p1 = sorted(self.model['c1'][feature], key=self.model['c1'][feature].get, reverse=True)

        for i in range(0, min(len(p0), len(p1), top)):
            print feature, " y0: ", p0[i], self.model['c0'][feature][p0[i]], " y1:", p1[i], self.model['c1'][feature][p1[i]]
