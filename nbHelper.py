#!/usr/bin/python

import numpy as np

class NaiveBayesHelper:

    def __init__(self, model=None):
        self.model = model

    def inspect(self, feature, top=10):
        p0 = sorted(self.model[0][feature], key=self.model[0][feature].get, reverse=True)
        p1 = sorted(self.model[1][feature], key=self.model[1][feature].get, reverse=True)

        print "features y0: %d, y1: %d" % (len(p0),len(p1))
        for i in range(0, min(len(p0), len(p1), top)):
            print feature, " y0: ", p0[i], self.model[0][feature][p0[i]], " y1:", p1[i], self.model[1][feature][p1[i]]

    def inspectRaw(self, feature, top=10):
        p0 = sorted(self.model['c0'][feature], key=self.model['c0'][feature].get, reverse=True)
        p1 = sorted(self.model['c1'][feature], key=self.model['c1'][feature].get, reverse=True)

        print "features y0: %d, y1: %d" % (len(p0),len(p1))
        for i in range(0, min(len(p0), len(p1), top)):
            print feature, " y0: ", p0[i], self.model['c0'][feature][p0[i]], " y1:", p1[i], self.model['c1'][feature][p1[i]]

    # Utility functions for higher order data
    def doubs_generator(self, n):
        """
        Generate doubles (i,j) with i<j from 0 to n
        """
        for i in range(n):
            for j in range(i+1,n):
                yield (i,j)

    def trips_generator(self, n):
        """
        Generate triples (i,j,k) with i<j<k from 0 to n
        """
        for i,j in self.doubs_generator(n):
            for k in range(j+1,n):
                yield (i,j,k)

    def group_data(self, data, hash=hash, order=2):
        """
        numpy.array -> numpy.array

        Groups all columns of data into all combinations
        """
        generator = self.doubs_generator
        if (2 < order):
            generator = self.trips_generator

        new_data = []
        m,n = data.shape
        for indices in generator(n):
            new_data.append([hash(tuple(v)) for v in data[:,indices]])
        return np.array(new_data).T

    def group_names(self, data, hash=hash, order=2):
        """
        numpy.array -> numpy.array

        Groups all columns of data into all combinations
        """
        generator = self.doubs_generator
        if (2 < order):
            generator = self.trips_generator

        new_data = []
        m,n = data.shape
        for indices in generator(n):
            for i, v in enumerate(data[:,indices]):
                if (i > 0):
                    break # hack
                s = []
                for r in tuple(v):
                    s.append(str(r))
                new_data.append('-'.join(s))
        return np.array(new_data).T
