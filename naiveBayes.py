#!/usr/bin/python

from collections import OrderedDict
import numpy as np
import types
from sklearn import metrics

## This as of now serves only for Binary classification
## X is treated as a Bernoulli variable (categorical)
class NaiveBayes:

    def __init__(self):
        self.default = 'DEFAULT'
        self.model={}
        self.predictions=0
        self.metrics={}
        self.dKL = None # Kullback-Lebler
        self.dJS = None # Jensen-Shannon

    def getTable(self, fname, c_offset=0, dtype=types.IntType):
        f = open(fname, 'r')
        line = f.readline()
        f.close()
        headers = line.strip().split(",")

        Y = headers[c_offset]
        X = headers[(c_offset+1):]

        mat = np.genfromtxt(fname, delimiter=",", skip_header=1, dtype=dtype, comments=types.NoneType)
        y = mat[:, c_offset]
        return Y,X,y,mat[:,(c_offset+1):]


    def score(self, true_labels, pos_label=1):
        self.metrics.clear()

        fpr, tpr, thresholds = metrics.roc_curve(true_labels, self.predictions, pos_label)
        auc = metrics.auc(fpr,tpr)

        self.metrics['auc'] = auc
        self.metrics['fpr'] = fpr
        self.metrics['tpr'] = tpr

    def measureDKL(self):
        """
        measure kullback_leibler divergence from uniform distribution
        """
        self.dKL = {}
        X = self.model['x_names']
        for i in range(2):
            self.dKL[i] = {}
            for j in range(len(self.model[i])):
                n = len(self.model[i][X[j]])
                unif = np.zeros(n) + 1.0 / n
                #dkl = np.dot(np.log(unif / self.model[i][X[j]].values()), unif)
                p = self.model[i][X[j]].values()
                dkl = np.dot(np.log(p/ unif), p)
                self.dKL[i][X[j]] = dkl
            self.dKL[i] = OrderedDict(sorted(self.dKL[i].items(), key=lambda x:-x[1]))

    '''
    def compareDiv(self):
        """
        measure kullback_leibler divergence between labels
        """
        self.dKL = {}
        X = self.model['x_names']
        for j in range(len(self.model[0])):

            #k_i = list(set(self.model[0][X[j]].keys()) & set(self.model[1][X[j]].keys()))
            k_i = list(set(self.model[0][X[j]].keys()) | set(self.model[1][X[j]].keys()))

            # TODO normalize probs to 1.0
            n = len(k_i)
            rat = np.zeros(n) + 1.0 / self.model[1][X[j]][self.default]
            ref = np.zeros(n) + self.model[0][X[j]][self.default]

            for i, k in enumerate(k_i):
                if (k in self.model[0][X[j]]):
                    ref[i] = self.model[0][X[j]][k]
                if (k in self.model[1][X[j]]):
                    rat[i] = ref[i] / self.model[1][X[j]][k]
                else:
                    rat[i] *= ref[i] # == default

            dkl = np.dot(np.log(rat), ref)
            self.dKL[X[j]] = dkl

        #self.dKL = sorted(self.dKL.iteritems(), key=lambda item: -item[1])
        #self.dKL = sorted(self.dKL, key=self.dKL.get, reverse=True)
        self.dKL = OrderedDict(sorted(self.dKL.items(), key=lambda x:-x[1]))
    '''
    def compareDiv(self, X=None):
        """
        measure Shannon-Jensen divergence between labels
        """
        self.dJS = {}
        if (X == None):
            X = self.model['x_names']
        for x_j in X:
            # get union of keys from both classes
            k_i = list(set(self.model[0][x_j].keys()) | set(self.model[1][x_j].keys()))

            n = len(k_i)
            P = np.zeros(n) + self.model[0][x_j][self.default]
            Q = np.zeros(n) + self.model[1][x_j][self.default]
            M = np.zeros(n) + (self.model[0][x_j][self.default] + self.model[0][x_j][self.default]) * 0.5

            for i, k in enumerate(k_i):
                if (k in self.model[0][x_j]):
                    P[i] = self.model[0][x_j][k]
                    M[i] = P[i] * 0.5
                if (k in self.model[1][x_j]):
                    Q[i] = self.model[1][x_j][k]
                    M[i] += Q[i] * 0.5
                else:
                    M[i] += self.model[0][x_j][self.default] * 0.5

            # normalize probs to 1.0 -- since both classes may not have the same domain
            P = P / np.sum(P)
            Q = Q / np.sum(Q)
            M = M / np.sum(M)
            self.dJS[x_j] = 0.5 * ( np.dot(np.log2(P), P) + np.dot(np.log2(Q), Q) ) - np.dot(np.log2(M),M)

        self.dJS = OrderedDict(sorted(self.dJS.items(), key=lambda x:-x[1]))

    def predict(self, mat, X=None):
        r,c = mat.shape
        p_y = self.model['y1'] # prior
        self.predictions = np.zeros(r)
        if (X == None):
            X = self.model['x_names']
        else:
            """ select columns """
            cols = []
            x_orig = list(self.model['x_names'])
            for j in range(0, len(X)):
                cols.append(x_orig.index(X[j]))
            mat = mat[:, cols]
        for i in range(0,r):
            y1=p_y
            y0=1.0 - p_y
            for j in range(0,len(X)):
                if (mat[i,j] in self.model[1][X[j]]):
                    y1 *= self.model[1][X[j]][mat[i,j]]
                else:
                    y1 *= self.model[1][X[j]][self.default]

                if (mat[i,j] in self.model[0][X[j]]):
                    y0 *= self.model[0][X[j]][mat[i,j]]
                else:
                    y0 *= self.model[0][X[j]][self.default]

            den = y1 + y0
            if (den > 0.0):
                self.predictions[i] = y1 / den
            else:
                self.predictions[i] = 0.0

    def caliberate(self, Y, X, y, mat, L=1):
        self.model.clear()
        r,c = mat.shape
        # get counts
        x_y1_dict = {}
        x_y0_dict = {}
        for j in range(0,c):
            kv_1 = {}
            kv_0 = {}
            for i in range(0,r):
                if (y[i]):
                    if (mat[i,j] in kv_1):
                        kv_1[mat[i,j]] += 1
                    else:
                        kv_1[mat[i,j]] = L + 1
                else:
                    if (mat[i,j] in kv_0):
                        kv_0[mat[i,j]] += 1
                    else:
                        kv_0[mat[i,j]] = L + 1
            kv_1[self.default] = L # + np.mean(kv_1.values()) # np.mean
            kv_0[self.default] = L # + np.mean(kv_0.values()) # np.mean
            x_y1_dict[X[j]] = kv_1
            x_y0_dict[X[j]] = kv_0
            # since idx 0 is Y

        # get probabilities
        p_x_y1 = {}
        p_x_y0 = {}
        for k1 in x_y1_dict.keys():
            p_x_y1[k1] = {}
            T = np.float64(sum(x_y1_dict[k1].values()))
            for k in x_y1_dict[k1].keys():
                p_x_y1[k1][k] = np.float64(x_y1_dict[k1][k]) / T

        for k0 in x_y0_dict.keys():
            p_x_y0[k0] = {}
            T = np.float64(sum(x_y0_dict[k0].values()))
            for k in x_y0_dict[k0].keys():
                p_x_y0[k0][k] = np.float64(x_y0_dict[k0][k]) / T

        ## save

        self.model[Y] = y
        self.model['y_name'] = list(Y)
        self.model['x_names'] = list(X)
        self.model['y1'] = np.float64(sum(y)) / np.float64(len(y))
        self.model[0] = p_x_y0
        self.model[1] = p_x_y1
        self.model['c0'] = x_y0_dict
        self.model['c1'] = x_y1_dict
        self.model[0] = p_x_y0
