#!/usr/bin/python

import numpy as np
import types
from sklearn import metrics

## This as of now serves only for Binary classification
class NaiveBayes:

    def __init__(self):
        self.default = 'DEFAULT'
        self.model={}
        self.predictions=0
        self.metrics={}

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

    def predict(self, Y, X, mat):
        self.predictions = 0
        r,c = mat.shape
        p_y = self.model['y1'] # prior
        pred = np.zeros(r)
        for i in range(0,r):
            y1=p_y
            y0=1.0 - p_y
            for j in range(0,c):
                if (mat[i,j] in self.model[1][X[j]]):
                    y1 *= self.model[1][X[j]][mat[i,j]]
                else:
                    y1 *= self.model[1][X[j]][self.default]

                if (mat[i,j] in self.model[0][X[j]]):
                    y0 *= self.model[0][X[j]][mat[i,j]]
                else:
                    y0 *= self.model[0][X[j]][self.default]

            pred[i] = y1 / (y1 + y0)

        self.predictions = pred

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
        self.model['y_name'] = Y
        self.model['x_names'] = X
        self.model['y1'] = np.float64(sum(y)) / np.float64(len(y))
        self.model[1] = p_x_y1
        self.model[0] = p_x_y0
