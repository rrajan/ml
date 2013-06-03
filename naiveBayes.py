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

    def score(self, true_labels, pos_label=1):
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, self.predictions, pos_label)
        auc = metrics.auc(fpr,tpr)

        self.metrics['auc'] = auc
        self.metrics['fpr'] = fpr
        self.metrics['tpr'] = tpr

    def predict(self, fname, h_offset=0):

        f = open(fname, 'r')
        line = f.readline()
        f.close()
        headers = line.strip().split(",")

        X = headers[h_offset:]

        mat = np.genfromtxt(fname, delimiter=",", skip_header=1, dtype=types.IntType, comments=types.NoneType)

        r,c = mat.shape

        p_y = self.model['y1'] # prior
        pred = np.zeros(r)
        for i in range(0,r):
            y1=p_y
            y0=1.0 - p_y
            for j in range(h_offset,c):
                if (mat[i,j] in self.model[1][X[j-h_offset]]):
                    y1 *= self.model[1][X[j-h_offset]][mat[i,j]]
                else:
                    y1 *= self.model[1][X[j-h_offset]][self.default]

                if (mat[i,j] in self.model[0][X[j-h_offset]]):
                    y0 *= self.model[0][X[j-h_offset]][mat[i,j]]
                else:
                    y0 *= self.model[0][X[j-h_offset]][self.default]

            pred[i] = y1 / (y1 + y0)

        self.predictions = pred

    def caliberate(self, fname, L=1):

        f = open(fname, 'r')
        line = f.readline()
        f.close()
        headers = line.strip().split(",")

        Y = headers[0]
        X = headers[1:]

        mat = np.genfromtxt(fname, delimiter=",", skip_header=1, dtype=types.IntType, comments=types.NoneType)

        r,c = mat.shape

        y = mat[:, 0] == 1

        # get counts
        x_y1_dict = {}
        x_y0_dict = {}
        for j in range(1,c):
            kv_1 = {}
            kv_0 = {}
            kv_1[self.default] = 1.0 * L
            kv_0[self.default] = 1.0 * L
            for i in range(0,r):
                if (y[i]):
                    if (mat[i,j] in kv_1):
                        kv_1[mat[i,j]] += 1
                    else:
                        kv_1[mat[i,j]] = 1.0 * L
                else:
                    if (mat[i,j] in kv_0):
                        kv_0[mat[i,j]] += 1
                    else:
                        kv_0[mat[i,j]] = 1.0 * L
            x_y1_dict[X[j-1]] = kv_1
            x_y0_dict[X[j-1]] = kv_0
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

        self.model['y'] = y
        self.model['y_name'] = Y
        self.model['x_names'] = X
        self.model['y1'] = np.float64(sum(y)) / np.float64(len(y))
        self.model[1] = p_x_y1
        self.model[0] = p_x_y0
