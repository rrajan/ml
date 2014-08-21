#!/usr/bin/python

import numpy as np
import pandas as pd

class MarkovTMat:

    # expect sorted df by group AND other values
    def build(self, df, key, group=None, L=1):
        states = df[key].unique()
        states.sort()
        nStates = len(states)
        mat = pd.DataFrame(np.zeros(shape=(nStates, nStates)), columns=states, index=states) + L

        prevS = None
        if (group is not None):
            prevG = None
            for d in df[[group, key]].values:
                curG = d[0]
                curSt = d[1]
                if (prevG == curG): mat.loc[prevS, curSt] += 1
                prevS = curSt
                prevG = curG
        else:
            prevS = df[key][0]
            for curSt in df[key][1:]:
                mat.loc[prevS, curSt] += 1
                prevS = curSt

        return mat

    def rowNorm(self, mat):
        ms = mat.sum(axis=1)
        mat2 = pd.DataFrame(mat, copy=True)
        for ind in ms.index: mat2.loc[ind] /= ms[ind]
        return mat2

    def colNorm(self, mat):
        ms = mat.sum(axis=0)
        mat2 = pd.DataFrame(mat, copy=True)
        for ind in ms.index: mat2[ind] /= ms[ind]
        return mat2
