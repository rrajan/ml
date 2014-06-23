import numpy as np
import scipy as sp
from scipy import stats

class CorrespondenceAnalysis:

    def __init__(self, ct=0, rt=0, cit=0, rit=0):
        self.min        = 0.00000001
        self.N          = 0
        self.Z          = 0
        self.Chi2Stat   = 0
        self.dof        = 0
        self.pval       = 1.0 # => independence or probab of chi2Stat follows Chi2 Distribution
        self.r          = 0
        self.c          = 0
        self.S          = 0
        self.P          = 0
        self.Q          = 0
        self.E          = 0
        self.W          = 0
        self.Fr         = 0
        self.Fc         = 0
        self.ct         = 0
        self.cit        = 0
        self.rt         = 0
        self.rit        = 0
        self.idxC       = 0
        self.idxR       = 0

        self.setThresholds(ct,rt,cit,rit)

    def setThresholds(self, ct=0, rt=0, cit=0, rit=0):
        self.ct     = ct
        self.cit    = cit
        self.rt     = rt
        self.rit    = rit

    def filter(self, mat):
        d = np.array(mat, dtype=np.float64)

        # first filter Rows
        di = np.zeros(shape=d.shape, dtype=np.int32)
        di[d > 0.0] = 1
        R = np.sum(d, 1)
        RI = np.sum(di, 1)

        self.idxR = np.logical_and(R > self.rt, RI > self.rit)

        # now filter Columns
        dR = d[self.idxR, :]
        diR = di[self.idxR, :]
        C = np.sum(dR, 0)
        CI = np.sum(diR, 0)

        self.idxC = np.logical_and(C > self.ct, CI > self.cit)

        d = dR[:, self.idxC]

        return d

    def chiSqScores(self, mat):
        if (self.ct + self.rt + self.cit + self.rit > 0):
            d = self.filter(mat)
        else:
            d = np.array(mat, dtype=np.float64)

        N = np.sum(d)
        Z = d / N

        self.r = np.sum(Z, 1)
        self.c = np.sum(Z, 0)

        r_mH = np.power(self.r, -0.5)
        c_mH = np.power(self.c, -0.5)

        y,x = Z.shape
        self.dof = (y-1) * (x-1)
        for ix in range(x):
            for iy in range(y):
                Z[iy,ix] = ( Z[iy,ix] - (self.r[iy] * self.c[ix]) ) * (r_mH[iy] * c_mH[ix])

        self.N = N
        self.Z = Z
        Z2 = np.power(Z, 2)
        self.Chi2Stat = np.sum(Z2) * N
        self.pval = sp.stats.chi2.sf(self.Chi2Stat, self.dof) # survival function i.e 1 - CDF

    def analyze(self, mat):

        self.chiSqScores(mat)
        M = self.Z

        self.P, self.S, self.Q = np.linalg.svd(M, False)
        self.Q = np.transpose(self.Q)
        self.E = np.power(self.S, 2)
        self.W = np.cumsum(self.E) / np.sum(self.E)

        DS = np.diag(self.S)

        # Factors
        Dr_mHalf = np.diag(np.power(self.r, -0.5))
        Dc_mHalf = np.diag(np.power(self.c, -0.5))
        self.Fr = np.dot(np.dot(Dr_mHalf, self.P), DS)
        self.Fc = np.dot(np.dot(Dc_mHalf, self.Q), DS)

    def analyzeCols(self, mat):

        self.chiSqScores(mat)
        M = self.Z

        self.P = 0
        _M = np.transpose(M)
        _MM = np.dot(_M,M)
        E, Q = np.linalg.eig(_MM) # not sorted?
        ord_inds = np.argsort(-E) # rev sort
        self.E = E[ord_inds]
        self.Q = Q[:, ord_inds]
        self.S = np.power(np.maximum(self.min, self.E), 0.5) # weird, R is much better (stable) at eigens (& perhaps svd)
        self.W = np.cumsum(self.E) / np.sum(self.E)

        DS = np.diag(self.S)

        # Factors
        Dc_mHalf = np.diag(np.power(self.c, -0.5))
        self.Fc = np.dot(np.dot(Dc_mHalf, self.Q), DS)

    def analyzeRows(self, mat):

        self.chiSqScores(mat)
        M = self.Z

        self.Q = 0
        _M = np.transpose(M)
        M_M = np.dot(M,_M)
        E, P = np.linalg.eig(M_M) # not sorted?
        ord_inds = np.argsort(-E) # rev sort
        self.E = E[ord_inds]
        self.P = P[:, ord_inds]
        self.S = np.power(np.maximum(self.min, self.E), 0.5) # weird, R is much better (stable) at eigens (& perhaps svd)
        self.W = np.cumsum(self.E) / np.sum(self.E)

        DS = np.diag(self.S)

        # Factors
        Dr_mHalf = np.diag(np.power(self.r, -0.5))
        self.Fr = np.dot(np.dot(Dr_mHalf, self.P), DS)

    def calcSimilarity(self, F1, F2=None):
        y,x = F1.shape
        Fnorm1 = np.zeros(x)
        for i in range(x):
            Fnorm1[i] = np.linalg.norm(F1[:,i])

        Fnorm2 = Fnorm1

        if (None == F2):
            F2 = F1
        else:
            y2,x2 = F2.shape
            Fnorm2 = np.zeros(x2)
            for i in range(x2):
                Fnorm2[i] = np.linalg.norm(F2[:,i])

        D = np.outer(Fnorm1, Fnorm2)
        S = np.dot(np.transpose(F1), F2)

        return (S/D)

    def calcColSimilarities(self, dim=0):
        F = self.Fc
        if (dim > 0):
            F = self.Fc[:, :dim]
        return self.calcSimilarity(F.T)

    def calcRowSimilarities(self, dim=0):
        F = self.Fr
        if (dim > 0):
            F = self.Fc[:, :dim]
        return self.calcSimilarity(F.T)

    def calcCrossSimilarities(self, dim=0):
        F1 = self.Fr
        F2 = self.Fc

        if (dim > 0):
            F1 = self.Fr[:, :dim]
            F2 = self.Fc[:, :dim]

        return self.calcSimilarity(F1.T, F2.T)
