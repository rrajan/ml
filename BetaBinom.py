#!/usr/bin/python

import numpy as np
from scipy.stats import beta
from scipy.stats import norm
import matplotlib.pyplot as plt

class BetaBinom:

    def __init__(self):
        self.DEBUG = False
        self.a = 1.0
        self.b = 1.0
        self.rv = beta(self.a, self.b)
        self.dkl = [0]
        self.djs = [0]
        self.x = 0
        self.y = 0
        self.dx = 0.01
        self.xn = 100

        # init
        self.x = np.zeros(shape=(self.xn+1,), dtype=np.float) + self.dx
        self.x = np.cumsum(self.x) - self.dx
        self.x = np.minimum(np.maximum(np.cumsum(self.x) - self.dx,0.0),1.0)
        self.y = self.rv.pdf(self.x)

    def rescale(self, N):
        self.xn = N
        self.dx = 1.0 / self.xn
        self.x = np.zeros(shape=(N+1,), dtype=np.float) + self.dx
        self.x = np.minimum(np.maximum(np.cumsum(self.x) - self.dx,0.0),1.0)
        self.y = self.rv.pdf(self.x)

    def normalApprox(self, a, b):
        if (self.DEBUG):
            print "using normal approx"
        m = a * 1.0 / (a + b)
        v = (a * b * 1.0) / ((a + b) ** 3) # ignoring 1, such large numbers can ignore that
        self.rv = norm(loc=m, scale=np.sqrt(v))
        y = self.rv.pdf(self.x)
        return y

    def setNewEvidence(self, pos, tot):

        a = np.sum(pos)
        b = np.sum(tot) - a
        a_new = self.a + a
        b_new = self.b + b

        # get new PDF
        self.rescale((a_new + b_new)*1.0) # some multiplicative factor

        y_new = np.zeros(shape=(len(self.y),),dtype=np.float)
        ## use normal approximation for large a and b, unfortunately we reach large a and b very quickly
        if (a_new + b_new > 1000):
            y_new = self.normalApprox(a_new, b_new)
        else:
            self.rv = beta(a_new, b_new)
            y_new = self.rv.pdf(self.x)
        ## just incase something messes up
        if (any(np.isnan(y_new))):
            y_new = self.normalApprox(a_new, b_new)

        # measure dKL and dJS before update
        self.measureDKL(y_new)
        self.measureDJS(y_new)

        # update
        self.a = a_new
        self.b = b_new
        self.y = y_new

    def measureDKL(self, y):
        y = y / np.sum(y)
        yprev = self.y / np.sum(self.y)

        logyprev = np.zeros(shape=(len(yprev),),dtype=np.float)
        iprev = np.where(yprev > 0.0)[0]
        logyprev[iprev] = np.log(yprev[iprev])

        logy = np.zeros(shape=(len(y),),dtype=np.float)
        i = np.where(y > 0.0)[0]
        logy[i] = np.log(y[i])

        logyprev[np.isinf(logyprev)] = 0.0
        logy[np.isinf(logy)] = 0.0

        self.dkl.append( np.dot(y, logy - logyprev) )

    def measureDJS(self, y):
        y = y / np.sum(y)
        yprev = self.y / np.sum(self.y)

        logyprev = np.zeros(shape=(len(yprev),),dtype=np.float)
        iprev = np.where(yprev > 0.0)[0]
        logyprev[iprev] = np.log2(yprev[iprev])

        logy = np.zeros(shape=(len(y),),dtype=np.float)
        i = np.where(y > 0.0)[0]
        logy[i] = np.log2(y[i])

        m = (y + yprev) / 2.0
        im = np.where(m > 0.0)[0]
        logm = np.zeros(shape=(len(y),),dtype=np.float)
        logm[im] = np.log2(m[im])

        logyprev[np.isinf(logyprev)] = 0.0
        logy[np.isinf(logy)] = 0.0
        logm[np.isinf(logm)] = 0.0

        #self.djs.append( 0.5 * (np.dot(yprev, (logyprev - logm)) + np.dot(y, (logy - logm)) ) )
        self.djs.append( 0.5 * (np.dot(yprev, logyprev) + np.dot(y, logy)) - np.dot(m,logm) )

    def getMean(self):
        return self.a / (self.a + self.b)

    def getMode(self):
        return (self.a - 1) / (self.a + self.b - 2)

    def predictMAP(self, tot):
        return self.getMode() * tot

    def predictFreq(self, tot):
        return self.getMean() * tot

    def predict(self, tot):
        return np.dot( np.outer(tot, self.x), self.y) * self.dx

    def plotPdf(self, show=True):
        plt.plot(self.x, self.y)
        plt.title('PDF')
        if (show):
            plt.show()

    def plotDKL(self, show=True):
        plt.plot(self.dkl)
        plt.plot([0] * len(self.dkl), 'r')
        plt.title('dKL')
        if (show):
            plt.show()

    def plotDJS(self, show=True):
        plt.plot(self.djs)
        plt.plot([0] * len(self.djs), 'r')
        plt.title('dJS')
        if (show):
            plt.show()

    def plotShow(self):
        plt.show()
