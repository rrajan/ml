import numpy as np
from env.plot import *

class CAPlot:

    def __init__(self, ca):
        self.ca = ca

    def plot(self, pdata, d=2, factors=np.array([0]), r=1, labels="", title="CA"):

        for i,f in enumerate(factors):

            x = np.array(range(len(pdata))) * r
            if (2 == d):
                x = pdata[:,f]
                y = pdata[:,(f+1)]
            else:
                y = pdata[:,f]

            plt.figure(i+1)
            plt.plot(x, y, 'ro')

            if (len(labels) > 0 and len(labels) == len(x)):
                for i, l in enumerate(labels):
                    plt.annotate(labels[i], xy=(x[i], y[i]))
            else:
                print "plot: error in len(labels)"

            plt.title(title + " factor " + `f`)

        plt.show()

    def plotHist(self, pdata, bins=50, wf=1.0, title="Hist"):
        h, b = np.histogram(pdata, bins)
        w = (b[1] - b[0]) * wf
        cen = (b[:-1] + b[1:])/2.0
        plt.bar(cen, h, align='center', width=w)
        plt.title(title)
        plt.show()

    def plotCols(self, d=2, factors=np.array([0]), r=1, labels="", title="Cols"):
        self.plot(self.ca.Fc, d, factors, r, labels, title)

    def plotRows(self, d=2, factors=np.array([0]), r=1, labels="", title="Rows"):
        self.plot(self.ca.Fr, d, factors, r, labels, title)

    def plotHistCols(self, bins=50, wf=1.0, title="ColumnScores"):
        self.plotHist(self.ca.c * self.ca.N, bins, wf, title)

    def plotHistRows(self, bins=50, wf=1.0, title="RowScores"):
        self.plotHist(self.ca.r * self.ca.N, bins, wf, title)


class CAClusters:

    def __init__(self, ca):
        self.ca = ca
        self.Dc = 0

    def calcDistance(self, F, dim=1):
        y,x = F.shape
        dim = min(dim, x + 1)
        D = np.array(F * 0, dtype=np.float32)

        F2 = F[:, np.array(range(dim))]
        for i in range(y):
            m = F2[i,:]
            d2 = np.power(F2 - m, 2)
            D[:, i] = np.power(np.sum(d2,1), 0.5)

        return D

    def calcColDistance(self, dim=1):
        self.Dc = self.calcDistance(self.ca.Fc, dim)

    def plot(self, pdata, factors=np.array([0]), r=1, labels="", title="Distance"):

        for i,f in enumerate(factors):

            x = np.array(range(len(pdata))) * r
            y = pdata[:,f]

            plt.figure(i+1)
            plt.plot(x, y, 'go')

            if (len(labels) > 0 and len(labels) == len(x)):
                for i, l in enumerate(labels):
                    plt.annotate(labels[i], xy=(x[i], y[i]))
            else:
                print "plot: error in len(labels)"

            plt.title(title + " from " + labels[f])

        plt.show()

    def plotColDistance(self, factors=np.array([0]), r=1, labels="", title="Cols"):
        self.plot(self.Dc, factors, r, labels, title)

    def sortedNames(self, names, axis=1):
        if (1 == axis):
            if (len(names) != len(self.Dc)):
                print "sortedNames: error len(names)"
                return 0
            return names[np.argsort(self.Dc,0)]
        else:
            print "sortedNames: only columns supported, re-run with axis=1"
            return 0
