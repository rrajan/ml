import numpy as np
from env.plot import *

class CAPlot:

    def __init__(self, ca):
        self.ca = ca

    def plot(self, pdata, d=2, factors=np.array([0]), r=1, labels=[], title="CA", marker="ro", show=True):

        for i,f in enumerate(factors):

            x = np.array(range(len(pdata))) * r
            if (2 == d):
                x = pdata[:,f]
                y = pdata[:,(f+1)]
            else:
                y = pdata[:,f]

            plt.figure(i+1)
            plt.plot(x, y, marker)

            if (len(labels) > 0 and len(labels) == len(x)):
                for i, l in enumerate(labels):
                    plt.annotate(labels[i], xy=(x[i], y[i]))
            else:
                print "plot: error in len(labels)"

            plt.title(title + " factor " + `f`)

        plt.grid(True)
        if (show):
            plt.show()
        return plt

    def plotHist(self, pdata, bins=50, wf=1.0, title="Hist", show=True):
        h, b = np.histogram(pdata, bins)
        w = (b[1] - b[0]) * wf
        cen = (b[:-1] + b[1:])/2.0
        plt.bar(cen, h, align='center', width=w)
        plt.title(title)
        plt.grid(True)
        if (show):
            plt.show()

    def getCounts2D(self, x, y, bins=(50,50)):
        h, xedges, yedges = np.histogram2d(x, y, bins)
        cenx = (xedges[:-1] + xedges[1:])/2.0
        ceny = (yedges[:-1] + yedges[1:])/2.0
        count = h.shape[0] * h.shape[1]
        c=0
        nx = np.zeros(count)
        ny = np.zeros(count)
        nc = np.zeros(count)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                nx[c] = cenx[i]
                ny[c] = ceny[j]
                nc[c] = h[i,j]
                c = c + 1

        return nc, nx, ny

    def plotHist2D(self, x, y, bins=(50,50), wf=1.0, title="Hist", show=True, useColor=False, color='k', marker='o', ct=0.0):
        nc, nx, ny = self.getCounts2D(x, y, bins)
        nc = nc * wf
        idx = (np.argwhere(nc > ct)).flatten()
        if (useColor):
            plt.scatter(nx[idx],ny[idx],c=nc[idx], marker=marker, color=color)
            plt.colorbar()
        else:
            plt.scatter(nx[idx],ny[idx],s=nc[idx], marker=marker, color=color)
            self.annotateCounts(nc[idx],nx[idx],ny[idx])
        plt.title(title)
        plt.grid(True)
        if (show):
            plt.show()

    def annotateCounts(self, nc, nx, ny):
        for i in range(len(nc)):
            plt.annotate(`nc[i]`, xy=(nx[i], ny[i]))

    def plotCols(self, d=2, factors=np.array([0]), r=1, labels=[], title="Cols", marker = "ro", show=True):
        return self.plot(self.ca.Fc, d, factors, r, labels, title, marker, show)

    def plotRows(self, d=2, factors=np.array([0]), r=1, labels=[], title="Rows", marker = "go", show=True):
        return self.plot(self.ca.Fr, d, factors, r, labels, title, marker, show)

    def plotHistCols(self, bins=50, wf=1.0, title="ColumnScores"):
        self.plotHist(self.ca.c * self.ca.N, bins, wf, title)

    def plotHistRows(self, bins=50, wf=1.0, title="RowScores"):
        self.plotHist(self.ca.r * self.ca.N, bins, wf, title)

    def plotHist2DCols(self, f=0, bins=(50,50), wf=1.0, title="ColumnScores", show=True, useColor=True, color='k', marker='o', ct=0.0):
        self.plotHist2D(self.ca.Fc[:,f], self.ca.Fc[:, f+1], bins, wf, title, show=show, useColor=useColor, color=color, marker=marker, ct=ct)

    def plotHist2DRows(self, f=0, bins=(50,50), wf=1.0, title="RowScores", show=True, useColor=True, color='k', marker='o', ct=0.0):
        self.plotHist2D(self.ca.Fr[:,f], self.ca.Fr[:,f+1], bins, wf, title, show=show, useColor=useColor, color=color, marker=marker, ct=ct)


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

    def plot(self, pdata, factors=np.array([0]), r=1, labels=[], title="Distance"):

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

    def plotColDistance(self, factors=np.array([0]), r=1, labels=[], title="Cols"):
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
