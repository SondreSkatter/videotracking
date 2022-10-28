import numpy as np

class chidist:
    def __init__(self):
        self.resolution = 0.05
        self.zLimits = np.array([-50.0,50.0])
        self.N = int(1 + round((self.zLimits[1] - self.zLimits[0] ) / self.resolution)) 
        self.probs = np.zeros(self.N,np.float)
        self.dfs = np.array([3,5])
        self.dfs = np.array([2,3,4,5])
        #self.dfs = np.array([2,3,5,6])  # if color is included (3 feats)
        if (1):
            self.generateTable()
        else:
            self.readTable()
    def shutdown(self):
        pass

    def generateTable(self):
        import scipy.stats
        zVals = np.linspace(self.zLimits[0],self.zLimits[1],self.N)
        self.probs = np.zeros((zVals.size,self.dfs.size), np.float)
        for i, df in enumerate(self.dfs):
            self.probs[:,i] = scipy.stats.chi2.cdf(zVals, df)
        np.savez("utils/chivals"+str(self.dfs) + ".npz", probs = self.probs,
            zLimits = self.zLimits,
            resolution = self.resolution)

    def readTable(self):
        data = np.load("utils/chivals"+str(self.df) + ".npz")
        self.probs = data['probs']
        data.close()

    def cdf(self,zValue,dfin):

        if zValue.size > 1:
            # zValue is numpy
            Shape = zValue.shape
            inds = np.round(((zValue.flatten() - self.zLimits[0] ) / self.resolution - 0.5)).astype(np.int)
            inds[inds<0] = 0
            inds[inds> self.N-1] = self.N-1
            Out = np.zeros(zValue.size, np.float)

            dfins = np.unique(dfin)
            assert np.all(np.isin(dfins,self.dfs)), "No precomputed chisquare values..."

            for i, df in enumerate(self.dfs):
                dfInd = np.where(dfin.ravel()==df)[0]
                if dfInd.size > 0:
                    Out[dfInd] = self.probs[inds[dfInd],i]

            return Out.reshape(Shape)
        elif zValue.size == 1:    
            dfInd = np.where(self.dfs == dfin)[0]
            assert dfInd.size == 1, "No precomputed chisquare values..."
            ind = min(self.N-1,max(0,int(np.round(((zValue - self.zLimits[0]) / self.resolution - 0.5)))))
            return self.probs[ind,dfInd].copy()
        else: # zValue.size == 0:
            return np.zeros(0)