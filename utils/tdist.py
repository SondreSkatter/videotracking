import numpy as np
import platform

class tdist:
    def __init__(self):
        self.resolution = 0.05
        self.tLimits = np.array([-50.0,50.0])
        self.N = int(1 + round((self.tLimits[1] - self.tLimits[0] ) / self.resolution)) 
        self.probs = np.zeros(self.N,np.float)
        if (0 and platform.system() == 'Windows'):
            self.generateTable()
        else:
            self.readTable()
    def shutdown(self):
        pass

    def generateTable(self):
        import scipy.stats
        tVals = np.linspace(self.tLimits[0],self.tLimits[1],self.N)
        self.probs = scipy.stats.t.cdf(tVals,1)
        np.savez('tvals.npz', probs = self.probs,
            tLimits = self.tLimits,
            resolution = self.resolution)

    def readTable(self):
        data = np.load('utils/tvals.npz')
        self.probs = data['probs']
        data.close()

    def tcdf(self,tValue):
        # tValue must be numpy
        inds = np.round(((tValue - self.tLimits[0] ) / self.resolution - 0.5)).astype(np.int)
        inds[inds<0] = 0
        inds[inds> self.N-1] = self.N-1
        return self.probs[inds]




#TT = tdist()