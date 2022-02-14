import Filter
import numpy as np
import scipy.optimize

class Hit(object):
    xResolutionMm = 0.8585
    yResolutionMm = 0.0171
    def __init__(self, state):
        self.dx = Hit.xResolutionMm
        self.dy = Hit.yResolutionMm
        p = state.getPosition()
        self.x = p[0] + np.random.normal(0, self.dx)
        self.y = p[1] + np.random.normal(0, self.dy)
        self.z = p[2]
        self.covariance = np.matrix([[self.dx**2, 0],[0, self.dy**2]])
        self.measurement = np.matrix([[self.x],[self.y]])

    @staticmethod
    def chi2(x, *args):
        hits = args[0]
        sum = 0
        zStart = hits[len(hits)-1].z
        for hit in hits:
            sum += (hit.x - x[0] - x[2] * (hit.z - zStart))**2/hit.dx**2
            sum += (hit.y - x[1] - x[3] * (hit.z - zStart))**2/hit.dy**2
            #print(hit.z)
        return sum

    @staticmethod
    def lineFit(hits, n = 3):
        return scipy.optimize.minimize(Hit.chi2, (0,0,0,0), args=(hits[-n:]) )