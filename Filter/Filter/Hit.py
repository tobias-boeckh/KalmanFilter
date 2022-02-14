import Filter
import numpy as np
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
