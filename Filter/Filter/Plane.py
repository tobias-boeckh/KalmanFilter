import numpy as np

class Plane(object):
    
    def __init__(self, z0, x0 = 0, y0 = 0, normal = np.array([0, 0, 1])):
        self.normal = normal/np.linalg.norm(normal)
        self.p = np.array([x0, y0, z0])

    # computes distance from state to nearest point on (presumed infinite) plane; non-negative
    def projectedDistance(self, state):
        return(np.abs(np.dot(self.normal, self.p - state.getPosition())))

    # computes straight-line distance to intersection along current direction; may be negative
    def intersectionDistance(self, state):
        if self.projectedDistance(state) == 0 : return 0
        vdotn = np.dot(self.normal, state.getDirection())
        if vdotn == 0 : return np.inf
        return np.dot(self.normal, self.p - state.getPosition()) / vdotn