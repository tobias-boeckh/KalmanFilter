import numpy as np
from numpy.lib.arraysetops import isin
#from Filter.State import State
import Filter
import auto_diff

class Plane(object):
    
    def __init__(self, z0, x0 = 0, y0 = 0, normal = np.array([0, 0, 1])):
        self.normal = normal/np.linalg.norm(normal)
        self.p = np.array([x0, y0, z0])

    # computes distance from state to nearest point on (presumed infinite) plane; non-negative
    def projectedDistance(self, state):
        if isinstance(state, Filter.State):
            returnVal = np.abs(np.dot(self.normal, self.p - state.getPosition()))
        else:
            dotProd = np.array([Plane.dot(self.normal, self.p - np.array([state[0, 0], state[1, 0], state[2, 0]]))])
            returnVal = np.abs(dotProd[0])
        return returnVal

    # computes straight-line distance to intersection along current direction; may be negative
    def intersectionDistance(self, state):
        if self.projectedDistance(state) == 0 : return 0
        if isinstance(state, Filter.State):
            r = state.getPosition()
            v = state.getDirection()
        else:
            r = np.array([state[0, 0], state[1, 0], state[2, 0]])
            v = np.array([state[3, 0], state[4, 0], state[5, 0]])
            v = v / np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        vdotn = Plane.dot(self.normal, v)
        if vdotn == 0 : return np.inf
        return Plane.dot(self.normal, self.p - r) / vdotn

    @staticmethod
    def dot(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]