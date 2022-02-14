from BField import Field
from Filter.State import State
from Filter.Plane import Plane
import numpy as np

def CallSomething():
    s = State(signedMomentumGeV = 100)
    p = Plane(1000)
    print(p.projectedDistance(s), p.intersectionDistance(s))

    sNew = s.propagateTo(1000.0)
    print(sNew.state, sNew.z)
    print(p.projectedDistance(sNew), p.intersectionDistance(sNew))

    sNewer = sNew.propagateTo(10.0)
    print(p.projectedDistance(sNewer), p.intersectionDistance(sNewer))    

    sPlane = s.propagateTo(p)
    print(sPlane.state, sPlane.z)
    print(p.projectedDistance(sPlane), p.intersectionDistance(sPlane))
    planes = [p, Plane(-1000), Plane(1500)]
    sPlanes = s.propagateTo(planes)
    print(sPlanes.state, sPlanes.z)
    print(p.projectedDistance(sPlanes), p.intersectionDistance(sPlanes))
