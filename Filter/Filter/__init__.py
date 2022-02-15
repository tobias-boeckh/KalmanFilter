from numpy.random.mtrand import normal
from BField import Field
from Filter.State import State
from Filter.Plane import Plane
from Filter.Hit import Hit
import numpy as np

def CallSomething():
    s = State(signedMomentumGeV = 100)
    p = Plane(1000)
    print(p.projectedDistance(s), p.intersectionDistance(s))

    sNew = s.propagateTo(1000.0)
    print(sNew.state, sNew.z)
    print(p.projectedDistance(sNew), p.intersectionDistance(sNew))
    h = Hit(sNew)

    sNewer = sNew.propagateTo(10.0)
    print(p.projectedDistance(sNewer), p.intersectionDistance(sNewer))    

    sPlane = s.propagateTo(p)
    print(sPlane.state, sPlane.z)
    print(p.projectedDistance(sPlane), p.intersectionDistance(sPlane))
    planes = [p, Plane(-1000), Plane(1500)]
    sPlanes = s.propagateTo(planes)
    print(sPlanes.state, sPlanes.z)
    print(p.projectedDistance(sPlanes), p.intersectionDistance(sPlanes))
    p2 = Plane(-1000)
    sBack = s.propagateTo(planes, forward = False)
    print(sBack.state, sBack.z)
    print(p2.projectedDistance(sBack), p2.intersectionDistance(sBack))

def GenDetectors(zPositions):
    detectors = []
    for z in zPositions:
        detectors += [Plane(z)]
    return detectors


def GenHits(truthState, zDetectors):
    hits = []
    detectors = GenDetectors(zDetectors)

    sNow = truthState
    for detector in detectors:
        sNext = sNow.propagateTo(detector)
        hits += [Hit(sNext)]
        sNow = sNext
    return hits

def RunFilter(spatialWidthMm = 25, angularWidthRadians = 0.01, eGeV = 100.0):
    truthState = State(line = np.array([np.random.normal(0, spatialWidthMm), 
                                        np.random.normal(0, spatialWidthMm), 
                                        np.random.normal(0, angularWidthRadians),
                                        np.random.normal(0, angularWidthRadians)]), 
                       zLine = 0, 
                       signedMomentumGeV = eGeV)
    detectors =[16, 47, 78, 1207, 1238, 1269, 2398, 2429, 2460]
    hits = GenHits(truthState, detectors)
    result = Hit.lineFit(hits)
    seedState = State(line = result.x, lineCov = 9*result.hess_inv, zLine = detectors[-1])
    recoState = State.filter(seedState, reversed(hits))
    return recoState, truthState