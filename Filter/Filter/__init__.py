from BField import Field
from Filter.State import State
import numpy as np

def CallSomething():
    s=State(signedMomentumGeV = 10)
    sNew = s.propagateToDistance(1000.0, 1.0e-6)
    print(sNew.state)

