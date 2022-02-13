import math
from math import sin, cos, tan, asin, acos, atan, atan2, sqrt, pi
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from scipy.special import iv, kn
from scipy.integrate import quad
junk = np.seterr()

# Geometry description
rMin = 0.104
rMax = rMin + 0.0825
zMax = [1.5, 1.0, 1.0]
magnetZ = [-0.81232, 0.637726, 1.837726]
m0 = 1.04                                           # mag
sectorPhi = np.linspace(0, 2*pi, 17, True)       # central phi value for each sector
magPhi = 2 * sectorPhi                              # angle of the magnitization in each sector
phiConst = 4 * sqrt(2 - sqrt(2))                    # numerical result of phiPrime integration on curved surface
rhoLimit = rMin/100                                 # region to use linear approx for rho
proj = np.zeros(17)
magVector = {}
for j in range(0, 16, True):
    magLo = m0 * np.array([cos(magPhi[j]), sin(magPhi[j]), 0])
    magVector[j] = magLo
    magHi = m0 * np.array([cos(magPhi[j+1]), sin(magPhi[j+1]), 0])
    phiHat = np.array([-sin(j * pi / 8 + pi / 16), cos(j * pi / 8 + pi / 16), 0])
    proj[j] = np.matmul((magLo - magHi), phiHat)
    #print(j, proj[j])


def Field(position):
    # expects position in millimeters
    return bFieldXYZ(position[0]/1000, position[1]/1000, position[2]/1000)

# negative derivative (with respect to rho) of integrand over curved surface
def cylDrho(rho, phi, z, rhoPrime, L, k):
    if (rho < rhoPrime) :
        bessel_k = kn(1, k * rhoPrime)
        if (bessel_k == 0): return 0
        return -m0 * (phiConst / pi**2) * rhoPrime * (iv(0, k * rho) + iv(2, k * rho)) * \
            bessel_k * cos(k * z) * cos(phi) * sin(k * L / 2)

    else:
        bessel_k = kn(0, k * rho) + kn(2, k * rho)
        if (bessel_k == 0): return 0
        return -m0 * (phiConst / pi**2) * rhoPrime * iv(1, k * rhoPrime) * bessel_k * \
                cos(k * z) * cos(phi) * sin(k * L / 2)


# negative derivative (with respect to phi) of integrand over curved surface, divided by rho
def cylDphi(rho, phi, z, rhoPrime, L, k):
    if (rho < rhoLimit):
        bessel_k = kn(1, k * rhoPrime)
        if (bessel_k == 0): return 0        
        return m0 * (phiConst / pi**2) * rhoPrime * bessel_k * cos(k * z) * sin(k * L / 2) * sin(phi)
    elif (rho < rhoPrime):
        bessel_k = kn(1, k * rhoPrime)
        if (bessel_k == 0): return 0                
        return m0 * (2 * phiConst /(k * pi**2)) * rhoPrime * iv(1, k * rho) * bessel_k * cos(k * z) * sin(k * L / 2) * sin(phi) / rho
    else:
        bessel_k = kn(1, k * rho)
        if (bessel_k == 0): return 0
        return m0 * (2 * phiConst / (k * pi**2)) * rhoPrime * iv(1, k * rhoPrime) * bessel_k * cos(k * z) * sin(k * L / 2) * sin(phi) / rho

# negative derivative (with respect to z) of integrand over curved surface; valid for rho < rhoPrime
def cylDz(rho, phi, z, rhoPrime, L, k):
    if (rho < rhoPrime):
        bessel_k = kn(1, k * rhoPrime)
        if (bessel_k == 0): return 0        
        return -m0 * (2 * phiConst / pi **2) * rhoPrime * iv(1, k * rho) * bessel_k * sin(k * z) * sin(k * L / 2) * cos(phi)
    else:
        bessel_k = kn(1, k * rho)
        if (bessel_k == 0): return 0
        return m0 * (2 * phiConst / pi**2) * rhoPrime * iv(1, k * rhoPrime) * bessel_k * sin(k * z) * sin(k * L / 2) * cos(phi)

# negative derivative (with respect to rho) of integrand over sector boundary surface
def planeDrho(rho, phi, z, j, L, rhoPrime):
    phiPrime = j * pi / 8 + pi / 16
    aSq = rho**2 + rhoPrime**2 - 2 * rho * rhoPrime * cos(phi - phiPrime)
    aSqMinus = aSq + (z - L / 2)**2
    aSqPlus  = aSq + (z + L / 2)**2
    return (proj[j] / (4 * pi)) * (rho - rhoPrime * cos(phi - phiPrime)) * \
        (1/((sqrt(aSqPlus) + z + L / 2) * sqrt(aSqPlus)) - 1/((sqrt(aSqMinus) + z - L / 2) * sqrt(aSqMinus)))

# negative derivative (with respect to phi) of integrand over sector boundary surface, divided by rho
def planeDphi(rho, phi, z, j, L, rhoPrime):
    phiPrime = j * pi / 8 + pi / 16
    aSq = rho**2 + rhoPrime**2 - 2 * rho * rhoPrime * cos(phi - phiPrime)
    aSqMinus = aSq + (z - L / 2)**2
    aSqPlus  = aSq + (z + L / 2)**2
    return (proj[j] / (4 * pi)) * rhoPrime * \
        (1/((sqrt(aSqPlus) + z + L / 2) * sqrt(aSqPlus)) - 1/((sqrt(aSqMinus) + z - L / 2) * sqrt(aSqMinus))) \
        * sin(phi - phiPrime)

# negative derivative (with respect to z) of integrand over sector boundary surface, divided by rho    
def planeDz(rho, phi, z, j, L, rhoPrime):
    phiPrime = j * pi / 8 + pi / 16
    aSq = rho**2 + rhoPrime**2 - 2 * rho * rhoPrime * cos(phi - phiPrime)
    aSqMinus = aSq + (z - L / 2)**2
    aSqPlus  = aSq + (z + L / 2)**2
    return (proj[j] / (4 * pi)) * (1/sqrt(aSqPlus) - 1/sqrt(aSqMinus))

def field(rho, phi, z, L):
    fieldCylRho = lambda k : -cylDrho(rho, phi, z, rMin, L, k) + cylDrho(rho, phi, z, rMax, L, k)
    fieldPlaneRho = lambda rhoPrime : sum(list(map(lambda j : planeDrho(rho, phi, z, j, L, rhoPrime), range(16))))
    
    fieldCylPhi = lambda k : -cylDphi(rho, phi, z, rMin, L, k) + cylDphi(rho, phi, z, rMax, L, k)
    fieldPlanePhi = lambda rhoPrime : sum(list(map(lambda j : planeDphi(rho, phi, z, j, L, rhoPrime), range(16))))
    
    fieldCylZ = lambda k : -cylDz(rho, phi, z, rMin, L, k) + cylDphi(rho, phi, z, rMax, L, k)
    fieldPlaneZ = lambda rhoPrime : sum(list(map(lambda j : planeDz(rho, phi, z, j, L, rhoPrime), range(16))))
    myLimit = 1000    
    return np.array([quad(fieldCylRho, 0, np.inf, limit = myLimit)[0] + quad(fieldPlaneRho, rMin, rMax, limit = myLimit)[0], \
                     quad(fieldCylPhi, 0, np.inf, limit = myLimit)[0] + quad(fieldPlanePhi, rMin, rMax, limit = myLimit)[0], \
                     quad(fieldCylZ, 0, np.inf, limit = myLimit)[0] + quad(fieldPlaneZ, rMin, rMax, limit = myLimit)[0]])

def mField(rho, phi, z):
    magnetization = np.zeros(3)
    if phi < 0: phi = phi + 2 * pi
    for iz in range(3):
        if z < (magnetZ[iz] - zMax[iz]/2) or z > (magnetZ[iz] + zMax[iz]/2) or rho < rMin or rho > rMax: continue
        j = (phi + pi/16)//(pi/8)
        j = j % 16
        magnetization = magVector[j]  # this is cartesian!
        break
    rhoUnit = np.array([cos(phi), sin(phi), 0])
    phiUnit = np.array([-sin(phi), cos(phi), 0])
    zUnit   = np.array([0, 0, 1])
    return np.array([np.matmul(magnetization, rhoUnit), np.matmul(magnetization, phiUnit), np.matmul(magnetization, zUnit)])

def mFieldXYZ(x, y, z):
    rho = sqrt(x**2 + y**2)
    if rho > 0:
        phi = atan2(y, x)
    else:
        phi = 0
    
    m = mField(rho, phi, z)
    return np.array([m[0] * cos(phi) - m[1] * sin(phi), m[0] * sin(phi) + m[1] * cos(phi), m[2]])

def bField(rho, phi, z):
    return hField(rho, phi, z - magnetZ[0], zMax[0]) \
         + hField(rho, phi, z - magnetZ[1], zMax[1]) \
         + hField(rho, phi, z - magnetZ[2], zMax[2]) \
         + mField(rho, phi, z)


def bFieldXYZ(x, y, z):
    return hFieldXYZ(x, y, z) + mFieldXYZ(x, y, z)

def hField(rho, phi, z):
    return field(rho, phi, z - magnetZ[0], zMax[0]) \
         + field(rho, phi, z - magnetZ[1], zMax[1]) \
         + field(rho, phi, z - magnetZ[2], zMax[2]) 

def hFieldXYZ(x, y, z):
    rho = sqrt(x**2 + y**2)
    if rho > 0:
        phi = atan2(y, x)
    else:
        phi = 0
    
    h = hField(rho, phi, z)
    return np.array([h[0] * cos(phi) - h[1] * sin(phi), h[0] * sin(phi) + h[1] * cos(phi), h[2]])