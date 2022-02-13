import auto_diff
import numpy as np  # all math function calls must use np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from BField import Field

# Adaptive RK code stolen from ATLAS: ATL-SOFT-PUB-2009-001
class State:
    lightSpeedSI = 299792458 # exact   
    bScale = lightSpeedSI/1.0e12                       # unit conversion factor to allow:
                                                             # charge in units of e
                                                             # lengths in mm
                                                             # momentum in GeV/c and 
                                                             # B in Tesla
    u = 0
    v = 1
    tanAlpha = 2
    psi = 3
    qOverP = 4
    
    def __init__(self, 
                 absCharge = 1,
                 line = np.array([0, 0, 0, 0]),
                 zLine = 0,
                 lineCov = np.matrix([[1.0, 0.0, 0.0, 0.0],   # covariance of the four fit parameters
                                     [0.0, 1.0, 0.0, 0.0],   # default uncertainties on positions and angles are 1mm / ~45 degrees, respectively
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]]),
                 signedMomentumGeV = np.inf,
                 qOverPUncertainty = 1.0):
        self.history = np.array([])
        self.pathLength = 0
        self.J = np.identity(n = 5)
        self.absCharge = absCharge
        
        # translate linear fit to state vector variables

        psi = np.arctan(line[3])
        alpha = np.arccos(1/np.sqrt(1 + np.cos(psi)**2 * line[2]**2))
        if signedMomentumGeV == np.inf:
            qOverP = 0
        else:
            qOverP = absCharge / signedMomentumGeV
            
        self.state = np.matrix([[line[0]], [line[1]], [np.tan(alpha)], [psi], [qOverP]])
        self.z = zLine
        
        # Jacobian for fit -> state variable transformation
        
        fitToState = np.matrix([[1, 0, 0, 0], 
                                [0, 1, 0, 0],  
                                [0, 0, 1/np.sqrt(1 + line[3]**2), -line[2] * line[3]/(1 + line[3]**2)**(3/2)],
                                [0, 0, 0, 1/(1+line[3]**2)]])
        
        stateCov = np.matmul(fitToState, np.matmul(lineCov, fitToState.T))
        
        # initialize covariance matrix

        self.covariance = np.matrix([[stateCov[0, 0], stateCov[0, 1], stateCov[0, 2], stateCov[0, 3], 0],
                                     [stateCov[1, 0], stateCov[1, 1], stateCov[1, 2], stateCov[1, 3], 0],
                                     [stateCov[2, 0], stateCov[2, 1], stateCov[2, 2], stateCov[2, 3], 0],
                                     [stateCov[3, 0], stateCov[3, 1], stateCov[3, 2], stateCov[3, 3], 0],
                                     [0, 0, 0, 0, qOverPUncertainty**2]])
        
        self.recordState()
    
    def __getitem__(self, index):
        return self.state[index, 0]
    
    def recordState(self):
        now = { "absCharge" : self.absCharge,
                "pathLength" : self.pathLength,
                "z" : self.z,
                "state" : self.state.copy(),
                "covariance" : self.covariance.copy(),
              }
        if len(self.history) == 0 or now != self.history[-1]:
            self.history = np.append(self.history, now)

    def show(self):
        # analyze upper 2x2 covariance submatrix
        sxx = self.covariance[State.u, State.u]
        sxy = self.covariance[State.u, State.v]
        syy = self.covariance[State.v, State.v]
        
        lambdaHi = ((sxx + syy) + np.sqrt((sxx-syy)**2 + 4 * sxy**2))/2
        lambdaLo = ((sxx + syy) - np.sqrt((sxx-syy)**2 + 4 * sxy**2))/2        

        theta = np.arctan2(lambdaHi - sxx, sxy)
        
        center = (self[State.u],self[State.v])
        errorEllipse = Ellipse(center, lambdaHi, lambdaLo, theta*180/np.pi)
        
        a = plt.subplot(111, aspect='equal')
        errorEllipse.set_clip_box(a.bbox)
        errorEllipse.set_alpha(1)
        a.add_artist(errorEllipse)

        wid = 2.5*np.sqrt(max(sxx, syy))
        plt.xlim(self[State.u]-wid, self[State.u]+wid)
        plt.ylim(self[State.v]-wid, self[State.v]+wid)
        plt.show()
        
    def getGlobalState(state, z):
        x = np.array([state[State.u, 0], state[State.v, 0], z])
        v = np.array([np.sin(np.arctan(state[State.tanAlpha, 0])),
                         np.cos(np.arctan(state[State.tanAlpha, 0])) * np.sin(state[State.psi, 0]),
                         np.cos(np.arctan(state[State.tanAlpha, 0])) * np.cos(state[State.psi, 0])])
        return np.array([[x[0]], [x[1]], [x[2]], [v[0]], [v[1]], [v[2]], [state[State.qOverP, 0]]])
        
    def getKalmanState(globalState):
        # work-around missing arctan2
        if globalState[5, 0] != 0:
            psiRaw = np.arctan(globalState[4, 0] / globalState[5, 0])
        else:
            if globalState[4, 0] > 0:
                psiRaw = np.pi/2
            else:
                psiRaw = -np.pi/2
            
        if globalState[5, 0] < 0:
            if psiRaw > 0:
                psi = np.pi - psiRaw
            else:
                psi = -np.pi - psiRaw
        else:
            psi = psiRaw
            
        state = np.array([ [globalState[0, 0]], 
                           [globalState[1, 0]], 
                           [globalState[3, 0] / np.sqrt(globalState[4, 0]**2 + globalState[5, 0]**2)],
                           #[np.arctan2(globalState[4, 0], globalState[5, 0])],
                           [psi],
                           [globalState[6,0]] ])
        return state

    def rkInitialize(self):
        with auto_diff.AutoDiff(self.state) as state:
            kalmanToGlobal_eval = State.getGlobalState(state, self.z)
            return auto_diff.get_value_and_jacobian(kalmanToGlobal_eval)
        
    def rkFinalize(self, rk_out, rk_jacobian, kToG_jacobian, verbose):
        new = copy.deepcopy(self)
        # transform global state back to KF
        new.z = rk_out[2, 0]
        with auto_diff.AutoDiff(rk_out) as rk_out:
            globalToKalman_eval = State.getKalmanState(rk_out)
            new.state, gToK_jacobian = auto_diff.get_value_and_jacobian(globalToKalman_eval)

        if verbose :
            print("kf_out:", new.state, new.z)
            print("out_jac:", gToK_jacobian)
    
        # update covariance
        if verbose : print("cov_in:", self.covariance)
        cGlobal_in = np.matmul(kToG_jacobian, np.matmul(self.covariance, kToG_jacobian.T))
        if verbose : print("covGlobal_in:", cGlobal_in)
        cGlobal_out = np.matmul(rk_jacobian,np.matmul(cGlobal_in,rk_jacobian.T))
        if verbose : print("covGlobal_out:", cGlobal_out)
        new.covariance = np.matmul(gToK_jacobian,np.matmul(cGlobal_out,gToK_jacobian.T))
        if verbose : print("cov_out:", new.covariance)        
        return new    
    
    def propagateToDistance(self, distance, tolerance, verbose = False):
        # set up an RK state vector to propagate
        if verbose : print("kf_in:", self.state, self.z)
        rk_in, kToG_jacobian = self.rkInitialize()
        if verbose :
            print("rk_in:", rk_in)
            print("in_jac:", kToG_jacobian)
            
        # propagate
        with auto_diff.AutoDiff(rk_in) as rk_in:
            rk_eval = State.adaptiveRKToDistance(rk_in, distance, tolerance)
            rk_out, rk_jacobian = auto_diff.get_value_and_jacobian(rk_eval)
        if verbose :
            print("rk_out:", rk_out)
            print("rk_jac:", rk_jacobian)
        
        # return new, extrapolated kalman state object
        return self.rkFinalize(rk_out, rk_jacobian, kToG_jacobian, verbose)
            
    def adaptiveRKToDistance(rk_in, distance, tolerance):
        step = distance
        where = 0
        rkNow = rk_in.copy()
        bCache = {"first": np.zeros(3), "next": np.zeros(3), "firstStep": True}
        while where < distance:
            # try step with current size
            rkNext, error = State.rkStep(rkNow, step, bCache)
            firstStep = False

            # compute next stepsize
            lastStep = step
            step = step * min(max(0.25, np.sqrt(tolerance/error)), 4)

            # check tolerance
            if (error > 4 * tolerance): continue
                
            # step succeeded
            rkNow = rkNext.copy()
            bCache["first"] = bCache["next"].copy()
            where = where + lastStep
            if where + step > distance:
                step = distance - where
                
        return rkNow
    
    def rkStep(rk_in, step, bCache):
        initialPos = np.array([rk_in[0, 0], rk_in[1, 0], rk_in[2, 0]])
        initialDir = np.array([rk_in[3, 0], rk_in[4, 0], rk_in[5, 0]])
        
        pnt1 = initialPos.copy()
        dir1 = initialDir.copy()
        if bCache["firstStep"] :
            bCache["first"] = Field([pnt1.val[0], pnt1.val[1], pnt1.val[2]])
            bCache["firstStep"] = False
        
        k1 = State.bScale * rk_in[6, 0] * State.cross(dir1, bCache["first"])
        
        pnt23 = initialPos + (step / 2) * initialDir + (((step / 2)**2) / 2) * k1
        dir2 = initialDir + (step / 2) * k1
        
        b23 = Field([pnt23.val[0], pnt23.val[1], pnt23.val[2]])
        k2 = State.bScale * rk_in[6, 0] * State.cross(dir2, b23)
        
        dir3 = initialDir + (step / 2) * k2
        k3 = State.bScale * rk_in[6, 0] * State.cross(dir3, b23)
        
        pnt4 = initialPos + step * initialDir + ((step**2) / 2) * k3
        dir4 = initialDir + step * k3
        
        b4 = Field([pnt4.val[0], pnt4.val[1], pnt4.val[2]])
        bCache["next"] = b4.copy()
        k4 = State.bScale * rk_in[6, 0] * State.cross(dir4, b4)
        
        # error estimate
        error =  (k1 - k2 - k3 + k4)
        errorMag = (step * step) * np.sqrt(State.dot(error, error))
        errorSafe = max(errorMag, 1.0e-20)
        
        finalPos = initialPos + step * initialDir + ((step**2) / 6) * (k1 + k2 + k3)
        rawDir = initialDir + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        norm = np.sqrt(State.dot(rawDir, rawDir))
        finalDir = rawDir / norm
        
        return np.array([[finalPos[0]],[finalPos[1]],[finalPos[2]],[finalDir[0]],[finalDir[1]],[finalDir[2]],[rk_in[6, 0]]]), errorSafe

    def dot(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    
    def cross(v1, v2):
        return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])