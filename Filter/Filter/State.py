import auto_diff
import numpy as np  # all math function calls must use np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from BField import Field
from Filter.Plane import Plane

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
        self.absCharge = absCharge
        self.chi2 = np.inf
        
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
        
    def getPosition(self):
        return np.array([self[State.u], self[State.v], self.z])

    def getDirection(self):
        return np.array([np.sin(np.arctan(self[State.tanAlpha])),
                         np.cos(np.arctan(self[State.tanAlpha])) * np.sin(self[State.psi]),
                         np.cos(np.arctan(self[State.tanAlpha])) * np.cos(self[State.psi])])

    @staticmethod
    def getGlobalState(state, z):
        x = np.array([state[State.u, 0], state[State.v, 0], z])
        v = np.array([np.sin(np.arctan(state[State.tanAlpha, 0])),
                         np.cos(np.arctan(state[State.tanAlpha, 0])) * np.sin(state[State.psi, 0]),
                         np.cos(np.arctan(state[State.tanAlpha, 0])) * np.cos(state[State.psi, 0])])
        return np.array([[x[0]], [x[1]], [x[2]], [v[0]], [v[1]], [v[2]], [state[State.qOverP, 0]]])
        
    @staticmethod
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
    
    def propagateTo(self, where, forward = True, tolerance = 1.0e-6, verbose = False):
        # set up an RK state vector to propagate
        if verbose : print("kf_in:", self.state, self.z)
        rk_in, kToG_jacobian = self.rkInitialize()
        if verbose :
            print("rk_in:", rk_in)
            print("in_jac:", kToG_jacobian)
            
        # propagate
        with auto_diff.AutoDiff(rk_in) as rk_in:
            if isinstance(where, (list, Plane)):
                rk_eval = State.adaptiveRKToPlane(rk_in, where, forward, tolerance)
            else: # 'where' is a distance
                rk_eval = State.adaptiveRKToDistance(rk_in, where, tolerance)
            rk_out, rk_jacobian = auto_diff.get_value_and_jacobian(rk_eval)
        if verbose :
            print("rk_out:", rk_out)
            print("rk_jac:", rk_jacobian)
        
        # return new, extrapolated kalman state object
        return self.rkFinalize(rk_out, rk_jacobian, kToG_jacobian, verbose)

    @staticmethod
    def adaptiveRKToPlane(rk_in, planes, forward, tolerance):
        intersectSlopMm = 1.0e-3  # 1 micron

        if isinstance(planes, list):
            targets = planes
        else:
            targets = [planes]

        # Only planes in front of start position are valid targets
        validTargets = []
        if forward:
            delta = np.array([np.inf])
        else:
            delta = np.array([-np.inf])

        localPlane =[]
        for plane in targets:
            d = plane.intersectionDistance(rk_in)
            if abs(d) < intersectSlopMm : localPlane = [plane]
            if (d < intersectSlopMm and forward) or (d > -intersectSlopMm and not forward) : continue
            validTargets += [plane]
            if forward and d < delta[0] : 
                delta[0] = d
            elif not forward and d > delta[0] :
                delta[0] = d

        if len(validTargets) == 0 and len(localPlane) > 0:
            validTargets += localPlane
            delta[0] = 0
        print("Found ", len(validTargets), " valid target planes with minimum distance ", delta.val[0])

        step = delta[0]
        rkNow = rk_in.copy()
        bCache = {"first": np.zeros(3), "next": np.zeros(3), "firstStep": True}
        while delta[0] > intersectSlopMm or delta[0] < -intersectSlopMm :
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
    
            # check termination condition
            for plane in validTargets:
                d = plane.intersectionDistance(rkNow)
                if forward and d < delta[0]: 
                    delta[0] = d
                elif not forward and d > delta[0]:
                    delta[0] = d
            if forward and delta[0] < step: 
                step = delta[0]
            elif not forward and delta[0] > step:
                step = delta[0]

        print("Final distance to target plane: ", delta.val[0])
        return rkNow

    @staticmethod
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
    
    @staticmethod
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

    @staticmethod
    def dot(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    
    @staticmethod
    def cross(v1, v2):
        return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])

    @staticmethod
    def filter(theState, theHits, verbose = False):
        hProjection = np.matrix([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        if verbose :
            print("Input state: \n", theState.state.T, "z =", theState.z)
        state = copy.deepcopy(theState)
        chi2 = 0
        for hit in theHits:
            zHit = hit.z
            prediction = state.propagateTo(Plane(zHit), forward = False)
            if verbose :
                print("Predicted state: \n", prediction.state.T, "z = ", prediction.z)
                print("Predicted cov: \n", prediction.covariance)
            resPrediction = hit.measurement - np.matmul(hProjection, prediction.state)
            covResPrediction = hit.covariance + np.matmul(hProjection, np.matmul(prediction.covariance, hProjection.T))
        
            covFilteredInv = np.matmul(hProjection.T, np.matmul(hit.covariance.I, hProjection)) + prediction.covariance.I
            covFiltered = covFilteredInv.I
            state.z = zHit
            state.state = np.matmul(covFiltered, np.matmul(prediction.covariance.I, prediction.state) + np.matmul(hProjection.T, np.matmul(hit.covariance.I, hit.measurement)))
            state.covariance = covFiltered
            if verbose:
                print("Filtered state: \n", state.state.T, "z = ",state.z)
                print("Filtered cov: \n", state.covariance)
                print("x (pred, meas, filt):", prediction[State.u], hit.measurement[0,0], state[State.u])
                print("y (pred, meas, filt):", prediction[State.v], hit.measurement[1,0], state[State.v])


            resFiltered = hit.measurement - np.matmul(hProjection, state.state)
            chi2Add = np.matmul(resFiltered.T, np.matmul(hit.covariance.I, resFiltered)) + np.matmul((state.state.T - prediction.state.T),np.matmul(prediction.covariance.I, state.state - prediction.state))
            chi2 += chi2Add
    
        if verbose :
            print("Fit pt, Chi2: ", state.PGeV(), chi2[0,0])
        
        state.chi2 = chi2[0,0]
        return state
    