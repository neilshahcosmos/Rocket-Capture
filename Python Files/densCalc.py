# %%
# Import Statements
import numpy as np
import scipy as sp
from scipy import integrate, interpolate
import scipy.fft as fft
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import h5py
import os
import time

# %%
# Global Rocket Parameters
xi = 50
Gmu = 1e-14
GammaFrac = 0.1
vf = 0.3

# Global Simulation Parameters
tmax = 498
zi = 127
h = 0.7
rhoScale = 4.78e-20 # Mpc / Msun

Lbox = 100
LLowRes, LHighRes = [100, 1]
dxLowRes, dxHighRes = [1, 0.01]
binsLowRes, binsHighRes = [int(LLowRes / dxLowRes) + 1, int(LHighRes / dxHighRes)]

# Global Physical Parameters
t0 = 4213 / h # Mpc / h
H0 = 0.0003333

# No. of Rockets
densRocket = 1e-6 * ((1e-12 * xi) ** (-3/2)) * (t0 ** -3) * ((1 + zi) ** 3)
nRockets = int(densRocket * ((LHighRes / 2) ** 3))

# %%
# Set the Environment
envi = 0
envs = ["Cluster", "Local"]
env = envs[envi]

if env == "Cluster":
    # Cluster dirs
    neilDir = "/cluster/tufts/hertzberglab/nshah14/"
    homeDir = "/cluster/tufts/hertzberglab/shared/Rockets/"
    simName = "Sim_500v"
    simDir = homeDir + simName + "/"
    Div = "/"
elif env == "Local":
    # Cluster dirs
    homeDir = "C:\\Users\\Neil\\Documents\\PhD\\Rocket Force\\"
    simName = "Sim_500v"
    simDir = homeDir + simName + "\\"
    Div = "\\"

print("Working in the {} environment".format(env))

# %%
def toMS(t):
    s = np.floor(np.mod(t, 60))
    m = np.floor(np.mod(t, 3600) / 60)
    h = np.floor(t / 3600)

    if t < 1:
        tstr = "{:f} s".format(t)
    elif t < 3600:
        tstr = "{:02d}m {:02d}s".format(int(m), int(s))
    else:
        tstr = "{}h {:02d}m {:02d}s".format(int(h), int(m), int(s))
    return tstr

# make more realistic, add in influence of overdensity
def getHubbleEvol():
    OMatter = 0.25
    OLambda = 0.75

    da_dt = lambda a, t: a * H0 * np.sqrt(OMatter / a**3 + OLambda)

    a0 = 1 / (1 + zi)
    tInt = np.linspace(0, 1.1 / H0, 1000)
    af = interpolate.InterpolatedUnivariateSpline(tInt, integrate.odeint(da_dt, y0=a0, t=tInt)[:, 0])
    aDotf = af.derivative(n=1)
    tEnd = sp.optimize.fsolve(lambda t: af(t) - 1.0, x0=(1.0 / H0))[0]

    tArr = np.asarray([(tEnd + t)**(t / tmax) - 1 for t in range(tmax)])
    aArr = np.asarray([af(t) for t in tArr])
    HArr = np.asarray([aDotf(t) / af(t) for t in tArr])

    return tArr, aArr, HArr

# %%
def getCoords(t):
    snapDir = simDir + "snapdir_{:03d}{}".format(t, Div)
    pathArr = np.asarray(os.listdir(snapDir))
    
    ptypeN = 3
    coordsArr = [np.empty((0, 3), dtype=float) for i in range(ptypeN)]

    for pi in range(ptypeN):
        ptype = pi + 1
        for pathi in np.arange(0, pathArr.size):
            datGet = "/PartType{:d}/Coordinates".format(ptype)
            try:
                coords = np.asarray(h5py.File(snapDir + pathArr[pathi], 'r')[datGet])
                coordsArr[pi] = np.concatenate([coordsArr[pi], coords], axis=0)
            except KeyError:
                # print("   (Warning! Could not find ptype {:d} for snapshot t = {:d})".format(ptype, t))
                pass
    
    return(coordsArr)

def getHaloCoords(t):
    haloPtype = 1

    groupDir = simDir + "groups_{:03d}{}".format(t, Div)
    snapDir = simDir + "snapdir_{:03d}{}".format(t, Div)
    groupPaths = np.asarray(os.listdir(groupDir))
    snapPaths = np.asarray(os.listdir(snapDir))

    coordsArr = getCoords(t)[haloPtype - 1]
    
    haloCoords = np.empty((0, 0))
    for i, pathi in enumerate(groupPaths):
        gfile = h5py.File(groupDir + pathi, 'r')
        try:
            haloInit = int(np.asarray(gfile["/Group/GroupOffsetType/"])[0, haloPtype - 1])
            haloN = int(np.asarray(gfile["/Group/GroupLen/"])[haloPtype - 1])
            haloCoords = coordsArr[haloInit:(haloInit + haloN + 1), :]
            break
        except KeyError:
            pass
    
    return(haloCoords)

def getHaloStartTime():
    pathSave = simDir + "HaloStartTime.npy"

    if os.path.exists(pathSave):
        tStart = np.load(pathSave)[0]
    else:
        tStart = 0
        go = True
        for t in range(tmax):
            groupDir = simDir + "groups_{:03d}{}".format(t, Div)
            snapDir = simDir + "snapdir_{:03d}{}".format(t, Div)
            groupPaths = np.asarray(os.listdir(groupDir))
            snapPaths = np.asarray(os.listdir(snapDir))
            
            for i, pathi in enumerate(groupPaths):
                gfile = h5py.File(groupDir + pathi, 'r')
                try:
                    catch = gfile["/Group/GroupOffsetType/"]
                    tStart = t
                    go = False
                    break
                except KeyError:
                    continue
            if not go:
                break
        np.save(pathSave, np.asarray([tStart]))
    
    return(tStart)

def getHaloC(t):
    haloCoords = getHaloCoords(t)
    if haloCoords.shape[0] == 0:
        t0 = getHaloStartTime()
        halo0 = getHaloCoords(t0)
        xc = np.asarray([np.mean(halo0[:, i]) for i in range(3)])
    else:
        xc = np.asarray([np.mean(haloCoords[:, i]) for i in range(3)])
    
    return(xc)

# %%
def getMass(t):
    snapDir = simDir + "snapdir_{:03d}/".format(t)
    snapPaths = np.asarray(os.listdir(snapDir))

    ptypeN = 3
    massArr = [np.empty(0, dtype=float) for i in range(ptypeN)]
    for pi in range(ptypeN):
        ptype = pi + 1
        for pathi, path in enumerate(snapPaths):
            datGet = "/PartType{:d}/Masses".format(ptype)
            try:
                masses = np.asarray(h5py.File(snapDir + path, 'r')[datGet])
                massArr[pi] = np.concatenate([massArr[pi], masses], axis=0)
            except KeyError:
                # print("   (Warning! Could not find ptype {:d} for snapshot t = {:d})".format(ptype, t))
                pass
    
    return(massArr)

# %%
def getRhoxHC(t, override=False):
    dirSave = simDir + "rhox_HaloCentered{}rhox_HC__LLR_{:03d}_{:03d}__bLR_{:04d}__LHR_{:03d}_{:03d}__bHR_{:04d}{}".format(
        Div, int(np.floor(LLowRes)), int(1e3 * np.mod(LLowRes, 1)), binsLowRes, int(np.floor(LHighRes)), int(1e3 * np.mod(LHighRes, 1)), binsHighRes, Div)
    if not os.path.exists(dirSave):
        os.makedirs(dirSave)
    pathSave = dirSave + "rhox_HC__LLR_{:03d}_{:03d}__bLR_{:04d}__LHR_{:03d}_{:03d}__bHR_{:04d}__t_{:03d}.npz".format(
        int(np.floor(LLowRes)), int(1e3 * np.mod(LLowRes, 1)), binsLowRes, int(np.floor(LHighRes)), int(1e3 * np.mod(LHighRes, 1)), binsHighRes, t)
    
    if os.path.exists(pathSave) and not override:
        rhoxLoad = np.load(pathSave)
        rhoxLowRes, rhoxHighRes, rhoxTotal = [rhoxLoad[rhoStr] for rhoStr in rhoxLoad.files]
        return [rhoxLowRes, rhoxHighRes, rhoxTotal]
    
    xC = getHaloC(t)
    ptypeN = 3
    massArr = getMass(t)

    rhoxLowRes = np.zeros((binsLowRes, binsLowRes, binsLowRes))
    rhoxHighRes = np.zeros((binsHighRes, binsHighRes, binsHighRes))
    coords = getCoords(t)
    for pi in np.arange(0, ptypeN):
        ptype = pi + 1

        for ci, coordRaw in enumerate(coords[pi]):
            mi = massArr[pi][ci]
            coord = [coordRaw[i] - Lbox if coordRaw[i] > (xC[i] + Lbox/2) else coordRaw[i] for i in range(3)]
            try:
                if False in (np.abs(coord - xC) < LLowRes/2):
                    continue
                elif False in (np.abs(coord - xC) < LHighRes/2):
                    iLR, jLR, kLR = [int(np.floor((coord[i] - (xC[i] - LLowRes/2) + dxLowRes) / dxLowRes)) for i in range(3)]
                    rhoxLowRes[iLR, jLR, kLR] += mi / (dxLowRes ** 3)
                else:
                    iHR, jHR, kHR = [int(np.floor((coord[i] - (xC[i] - LHighRes/2)) / dxHighRes)) for i in range(3)]
                    rhoxHighRes[iHR, jHR, kHR] += mi / (dxHighRes ** 3)
            except IndexError:
                pass

    rhoxTotal = rhoxLowRes.copy()
    rhoxHighResAvg = np.asarray([rhoxHighRes[i, j, k] for i in range(binsHighRes) for j in range(binsHighRes) for k in range(binsHighRes)]).sum() / binsHighRes**3
    rhoxTotal[int((binsLowRes - 1) / 2), int((binsLowRes - 1) / 2), int((binsLowRes - 1) / 2)] = rhoxHighResAvg
    
    rhoxLowRes *= rhoScale * 1e10
    rhoxHighRes *= rhoScale * 1e10
    rhoxTotal *= rhoScale * 1e10
    np.savez(pathSave, rhoxLowRes, rhoxHighRes, rhoxTotal)
    
    return([rhoxLowRes, rhoxHighRes, rhoxTotal])

def rhoxHCInstantiate():
    print("Instantiating Density Matrices:")

    T0 = time.time()
    for t in range(tmax):
        if np.mod(t, 10) == 0:
            print("   Processing: {:03d} / {:03d} (Time Elapsed: {}, Time Estimated Remaining: {})".format(t, tmax, toMS(time.time() - T0), toMS((tmax / (t + 1)) * (time.time() - T0))))
        catch1 = getRhoxHC(t, override=True)
    print("Finished! Took {} total".format(toMS(time.time() - T0)), end="\n\n")

# %%
# rhoxHCInstantiate()
haloCArr = np.asarray([getHaloC(t) for t in range(tmax)])
np.save(simDir + "haloCArr_Sim_500v.npy", haloCArr)


