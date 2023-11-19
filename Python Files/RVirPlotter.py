# %%


# %%
import numpy as np
import scipy as sp
from scipy import integrate, interpolate
import scipy.fft as fft
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

import h5py
import os
import time

# %% [markdown]
# Global Parameters

# %%
# Global Rocket Parameters
xi = 50
Gmu = 1e-12
GammaFrac = 0.1
vf = 0.3
nRockets = 1000

# Global Simulation Parameters
tmax = 100
L = 100
binS = 100
zi = 127
dx = L / binS
h = 0.7
rhoScale = 4.78e-20 # Mpc / Msun

# Global Physical Parameters
t0 = 4213 / h # Mpc / h
H0 = 0.0003333

# %%
# Cluster dirs
homeDir = "/cluster/tufts/hertzberglab/nshah14/"
sharedDir = "/cluster/tufts/hertzberglab/shared/Rockets/"
simName = "Sim_100v"
simDir = sharedDir + simName + "/"

# %% [markdown]
# Utility Functions

# %%
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
def getFg(t, pos, rhokL):
    x, y, z = list(map(int, np.floor(pos)))
    if x <= 0 or y <= 0 or z <= 0 or x >= binS-1 or y >= binS-1 or z >= binS-1:
        return np.zeros(3) 
    else:
        kArr = getkArr()
        phix = fft.fftn(rhokL).real

        try:
            Fg = (4*np.pi / (2*dx)) * np.asarray([phix[x + 1, y, z] - phix[x - 1, y, z], phix[x, y + 1, z] - phix[x, y - 1, z], phix[x, y, z + 1] - phix[x, y, z - 1]])
        except IndexError:
            Fg = np.zeros(3)
        return(Fg)

# %%
def getkArr(override=False):
    kArrPath = simDir + "kArr__bins_{:03d}.npy".format(binS)
    if os.path.exists(kArrPath) and not override:
        kArr = np.load(kArrPath)
        return(kArr)

    kArr = np.zeros((binS, binS, binS), dtype=float)
    halfBins = int(binS / 2)
    for i in range(binS):
        for j in range(binS):
            for k in range(binS):
                lx = - (4 * binS**2 / L**2) * np.sin((np.pi / binS) * (i - halfBins))**2
                ly = - (4 * binS**2 / L**2) * np.sin((np.pi / binS) * (j - halfBins))**2
                lz = - (4 * binS**2 / L**2) * np.sin((np.pi / binS) * (k - halfBins))**2
                if i == halfBins and j == halfBins and k == halfBins:
                    kArr[i, j, k] = 1
                else:
                    kArr[i, j, k] = lx + ly + lz
    
    np.save(kArrPath, kArr)
    return kArr

def getRhokLambda(t, kArr):
    rhox = getRhox(t)
    rhok = fft.ifftn(rhox)

    halfBins = int(binS / 2)
    rhokL = np.divide(rhok, kArr)
    rhokL[halfBins, halfBins, halfBins] = 0
    
    return(rhokL)

# %%
def getRhox(t, override=False):
    snapDir = simDir + "snapdir_{:03d}/".format(t)
    pathArr = np.asarray(os.listdir(snapDir))
    
    saveDir = simDir + simName + "_rhoX/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    savePath = saveDir + "rhoX_test_{:d}.npy".format(t)

    if os.path.exists(savePath) and not override:
        rhox = np.load(savePath)
        return(rhox)
    else:
        ptypeN = 3
        massArr = getMass()
        rhox = np.zeros((binS, binS, binS))
        for pi in np.arange(0, ptypeN):
            ptype = pi + 1
            mi = massArr[pi]
            
            for pathi in np.arange(0, pathArr.size):
                datGet = "/PartType{:d}/Coordinates".format(ptype)
                try:
                    coords = np.asarray(h5py.File(snapDir + pathArr[pathi], 'r')[datGet])
                except KeyError:
                    # print("   (Warning! Could not find ptype {:d} for snapshot t = {:d})".format(ptype, t))
                    pass

                for ci, coord in enumerate(coords):
                    i, j, k = list(map(lambda x: x - 1 if x == binS else x, map(int, np.floor(coord))))
                    rhox[i, j, k] += mi
        
        rhox *= rhoScale * 1e10
        np.save(savePath, rhox)
        return(rhox)

def rhoInstantiate(redo=[]):
    t0 = time.time()
    override = False
    for t in range(0, tmax):
        catch = getRhox(t, override=False)
        if np.mod(t, 10) == 0:
            print("   Processing t = {}/{} (Time Elapsed: {}, Estimated Time Remaining: {})".format(t + 1, tmax, toMS(time.time() - t0), toMS((tmax / (t + 1) - 1) * (time.time() - t0))))
    
    print("Finished Instantiating Density Matrices! Took: {}\n".format(toMS(time.time() - t0)))

# %%
def getMass():
    tFix = 50
    massTab = np.zeros(3)

    snapDir = simDir + "snapdir_{:03d}/".format(tFix)
    path1 = snapDir + "snapshot_{:03d}.0.hdf5".format(tFix)
    fil = h5py.File(path1, 'r')

    ptypeN = 3
    for pi in np.arange(0, ptypeN):
        ptype = pi + 1
        datGet = "/PartType{:d}/Masses".format(ptype)
        massTab[pi] = np.asarray(fil[datGet])[0]

    return massTab

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

# %%
def getHaloC(t):
    haloFinalDir = simDir + "snapdir_{:03d}/".format(t)
    pathArr = np.asarray(os.listdir(haloFinalDir))
    haloFinalCoords = np.empty((0, 3))

    for pathi in np.arange(0, pathArr.size):
        datGet = "/PartType1/Coordinates"
        try:
            coords = np.asarray(h5py.File(haloFinalDir + pathArr[pathi], 'r')[datGet])
        except KeyError:
            continue
        haloFinalCoords = np.concatenate([haloFinalCoords, coords], axis=0)

    haloC = np.asarray([np.mean(haloFinalCoords[:, i]) for i in range(3)])
    return(haloC)

def getRVir(t):
    snapDir = simDir + "groups_{:03d}/".format(t)
    file = h5py.File(snapDir + "fof_subhalo_tab_{:03d}.0.hdf5".format(t), 'r')

    try:
        RVir200 = np.asarray(file["/Group/Group_R_Crit200"])[0]
        RVir500 = np.asarray(file["/Group/Group_R_Crit500"])[0]
    except KeyError:
        print("Couldn't Find RVir for: t = {:03d}".format(t))
        RVir200 = -1
        RVir500 = -1
    return RVir200, RVir500

# %% [markdown]
# Get RVir

# %%
RVirArr = np.asarray([getRVir(t) for t in range(tmax)])
np.save(simDir + "RVirArr.npy", RVirArr)


