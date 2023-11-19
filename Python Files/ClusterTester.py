# %%
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
L = 1
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
    kArrPath = simDir + "kArr__L_{:03d}__bins_{:03d}.npy".format(L, binS)
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
    rhox = getRhoxZoom(t)
    rhok = fft.ifftn(rhox)

    halfBins = int(binS / 2)
    rhokL = np.divide(rhok, kArr)
    rhokL[halfBins, halfBins, halfBins] = 0
    
    return(rhokL)

# %%
def getRhoxZoom(t, L=L, dx=dx, override=False):
    snapDir = simDir + "snapdir_{:03d}/".format(t)
    pathArr = np.asarray(os.listdir(snapDir))
    
    saveDir = simDir + simName + "_rhoXZoom__L_{:03d}Mpc__dx_{:03d}kPc/".format(L, int(1000 * dx))
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    savePath = saveDir + "rhoXZoom__L_{:03d}Mpc__dx_{:03d}kPc__t_{:03d}.npy".format(L, int(1000 * dx), t)

    if os.path.exists(savePath) and not override:
        rhoxZoom = np.load(savePath)
        return(rhoxZoom)
    else:
        ptypeN = 3
        massArr = getMass()
        haloC = getHaloC(tmax)
        binS = int(L / dx)
        rhoxZoom = np.zeros((binS, binS, binS))
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
                    i, j, k = list(map(int, np.floor((coord - haloC + L/2) / dx)))
                    if i < 0 or i > binS - 1 or j < 0 or j > binS - 1 or k < 0 or k > binS - 1:
                        continue
                    try:
                        rhoxZoom[i, j, k] += mi
                    except IndexError:
                        print("Indexing Error! i, j, k = {}".format([i, j, k]))
                        return -1
        rhoxZoom *= rhoScale * 1e10
        np.save(savePath, rhoxZoom)
        return(rhoxZoom)

def rhoInstantiate(redo=[]):
    t0 = time.time()
    override = False
    for t in range(0, tmax):
        catch = getRhoxZoom(t, L=L, dx=dx, override=False)
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

# %%
def getCoords(t):
    snapDir = simDir + "snapdir_{:03d}/".format(t)
    pathArr = np.asarray(os.listdir(snapDir))
    
    ptypeN = 3
    coordsArr = [np.empty((0, 3)) for i in range(ptypeN)]
    for pi in np.arange(0, ptypeN):
        ptype = pi + 1
        
        for pathi in np.arange(0, pathArr.size):
            datGet = "/PartType{:d}/Coordinates".format(ptype)
            try:
                coords = np.asarray(h5py.File(snapDir + pathArr[pathi], 'r')[datGet])
            except KeyError:
                # print("   (Warning! Could not find ptype {:d} for snapshot t = {:d})".format(ptype, t))
                continue

            coordsArr[pi] = np.concatenate([coordsArr[pi], coords], axis=0)
    
    return(coordsArr)

def getHaloCoords(t):
    groupDir = simDir + "groups_{:03d}/".format(t)
    snapDir = simDir + "snapdir_{:03d}/".format(t)
    groupPaths = np.asarray(os.listdir(groupDir))
    snapPaths = np.asarray(os.listdir(snapDir))

    ptype = 1
    coordsArr = np.empty((0, 3), dtype=float)

    for i, pathi in enumerate(snapPaths):
        datGet = "/PartType{:d}/Coordinates".format(ptype)
        try:
            coords = np.asarray(h5py.File(snapDir + pathi, 'r')[datGet])
            coordsArr = np.concatenate([coordsArr, coords], axis=0)
        except KeyError:
            # print("   (Warning! Could not find ptype {:d} for snapshot t = {:d})".format(ptype, t))
            pass
    
    haloCoords = np.empty((1, 3))
    for i, pathi in enumerate(groupPaths):
        gfile = h5py.File(groupDir + pathi, 'r')
        try:
            haloInit = int(np.asarray(gfile["/Group/GroupOffsetType/"])[0, 0])
            haloN = int(np.asarray(gfile["/Group/GroupLen/"])[0])
            haloCoords = coordsArr[haloInit:(haloInit + haloN + 1), :]
            break
        except KeyError:
            pass
    
    return(haloCoords)

# %% [markdown]
# PType Plotting

# %%
ptypeN = 3
colorArr = np.asarray(["k", "g", "r"])

saveDir = simDir + "AllPtypesPlts/"
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

coords0 = getCoords(0)
xmin, xmax, ymin, ymax = np.asarray([[np.min(coords0[i][:, 0]), np.max(coords0[i][:, 0]), np.min(coords0[i][:, 1]), np.max(coords0[i][:, 1])] for i in range(3)]).T
limRatio = np.asarray([1.5, 1.5, 1.2])
pltLims = np.asarray([limRatio[i] * np.asarray([
xmin[i] - 0.5 * (xmax[i] + xmin[i]), 
xmax[i] - 0.5 * (xmax[i] + xmin[i]),
ymin[i] - 0.5 * (ymax[i] + ymin[i]),
ymax[i] - 0.5 * (ymax[i] + ymin[i])]) for i in range(3)])

T0 = time.time()
for t in range(tmax):
    print("Processing: {:03d}/{:03d} (Time Elapsed: {}, Estimated Time Remaining: {})".format(t, tmax, toMS(time.time() - T0), toMS((tmax / (t + 1) - 1) * (time.time() - T0))))
    coordsArr = getCoords(t)

    fig1, axs1 = plt.subplots(1, 3, figsize=(26, 8), dpi=100)
    for pi in range(ptypeN):
        xc = np.mean(coordsArr[pi][:, 0])
        yc = np.mean(coordsArr[pi][:, 1])

        axi = axs1[pi]
        axi.scatter(coordsArr[pi][:, 0] - xc, coordsArr[pi][:, 1] - yc, c=colorArr[pi], s=0.1)
        axi.title.set_text("ptype = {:d}, t = {:03d}".format(pi + 1, t))
        axi.set_xlim([pltLims[pi, 0], pltLims[pi, 1]])
        axi.set_ylim([pltLims[pi, 2], pltLims[pi, 3]])
    plt.savefig(saveDir + "AllPtypesPlts_Separated__t_{:03d}".format(t))
    plt.close()

    fig2, axs2 = plt.subplots(figsize=(8, 8), dpi=100)
    for pi in range(ptypeN):
        plt.scatter(coordsArr[pi][:, 0], coordsArr[pi][:, 1], c=colorArr[pi], s=0.1, zorder=(3 - pi), label="ptype = {}".format(pi + 1))
    plt.title("t = {:03d}".format(t))
    plt.savefig(saveDir + "AllPtypesPlts_Overlayed__t_{:03d}".format(t))
    plt.close()

# %% [markdown]
# Group ID Testing

# %%
# haloCArr = np.zeros((tmax, 3))
# for t in range(tmax):
#     haloCoords = getHaloCoords(t)
#     if haloCoords.shape[0] > 1:
#         haloCArr[t, :] = np.asarray([np.mean(haloCoords[:, j]) for j in range(3)])
#     else:
#         haloCArr[t, :] = np.asarray([-1, -1, -1])
    
#     if (np.mod(t, 10) == 0):
#         print("Processing: {}/{}".format(t, tmax))

# np.save(simDir + "haloCenters.npy", haloCArr)


