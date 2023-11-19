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

# %%
# Global Rocket Parameters
xi = 50
Gmu = 1e-12
GammaFrac = 0.1
vf = 0.3
nRockets = 10

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
    groupDir = simDir + "groups_{:03d}/".format(t)
    file = h5py.File(groupDir + "fof_subhalo_tab_{:03d}.0.hdf5".format(t), 'r')

    try:
        RVir200 = np.asarray(file["/Group/Group_R_Crit200"])[0]
        RVir500 = np.asarray(file["/Group/Group_R_Crit500"])[0]
    except KeyError:
        RVir200 = -1
        RVir500 = -1
    return RVir200, RVir500

# %%
def getCoords(t, ptype=1):
    snapDir = simDir + "snapdir_{:03d}/".format(t)
    snapPaths = np.asarray(os.listdir(snapDir))
    
    ptypeN = 3
    pi = ptype - 1
    coordsArr = np.empty((0, 3), dtype=float)
    
    for i, pathi in enumerate(snapPaths):
        datGet = "/PartType{:d}/Coordinates".format(ptype)
        try:
            coords = np.asarray(h5py.File(snapDir + pathi, 'r')[datGet])
            coordsArr = np.concatenate([coordsArr, coords], axis=0)
        except KeyError:
            # print("   (Warning! Could not find ptype {:d} for snapshot t = {:d})".format(ptype, t))
            pass
    
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
    
    haloCoords = -1
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

# %%
def getHaloPlt(t, axesLims="Fixed"):
    saveDir = simDir + "HaloEvolPlots_{}Lims/".format(axesLims)
    savePath = saveDir + "HaloEvol__t_{:03d}.png".format(t)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    fig = plt.figure(1, figsize=(8, 8), dpi=100)
    ax = plt.gca()

    RVir200, RVir500 = getRVir(t)
    coords = getHaloCoords(t)
    haloC = np.asarray([np.mean(coords[:, i]) for i in range(3)])

    # nx, binsx = np.histogram(coords[:, 0], bins=20)
    # ny, binsy = np.histogram(coords[:, 1], bins=20)
    # try:
    #     xmin, xmax = [np.min(np.asarray([binsx[i] for i in np.arange(0, binsx.size - 1) if nx[i] > 10])), np.max(np.asarray([binsx[i] for i in np.arange(1, binsx.size) if nx[i - 1] > 10]))]
    #     ymin, ymax = [np.min(np.asarray([binsy[i] for i in np.arange(0, binsy.size - 1) if ny[i] > 10])), np.max(np.asarray([binsy[i] for i in np.arange(1, binsy.size) if ny[i - 1] > 10]))]
    # except ValueError:
    #     xmin, xmax = [np.min(coords[:, 0]), np.max(coords[:, 0])]
    #     ymin, ymax = [np.min(coords[:, 1]), np.max(coords[:, 1])]
    # pltLims = 1.0 * (np.asarray([xmin, xmax, ymin, ymax]) - np.asarray([haloC[0], haloC[0], haloC[1], haloC[1]])) + np.asarray([haloC[0], haloC[0], haloC[1], haloC[1]])
    if axesLims == "Dynamic":
        pltLims = 1.2 * np.asarray([-RVir200, RVir200, -RVir200, RVir200]) + np.asarray([haloC[0], haloC[0], haloC[1], haloC[1]])
    if axesLims == "Fixed":
        RVirFinal, _ = getRVir(tmax)
        pltLims = 1.2 * np.asarray([-RVirFinal, RVirFinal, -RVirFinal, RVirFinal]) + np.asarray([haloC[0], haloC[0], haloC[1], haloC[1]])

    plt.scatter(coords[:, 0], coords[:, 1], c="k", s=0.2)
    if RVir200 != -1:
        ax.add_patch(plt.Circle((haloC[0], haloC[1]), RVir200, fill=False, color="cyan", linewidth=3, zorder=10))
        ax.add_patch(plt.Circle((haloC[0], haloC[1]), RVir500, fill=False, color="red", linewidth=2, zorder=10))

    plt.title("Halo Evolution with Dynamic Axes Limits (t = {:03d})".format(t))
    plt.xlim((pltLims[0], pltLims[1]))
    plt.ylim((pltLims[2], pltLims[3]))

    plt.savefig(savePath)
    plt.close()
    return 1

def getHaloPltAxesFixed(t):
    saveDir = simDir + "HaloEvolPlots_FixedLims/"
    savePath = saveDir + "HaloEvol__t_{:03d}.png".format(t)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    fig = plt.figure(1, figsize=(8, 8), dpi=100)
    ax = plt.gca()

    coords = getHaloCoords(t)
    haloC = np.asarray([np.mean(coords[:, i]) for i in range(3)])

    RVirFinal200, RVirFinal500 = getRVir(tmax)
    pltLims = 1.2 * np.asarray([-RVirFinal200, RVirFinal200, -RVirFinal200, RVirFinal200])

    plt.scatter(coords[:, 0] - haloC[0], coords[:, 1] - haloC[1], c="k", s=0.2)
    RVir200, RVir500 = getRVir(t)
    if RVir200 != -1:
        ax.add_patch(plt.Circle((0, 0), RVir200, fill=False, color="cyan", linewidth=3, zorder=10))
        ax.add_patch(plt.Circle((0, 0), RVir500, fill=False, color="red", linewidth=2, zorder=10))

    plt.title("Halo Evolution with Fixed Axes Limits (t = {:03d})".format(t))
    plt.xlim((pltLims[0], pltLims[1]))
    plt.ylim((pltLims[2], pltLims[3]))

    plt.savefig(savePath)
    plt.close()
    return 1

# %%
for t in range(tmax):
    RVir200, RVir500 = getRVir(t)
    if RVir200 == -1:
        continue
    print("Processing: {:d} / {:d}".format(t, tmax))
    debug = getHaloPltAxesFixed(t)
    if debug == -1:
        break


