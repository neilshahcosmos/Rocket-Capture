# %% [markdown]
# <h1>Loop Runner - Single Xi</h1>

# %% [markdown]
# <b>Description:</b> Here, I'm evolving loops for the given simulation. I'm employing a new technique here of utilizing two different binned density matrices, one which is high resolution and localized on the halo, and a low resolution one which covers the rest of the box. I'm also shifting the coordinates so the halo is always in the center.

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
tmax = 100
zi = 127
h = 0.7
rhoScale = 4.78e-20 # Mpc / Msun

Lbox = 100
LLowRes, LHighRes = [100, 1]
dxLowRes, dxHighRes = [1, 0.01]
binsLowRes, binsHighRes = [int(LLowRes / dxLowRes), int(LHighRes / dxHighRes)]

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
    simName = "Sim_100v"
    simDir = homeDir + simName + "/"
    Div = "/"
elif env == "Local":
    # Cluster dirs
    homeDir = "C:\\Users\\Neil\\Documents\\PhD\\Rocket Force\\"
    simName = "Sim_100v"
    simDir = homeDir + simName + "\\"
    Div = "\\"

print("Working in the {} environment".format(env))

# %% [markdown]
# Utility Functions

# %%
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
    dirSave = simDir + "rhox_HaloCentered{}rhox_HC__LRTesting{}".format(Div, Div)
    if not os.path.exists(dirSave):
        os.makedirs(dirSave)
    pathSave = dirSave + "rhox_HC__LRTesting__t_{:03d}.npz".format(t)
    
    if os.path.exists(pathSave) and not override:
        rhoxLoad = np.load(pathSave)
        rhoxLowRes = rhoxLoad[rhoxLoad.files[0]]
        return rhoxLowRes
    
    ptypeN = 3
    massArr = getMass(t)
    tArr, aArr, HArr = getHubbleEvol()
    dxPhys = aArr[t] * dxLowRes

    rhoxLowRes = np.zeros((binsLowRes, binsLowRes, binsLowRes))
    coords = getCoords(t)
    for pi in np.arange(0, ptypeN):
        for ci, coord in enumerate(coords[pi]):
            mi = massArr[pi][ci]
            try:
                iLR, jLR, kLR = [int(np.floor(coord[i] / dxLowRes)) for i in range(3)]
                rhoxLowRes[iLR, jLR, kLR] += mi / (dxPhys ** 3)
            except IndexError:
                pass
    
    rhoxLowRes *= rhoScale * 1e10
    np.savez(pathSave, rhoxLowRes)
    
    return(rhoxLowRes)

def rhoxHCInstantiate():
    print("Instantiating Density Matrices:")

    T0 = time.time()
    for t in range(tmax):
        if np.mod(t, 10) == 0:
            print("   Processing: {:03d} / {:03d} (Time Elapsed: {}, Time Estimated Remaining: {})".format(t, tmax, toMS(time.time() - T0), toMS((tmax / (t + 1)) * (time.time() - T0))))
        catch1 = getRhoxHC(t, override=True)
    print("Finished! Took {} total".format(toMS(time.time() - T0)), end="\n\n")

# %%
def getkArr(L, bins, t, override=False):
    kArrDir = simDir + "kArrs/"
    if not os.path.exists(kArrDir):
        os.makedirs(kArrDir)
    kArrPath = kArrDir + "kArr__L_{:03d}__bins_{:03d}.npy".format(L, bins)
    if os.path.exists(kArrPath) and not override:
        kArr = np.load(kArrPath)
        return(kArr)

    if "aArr" not in globals():
        global aArr
        _, aArr, _ = getHubbleEvol()
    Lphys = L * aArr[t]

    kArr = np.zeros((bins, bins, bins), dtype=float)
    halfBins = 0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                lx = - (4 * bins**2 / Lphys**2) * np.sin((np.pi / bins) * (i - halfBins))**2
                ly = - (4 * bins**2 / Lphys**2) * np.sin((np.pi / bins) * (j - halfBins))**2
                lz = - (4 * bins**2 / Lphys**2) * np.sin((np.pi / bins) * (k - halfBins))**2
                if i == halfBins and j == halfBins and k == halfBins:
                    kArr[i, j, k] = 1
                else:
                    kArr[i, j, k] = lx + ly + lz
    
    np.save(kArrPath, kArr)
    return kArr

def getRhokLambda(t):
    rhoxLowRes = getRhoxHC(t, override=False)
    if not 'kArrLowRes' in globals():
        global kArrLowRes
        kArrLowRes = getkArr(LLowRes, binsLowRes, t)

    rhokLowRes = fft.ifftn(rhoxLowRes)
    rhokL = np.divide(rhokLowRes, kArrLowRes)
    
    rhokL[0, 0, 0] = 0
    return(rhokL)

# %%
def getPhix(t, rhokLOverride=False, phixOverride=False):
    if (not 'rhokLLowRes' in globals()) or rhokLOverride:
        global rhokLLowRes
        rhokLLowRes = getRhokLambda(t)
    
    if (not 'phixLowRes' in globals()) or phixOverride:
        global phixLowRes
        phixLowRes = fft.fftn(rhokLLowRes).real

    return phixLowRes

def getFg(t, pos, rhokLOverride=False, phixOverride=False):
    boxC = np.asarray([Lbox / 2 for i in range(3)])
    # pos = np.asarray([posRaw[i] - Lbox if posRaw[i] > haloC[i] + Lbox/2 else posRaw[i] for i in range(3)])
    phixTotal = getPhix(t, rhokLOverride, phixOverride)
    if "aArr" not in globals():
        global aArr
        _, aArr, _ = getHubbleEvol()
    dxPhys = aArr[t] * dxLowRes

    iTot, jTot, kTot = [int(np.floor(pos[i] / dxLowRes)) for i in range(3)]
    try:
        FgTot = - (4 * np.pi / (2 * dxPhys)) * np.asarray(
            [phixTotal[iTot + 1, jTot, kTot] - phixTotal[iTot - 1, jTot, kTot], phixTotal[iTot, jTot + 1, kTot] - phixTotal[iTot, jTot - 1, kTot], phixTotal[iTot, jTot, kTot + 1] - phixTotal[iTot, jTot, kTot - 1]])
    except IndexError:
        FgTot = np.zeros(3)
    Fg = FgTot
    
    return(Fg)

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
def evenQ(n):
    if int(np.mod(n, 2)) == 0:
        return True
    return False

def getk(t):
    bins = binsLowRes
    Lphys = aArr[t] * Lbox
    nvalues = np.arange(- bins / 2, bins / 2) if evenQ(bins) else np.arange(-(bins - 1) / 2, (bins + 1) / 2)
    kspace = 2.0 * bins / Lphys * np.sin(np.pi * nvalues / bins)

    kx, ky, kz = np.meshgrid(kspace, kspace, kspace, sparse=True)

    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kz = np.fft.ifftshift(kz)

    kArr = kx ** 2 + ky ** 2 + kz ** 2
    kArr[0, 0, 0] = 1
    return kArr

# %%
t0G = time.time()
print("Hello! Now Beginning Halo Centered Rocket Evolution")
print("This run has Global Parameters: xi = {}, {} Rockets, Gmu = {}, GammaFrac = {}, vf = {}".format(xi, nRockets, Gmu, GammaFrac, vf), end="\n\n")

print("Now Processing Rocket Evolution!:")

rocketSaveDir = simDir + "RocketTrajs_HC{}".format(Div)
if not os.path.exists(rocketSaveDir):
    os.makedirs(rocketSaveDir)
rocketSavePath = rocketSaveDir + "RT_HC__LRTesting__NoFRocket.npy".format(
    nRockets, int(-np.floor(np.log(Gmu) / np.log(10))), xi)

xArr = np.zeros((nRockets, 3, tmax))
vArr = np.zeros((nRockets, 3, tmax))
FgArr = np.zeros((nRockets, 3, tmax))
FRocketArr = np.zeros((nRockets, 3))

haloC0 = getHaloC(0)
xArr[:, :, 0] = np.asarray([haloC0 + 0.1 * LHighRes * (2*np.random.rand(3)-1) for i in range(0, nRockets)])
vArr[:, :, 0] = 0 * 2.6 * (1 + zi)**2 * np.sqrt(xi * Gmu) * vf * np.asarray([v / np.linalg.norm(v) for v in (2*np.random.rand(nRockets, 3)-1)])
FRocketArr = (H0 * GammaFrac / xi) * np.asarray([u / np.linalg.norm(u) for u in (2*np.random.rand(nRockets, 3)-1)])

tArr, aArr, HArr = getHubbleEvol()
dtArr = np.diff(tArr)

for ti in range(0, tmax - 1):
    rho = getRhoxHC(ti)
    kArr = getk(ti)
    rhok = np.fft.ifft(rho)
    phik = -1 * np.divide(rhok, kArr)
    phik[0, 0, 0] = 0
    phi = 4 * np.pi * np.fft.fft(phik).astype(float)

    dti, ai, Hi = [dtArr[ti], aArr[ti], HArr[ti]]
    dxPhys = ai * 1

    for ri in range(nRockets):
        xi = xArr[ri, :, ti]
        iR, jR, kR = [int(np.floor(xi[i])) for i in range(3)]
        # FRocketi = FRocketArr[ri, :]
        FRocketi = np.asarray([0, 0, 0])

        Fgi = - (1 / (2 * dxPhys)) * np.asarray([
            phi[iR + 1, jR, kR] - phi[iR - 1, jR, kR],
            phi[iR, jR + 1, kR] - phi[iR, jR - 1, kR],
            phi[iR, jR, kR + 1] - phi[iR, jR, kR - 1]])

        FgArr[ri, :, ti + 1] = Fgi
        vArr[ri, :, ti + 1] = vArr[ri, :, ti] + dti * ((Fgi + FRocketi) / ai - 2 * Hi * vArr[ri, :, ti])
        xArr[ri, :, ti + 1] = xArr[ri, :, ti] + dti * vArr[ri, :, ti]

    if np.mod(ti + 1, 10) == 0:
        print("   Processing t = {}/{} (Time Elapsed: {}, Estimated Time Remaining: {})".format(ti + 1, tmax, toMS(time.time() - t0), toMS((tmax / (ti + 1) - 1) * (time.time() - t0))))

print("Finished Processing Rocket Evolution! Time Taken: {}\n".format(toMS(time.time() - t0)))

np.save(rocketSavePath, np.asarray([xArr, vArr]))
np.save(simDir + "FgArr.npy", FgArr)
print("Rocket Trajectories Successfully Exported. \nWe are now done, thank you! Total Time Taken: {}\n".format(toMS(time.time() - t0G)))

# %% [markdown]
# To Do:
# Fix time looping so dtArr can go till last index


