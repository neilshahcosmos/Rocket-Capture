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

haloCArr = np.asarray([getHaloC(t) for t in range(tmax)])

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
    
    haloC = haloCArr[t]
    ptypeN = 3
    massArr = getMass(t)

    rhoxLowRes = np.zeros((binsLowRes, binsLowRes, binsLowRes))
    rhoxHighRes = np.zeros((binsHighRes, binsHighRes, binsHighRes))
    coords = getCoords(t)
    for pi in np.arange(0, ptypeN):
        ptype = pi + 1

        for ci, coordRaw in enumerate(coords[pi]):
            mi = massArr[pi][ci]
            coord = [coordRaw[i] - Lbox if coordRaw[i] > (haloC[i] + Lbox/2) else coordRaw[i] for i in range(3)]
            try:
                if False in (np.abs(coord - haloC) < LLowRes/2):
                    continue
                elif False in (np.abs(coord - haloC) < LHighRes/2):
                    iLR, jLR, kLR = [int(np.floor((coord[i] - (haloC[i] - LLowRes/2) + dxLowRes) / dxLowRes)) for i in range(3)]
                    rhoxLowRes[iLR, jLR, kLR] += mi / (dxLowRes ** 3)
                else:
                    iHR, jHR, kHR = [int(np.floor((coord[i] - (haloC[i] - LHighRes/2)) / dxHighRes)) for i in range(3)]
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

# def rhoxHCInstantiate():
#     print("Instantiating Density Matrices:")

#     T0 = time.time()
#     for t in range(tmax):
#         if np.mod(t, 10) == 0:
#             print("   Processing: {:03d} / {:03d} (Time Elapsed: {}, Time Estimated Remaining: {})".format(t, tmax, toMS(time.time() - T0), toMS((tmax / (t + 1)) * (time.time() - T0))))
#         catch1 = getRhoxHC(t, override=True)
#     print("Finished! Took {} total".format(toMS(time.time() - T0)), end="\n\n")

# %%
def getkArr(L, bins, override=False):
    kArrDir = simDir + "kArrs/"
    if not os.path.exists(kArrDir):
        os.makedirs(kArrDir)
    kArrPath = kArrDir + "kArr__L_{:03d}__bins_{:03d}.npy".format(L, bins)
    if os.path.exists(kArrPath) and not override:
        kArr = np.load(kArrPath)
        return(kArr)

    kArr = np.zeros((bins, bins, bins), dtype=float)
    halfBins = 0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                lx = - (4 * bins**2 / L**2) * np.sin((np.pi / bins) * (i - halfBins))**2
                ly = - (4 * bins**2 / L**2) * np.sin((np.pi / bins) * (j - halfBins))**2
                lz = - (4 * bins**2 / L**2) * np.sin((np.pi / bins) * (k - halfBins))**2
                if i == halfBins and j == halfBins and k == halfBins:
                    kArr[i, j, k] = 1
                else:
                    kArr[i, j, k] = lx + ly + lz
    
    np.save(kArrPath, kArr)
    return kArr

def getRhokLambda(t, res):
    rhoxLowRes, rhoxHighRes, rhoxTotal = getRhoxHC(t, override=False)
    if not 'kArrLowRes' in globals():
        global kArrLowRes
        kArrLowRes = getkArr(LLowRes, binsLowRes)
    if not 'kArrHighRes' in globals():
        global kArrHighRes
        kArrHighRes = getkArr(LHighRes, binsHighRes)
    
    if res == "Low":
        kArr = kArrLowRes
        dxi = dxLowRes
        halfBins = int(binsLowRes / 2)
        rhox = rhoxLowRes

        rhokLowRes = fft.ifftn(rhoxLowRes)
        rhokL = np.divide(rhokLowRes, kArrLowRes)
    elif res == "High":
        kArr = kArrHighRes
        dxi = dxHighRes
        halfBins = int(binsHighRes / 2)
        rhox = rhoxHighRes

        rhokHighRes = fft.ifftn(rhoxHighRes)
        rhokL = np.divide(rhokHighRes, kArrHighRes)
    elif res == "Total":
        kArr = kArrLowRes
        dxi = dxLowRes
        halfBins = int(binsLowRes / 2)
        rhox = rhoxTotal

        rhokTotal = fft.ifftn(rhoxTotal)
        rhokL = np.divide(rhokTotal, kArrLowRes)
    else:
        print("ERROR! You are trying to get a kArr for a resolution which doesn't exist")
        quit()
    
    rhokL[0, 0, 0] = 0
    return(rhokL)

# %%
def getPhix(t, rhokLOverride=False, phixOverride=False):
    if (not 'rhokLLowRes' in globals()) or rhokLOverride:
        global rhokLLowRes
        rhokLLowRes = getRhokLambda(t, "Low")
    if (not 'rhokLHighRes' in globals()) or rhokLOverride:
        global rhokLHighRes
        rhokLHighRes = getRhokLambda(t, "High")
    if (not 'rhokLTotal' in globals()) or rhokLOverride:
        global rhokLTotal
        rhokLTotal = getRhokLambda(t, "Total")
    
    if (not 'phixLowRes' in globals()) or phixOverride:
        global phixLowRes
        phixLowRes = fft.fftn(rhokLLowRes).real
    if (not 'phixHighRes' in globals()) or phixOverride:
        global phixHighRes
        phixHighRes = fft.fftn(rhokLHighRes).real
    if (not 'phixTotal' in globals()) or phixOverride:
        global phixTotal
        phixTotal = fft.fftn(rhokLTotal).real

    return [phixLowRes, phixHighRes, phixTotal]

def getFg(t, pos, rhokLOverride=False, phixOverride=False):
    haloC = haloCArr[t]
    # pos = np.asarray([posRaw[i] - Lbox if posRaw[i] > haloC[i] + Lbox/2 else posRaw[i] for i in range(3)])
    phixLowRes, phixHighRes, phixTotal = getPhix(t, rhokLOverride, phixOverride)

    if False in (np.abs(pos - haloC) < LLowRes/2):
        return np.zeros(3)
    elif False in (np.abs(pos - haloC) < LHighRes/2):
        iTot, jTot, kTot = [int(np.floor((pos[i] - (haloC[i] - LLowRes/2) + dxLowRes) / dxLowRes)) for i in range(3)]
        try:
            FgTot = - (4 * np.pi / (2 * dxLowRes)) * np.asarray(
                [phixTotal[iTot + 1, jTot, kTot] - phixTotal[iTot - 1, jTot, kTot], phixTotal[iTot, jTot + 1, kTot] - phixTotal[iTot, jTot - 1, kTot], phixTotal[iTot, jTot, kTot + 1] - phixTotal[iTot, jTot, kTot - 1]])
        except IndexError:
            FgTot = np.zeros(3)
        Fg = FgTot
    else:
        iLR, jLR, kLR = [int(np.floor((pos[i] - (haloC[i] - LLowRes/2) + dxLowRes) / dxLowRes)) for i in range(3)]
        iHR, jHR, kHR = [int(np.floor((pos[i] - (haloC[i] - LHighRes/2)) / dxHighRes)) for i in range(3)]
        iLRp1, jLRp1, kLRp1 = [i if i == binsHighRes - 1 else i + 1 for i in [iLR, jLR, kLR]]
        iLRm1, jLRm1, kLRm1 = [i if i == 0 else i - 1 for i in [iLR, jLR, kLR]]
        iHRp1, jHRp1, kHRp1 = [i if i == binsHighRes - 1 else i + 1 for i in [iHR, jHR, kHR]]
        iHRm1, jHRm1, kHRm1 = [i if i == 0 else i - 1 for i in [iHR, jHR, kHR]]

        try:
            FgLowRes = - (4 * np.pi / (2 * dxLowRes)) * np.asarray(
                [phixLowRes[iLRp1, jLR, kLR] - phixLowRes[iLRm1, jLR, kLR], phixLowRes[iLR, jLRp1, kLR] - phixLowRes[iLR, jLRm1, kLR], phixLowRes[iLR, jLR, kLRp1] - phixLowRes[iLR, jLR, kLRm1]])
        except IndexError:
            FgLowRes = np.zeros(3)
        try:
            FgHighRes = - (4 * np.pi / (2 * dxHighRes)) * np.asarray(
                [phixHighRes[iHRp1, jHR, kHR] - phixHighRes[iHRm1, jHR, kHR], phixHighRes[iHR, jHRp1, kHR] - phixHighRes[iHR, jHRm1, kHR], phixHighRes[iHR, jHR, kHRp1] - phixHighRes[iHR, jHR, kHRm1]])
        except IndexError:
            FgHighRes = np.zeros(3)
        Fg = FgLowRes + FgHighRes
    
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
t0G = time.time()
print("Hello! Now Beginning Halo Centered Rocket Evolution")
print("This run has Global Parameters: xi = {}, {} Rockets, Gmu = {}, GammaFrac = {}, vf = {}".format(xi, nRockets, Gmu, GammaFrac, vf), end="\n\n")

# print("Now Beginning Instantiation of rho matrices. Here I will calculate binned density matrices from the raw simulation data and save it to a file.")
# rhoxHCInstantiate()

saveHaloCFlag = False
savePhixFlag = False

if saveHaloCFlag:
    print("Saving Halo Centers\n")
    haloCArr = np.asarray([getHaloC(t) for t in range(tmax)])
    np.save(simDir + "haloCArr.npy", haloCArr)

if savePhixFlag:
    print("Saving Phi Arrays\n")
    dirSave = simDir + "phix_HaloCentered{}rhox_HC__LLR_{:03d}_{:03d}__bLR_{:04d}__LHR_{:03d}_{:03d}__bHR_{:04d}{}".format(
            Div, int(np.floor(LLowRes)), int(1e3 * np.mod(LLowRes, 1)), binsLowRes, int(np.floor(LHighRes)), int(1e3 * np.mod(LHighRes, 1)), binsHighRes, Div)
    if not os.path.exists(dirSave):
        os.makedirs(dirSave)
    for t in range(tmax):
        pathSave = dirSave + "phix_HC__LLR_{:03d}_{:03d}__bLR_{:04d}__LHR_{:03d}_{:03d}__bHR_{:04d}__t_{:03d}.npz".format(
            int(np.floor(LLowRes)), int(1e3 * np.mod(LLowRes, 1)), binsLowRes, int(np.floor(LHighRes)), int(1e3 * np.mod(LHighRes, 1)), binsHighRes, t)
        phit = getPhix(t)
        np.savez(pathSave, phit[0], phit[1], phit[2])



# %%
print("Now Processing Rocket Evolution!:")

rocketSaveDir = simDir + "RocketTrajs_HC{}".format(Div)
if not os.path.exists(rocketSaveDir):
    os.makedirs(rocketSaveDir)
rocketSavePath = rocketSaveDir + "RT_HC__Sim_500v__N_{:06d}__Gmu_1e-{:02d}__xi_{:03d}__NoFRocket.npy".format(
    nRockets, int(-np.floor(np.log(Gmu) / np.log(10))), xi)

xArr = np.zeros((nRockets, 3, tmax))
vArr = np.zeros((nRockets, 3, tmax))
FgArr = np.zeros((nRockets, 3, tmax))
FRocketArr = np.zeros((nRockets, 3))

haloC0 = getHaloC(0)
xArr[:, :, 0] = np.asarray([haloC0 + 0.1 * LHighRes * (2*np.random.rand(3)-1) for i in range(0, nRockets)])
vArr[:, :, 0] = 0 * 2.6 * (1 + zi)**2 * np.sqrt(xi * Gmu) * vf * np.asarray([v / np.linalg.norm(v) for v in (2*np.random.rand(nRockets, 3)-1)])
FRocketArr = (H0 * GammaFrac / xi) * np.asarray([u / np.linalg.norm(u) for u in (2*np.random.rand(nRockets, 3)-1)])
FgArr[:, :, 0] = np.asarray([getFg(0, xArr[i, :, 0]) for i in range(nRockets)])

tArr, aArr, HArr = getHubbleEvol()
dtArr = np.diff(tArr)
kArrLowRes = getkArr(LLowRes, binsLowRes, override=True)
kArrHighRes = getkArr(LHighRes, binsHighRes, override=True)

t0 = time.time()
for ti in np.arange(0, tmax - 1):
    t = tArr[ti]
    dti = dtArr[ti]
    ai = aArr[ti]
    Hi = HArr[ti]
    
    rhokLLowRes = getRhokLambda(ti, "Low")
    rhokLHighRes = getRhokLambda(ti, "High")
    rhokLTotal = getRhokLambda(ti, "Total")

    haloC = getHaloC(ti)
    xOffset = haloC - Lbox/2

    for ri in range(nRockets):
        # FRocketi = FRocketArr[ri, :]
        FRocketi = np.asarray([0, 0, 0])
        Fgi = 10 * getFg(ti, xArr[ri, :, ti])
        FgArr[ri, :, ti + 1] = Fgi
        vArr[ri, :, ti + 1] = vArr[ri, :, ti] + dti * ((Fgi + FRocketi) / ai - 2 * Hi * vArr[ri, :, ti])
        xArr[ri, :, ti + 1] = xArr[ri, :, ti] + dti * vArr[ri, :, ti]
    
    del phixLowRes
    del phixHighRes
    del phixTotal
    
    if np.mod(ti + 1, 10) == 0:
        print("   Processing t = {}/{} (Time Elapsed: {}, Estimated Time Remaining: {})".format(ti + 1, tmax, toMS(time.time() - t0), toMS((tmax / (ti + 1) - 1) * (time.time() - t0))))

print("Finished Processing Rocket Evolution! Time Taken: {}\n".format(toMS(time.time() - t0)))

np.save(rocketSavePath, np.asarray([xArr, vArr]))
np.save(simDir + "FgArr.npy", FgArr)
print("Rocket Trajectories Successfully Exported. \nWe are now done, thank you! Total Time Taken: {}\n".format(toMS(time.time() - t0G)))

# %% [markdown]
# To Do:
# Fix time looping so dtArr can go till last index


