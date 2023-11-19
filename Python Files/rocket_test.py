# %%
import numpy as np
import datetime

# %%
def rhokoverlambda(rhok,dx,dy,dz):
    nx=rhok.shape[0]
    ny=rhok.shape[1]
    nz=rhok.shape[2]
    Lx = dx*nx
    Ly = dy*ny
    Lz = dz*nz
    rhokl=np.copy(rhok)
    k2s=np.zeros((nx,ny,nz))
    iG = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i<nx/2:
                    ni=i
                else:
                    ni=i-nx
                if j<ny/2:
                    nj=j
                else:
                    nj=j-nz
                if k<nz/2:
                    nk=k
                else:
                    nk=k-nz
                lambdax = - 4* nx**2 / (Lx**2) * (np.sin(np.pi*ni/nx))**2
                lambday = - 4* ny**2 / (Ly**2) * (np.sin(np.pi*nj/ny))**2
                lambdaz = - 4* nz**2 / (Lz**2) * (np.sin(np.pi*nk/nz))**2
                if i==0 and j==0 and k==0:
                    rhokl[i,j,k] = 0
                else:
                    rhokl[i,j,k]= rhok[i,j,k]/(lambdax+lambday+lambdaz)
                iG += 1
    # print(iG)
    return rhokl

# %%
dx = 1; n = 100;
testDat = np.random.rand(n, n, n) 

t0 = datetime.datetime.now()
rhokoverlambda(testDat, dx, dx, dx)
print("The test run took: {}s".format((datetime.datetime.now() - t0).total_seconds()))

# %%



