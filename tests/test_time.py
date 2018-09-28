from lprec import lpTransform
import numpy as np
import struct
from timing import *

N = 2048
Nproj = 3*N/2
Nslices = 64
filter_type = 'None'
cor = N/2

fa=np.float32(np.random.random([Nslices,N,N]))
Ra=np.float32(np.random.random([Nslices,N,Nproj]))

print()

tic()
Nslices0 = min(int(pow(2,25)/float(N*N)),Nslices)
clpthandle = lpTransform.lpTransform(N, Nproj, Nslices0, filter_type,cor,'cubic')
clpthandle.precompute(0)
clpthandle.initcmem(0)
toc()
tic()
for k in range(0,int(np.ceil(Nslices/float(Nslices0)))):
    ids = range(k*Nslices0,min(Nslices,(k+1)*Nslices0))
    print(ids)
    fa[ids] = clpthandle.adj(Ra[ids])
t=toc()
print(t/Nslices)


