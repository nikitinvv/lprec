from lprec import lpTransform
import numpy as np
import struct 
N = 512
Nproj = np.int(3*N/2)
Nslices = 1
filter_type = 'None'
cor = N/2
interp_type = 'cubic'

f = np.float32(np.random.random([Nslices,N,N]))
R = np.float32(np.random.random([Nslices,Nproj,N]))

clpthandle = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
clpthandle.precompute(1)
clpthandle.initcmem(1)

Rf = clpthandle.fwd(f)
frec = clpthandle.adj(R)


#dot product test
sum1 = sum(np.ndarray.flatten(Rf)*np.ndarray.flatten(R))
sum2 = sum(np.ndarray.flatten(frec)*np.ndarray.flatten(f))
print(np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2))
