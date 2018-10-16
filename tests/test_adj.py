from lprec import lpTransform
import numpy as np
import struct

N = 512
Nproj = 300
Nslices = 1
filter_type = 'None'
cor = N/2
interp_type = 'cubic'
gpu = 0

f = np.float32(np.random.random([Nslices,N,N]))
R = np.float32(np.random.random([Nslices,Nproj,N]))


lp = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
lp.precompute(1)
lp.initcmem(1,gpu)

Rf = lp.fwd(f,gpu)
frec = lp.adj(R,gpu)
Rrec = lp.fwd(frec,gpu)


RR = lp.fwd(lp.adj(R,gpu),gpu)
print(np.sum(R*RR)/np.sum(RR*RR))



#dot product test
sum1 = sum(np.ndarray.flatten(Rrec)*np.ndarray.flatten(R))
sum2 = sum(np.ndarray.flatten(frec)*np.ndarray.flatten(frec))
print(np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2))

