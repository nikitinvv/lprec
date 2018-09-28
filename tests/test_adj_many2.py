from lprec import lpTransform
import numpy as np
import struct
from timing import *

N = 512
Nproj = 3*N/2
Nslices = 8
filter_type = 'None'
cor = N/2

fid = open('./data/f', 'rb')
f = np.float32(np.reshape(struct.unpack(N*N*'f', fid.read(N*N*4)), [1, N, N]))
fa = np.zeros([Nslices, N, N], dtype=np.float32)
for k in range(0, Nslices):
	fa[k, :, :] = f*(k+1)

fid = open('./data/R', 'rb')
R = np.float32(np.reshape(struct.unpack(
	Nproj*N*'f', fid.read(Nproj*N*4)), [1, N, Nproj]))
Ra = np.zeros([Nslices, N, Nproj], dtype=np.float32)
for k in range(0, Nslices):
	Ra[k, :, :] = R

tic()

clpthandle = lpTransform.lpTransform(N, Nproj, Nslices, filter_type,cor,'linear')
clpthandle.precompute(1)
clpthandle.initcmem(1)
toc()

for k in range(0,3):
	tic()
	Rf = clpthandle.fwd(fa)
	toc()
for k in range(0,3):
	tic()
	frec = clpthandle.adj(Ra)
	toc()
Rrec = clpthandle.fwd(frec)


#dot product test
sum1 = sum(np.ndarray.flatten(Rrec)*np.ndarray.flatten(Ra))
sum2 = sum(np.ndarray.flatten(frec)*np.ndarray.flatten(frec))
print np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2)

