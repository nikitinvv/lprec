from lprecmods import lpTransform
from lprecmods.timing import *
import matplotlib.pyplot as plt
import numpy as np
import struct

N = 2016
Nproj = 299
Nslices = 4
filter_type = 'hamming'
pad = True
cor = N/2

fid = open('./data/Rfoam', 'rb')
R = np.float32(np.reshape(struct.unpack(
	Nproj*N*'f', fid.read(Nproj*N*4)), [1, N, Nproj]))
Ra = np.float32(np.zeros([Nslices, N, Nproj]))

for k in range(0, Nslices):
	Ra[k, :, :] = R[0, :, :]

clpthandle = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, pad)
clpthandle.precompute()
clpthandle.initcmem()

for k in range(0, 3):
	tic()
	frec = clpthandle.adj(Ra, cor)
	toc()

fid = open('frec2', 'wb')
frec[1, :, :].tofile(fid)
fid.close()

plt.subplot(1, 2, 1)
plt.imshow(R[0, :, :])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(frec[0, :, :])
plt.colorbar()
plt.show()
