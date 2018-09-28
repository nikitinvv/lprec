from lprec import lpTransform
import matplotlib.pyplot as plt
import numpy as np
import struct

N = 2016
Nproj = 1800
Nslices = 1
filter_type = 'hamming'
cor = N/2+4
interp_type = 'cubic'

fid = open('./data/Rimp', 'rb')
R = np.float32(np.reshape(struct.unpack(
    Nproj*N*'f', fid.read(Nproj*N*4)), [Nslices, N, Nproj]))
R = R.swapaxes(1,2)

clpthandle = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
clpthandle.precompute(0)
clpthandle.initcmem(0)

frec = clpthandle.adj(R)


plt.subplot(1, 2, 1)
plt.imshow(R[0, :, :])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(frec[0, :, :])
plt.colorbar()
plt.show()
