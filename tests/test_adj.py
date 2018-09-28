from lprec import lpTransform
import matplotlib.pyplot as plt
import numpy as np
import struct

N = 512
Nproj = 3*N/2
Nslices = 1
filter_type = 'None'
cor = N/2
interp_type = 'cubic'

fid = open('./data/f', 'rb')
f = np.float32(np.reshape(struct.unpack(N*N*'f', fid.read(N*N*4)), [Nslices, N, N]))

fid = open('./data/R', 'rb')
R = np.float32(np.reshape(struct.unpack(
    Nproj*N*'f', fid.read(Nproj*N*4)), [Nslices, Nproj, N]))

clpthandle = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
clpthandle.precompute(1)
clpthandle.initcmem(1)

Rf = clpthandle.fwd(f)
frec = clpthandle.adj(R)
Rrec = clpthandle.fwd(frec)


#dot product test
sum1 = sum(np.ndarray.flatten(Rrec)*np.ndarray.flatten(R))
sum2 = sum(np.ndarray.flatten(frec)*np.ndarray.flatten(frec))
print np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2)

plt.subplot(2, 3, 1)
plt.imshow(f[0, :, :])
plt.colorbar()
plt.subplot(2, 3, 2)
plt.imshow(frec[0, :, :])
plt.colorbar()
plt.subplot(2, 3, 3)
plt.imshow(frec[0, :, :]-f[0, :, :])
plt.colorbar()
plt.subplot(2, 3, 4)
plt.imshow(Rrec[0, :, :])
plt.colorbar()
plt.subplot(2, 3, 5)
plt.imshow(Rf[0, :, :])
plt.colorbar()
plt.subplot(2, 3, 6)
plt.imshow(Rrec[0, :, :]-Rf[0, :, :])
plt.colorbar()

plt.show()
