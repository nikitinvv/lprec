from lprec import lpTransform
import matplotlib.pyplot as plt
import numpy as np
import struct

N = 512
Nproj = int(3*N/2)
Nslices = 8
filter_type = 'None'
cor = N/2
interp_type = 'cubic'

fid = open('tests/data/f', 'rb')
f = np.float32(np.reshape(struct.unpack(N*N*'f', fid.read(N*N*4)), [1, N, N]))
fa = np.zeros([Nslices, N, N], dtype=np.float32)
for k in range(0, Nslices):
	fa[k, :, :] = f*(Nslices-k)

fid = open('tests/data/R', 'rb')
R = np.float32(np.reshape(struct.unpack(
	Nproj*N*'f', fid.read(Nproj*N*4)), [1, Nproj, N]))
Ra = np.zeros([Nslices, Nproj, N], dtype=np.float32)
for k in range(0, Nslices):
	Ra[k, :, :] = R*(Nslices-k)


clpthandle = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
clpthandle.precompute(1)
clpthandle.initcmem(1)

Rf = clpthandle.fwd(fa)
frec = clpthandle.adj(Ra)
Rrec = clpthandle.fwd(frec)


#dot product test
sum1 = sum(np.ndarray.flatten(Rrec)*np.ndarray.flatten(Ra))
sum2 = sum(np.ndarray.flatten(frec)*np.ndarray.flatten(frec))
print(np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2))

plt.subplot(2, 3, 1)
plt.imshow(fa[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 2)
plt.imshow(frec[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 3)
plt.imshow(frec[-1, :, :]-fa[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 4)
plt.imshow(Rrec[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 5)
plt.imshow(Rf[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 6)
plt.imshow(Rrec[-1, :, :]-Rf[-1, :, :])
plt.colorbar()

plt.show()
