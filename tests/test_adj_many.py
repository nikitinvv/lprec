from lprec import lpTransform
from lprec import lpmethods

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import struct

N = 512
Nproj = int(3*N/2)
Nslices = 8
filter_type = 'None'
cor = N/2
interp_type = 'cubic'
gpu = 0

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


# copy arrays to gpu
fg = cp.array(fa)
Rg = cp.array(Ra)

# allocate gpu memory for results
Rfg = cp.zeros([Nslices,Nproj,N],dtype="float32")
fRg = cp.zeros([Nslices,N,N],dtype="float32")
RfRg = cp.zeros([Nslices,Nproj,N],dtype="float32")


# class lprec
lp = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
lp.precompute(1)
lp.initcmem(1,gpu)

# compute with gpu pointers
lp.fwdp(Rfg,fg,gpu)
lp.adjp(fRg,Rg,gpu)
lp.fwdp(RfRg,fRg,gpu)

print("Adjoint test (should be < 0.01)")
#self adjoint test
sum1 = sum(np.ndarray.flatten(Rfg.get())*np.ndarray.flatten(Rg.get()))
sum2 = sum(np.ndarray.flatten(fRg.get())*np.ndarray.flatten(fg.get()))
print(np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2))

Rf = Rfg.get()
fR = fRg.get()
RfR = RfRg.get()


#dot product test
sum1 = sum(np.ndarray.flatten(Rf)*np.ndarray.flatten(Ra))
sum2 = sum(np.ndarray.flatten(fR)*np.ndarray.flatten(fR))
print(np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2))

plt.subplot(2, 3, 1)
plt.imshow(fa[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 2)
plt.imshow(fR[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 3)
plt.imshow(fR[-1, :, :]-fa[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 4)
plt.imshow(RfR[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 5)
plt.imshow(Rf[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 6)
plt.imshow(RfR[-1, :, :]-Rf[-1, :, :])
plt.colorbar()

plt.show()
