from lprec import lpTransform
from lprec import lpmethods
import matplotlib.pyplot as plt
import numpy as np
import struct

N = 2016
Nproj = 299
Ns = 1
filter_type = 'None'
cor = N/2
interp_type = 'cubic'
reg_par = 1e-1
gpu = 3

num_iter ={'fbp': 1,
			'grad': 32,
			'cg': 8,
			'em': 16,
			'tv': 64,
}

lp = lpTransform.lpTransform(N, Nproj, Ns, filter_type, cor, interp_type)
lp.precompute(1)
lp.initcmem(1,gpu)

fid = open('./tests/data/Rfoam', 'rb')
tomo = np.float32(np.reshape(struct.unpack(
    Nproj*N*'f', fid.read(Nproj*N*4)), [Ns, N, Nproj])).swapaxes(1,2)

#rec
fRfbp = np.zeros([Ns,N,N],dtype="float32")+1e-3
fRfbp = lpmethods.fbp(lp,fRfbp, tomo, num_iter['fbp'], reg_par, gpu)
fRgrad = np.zeros([Ns,N,N],dtype="float32")+1e-3
fRgrad = lpmethods.grad(lp,fRgrad, tomo, num_iter['grad'], reg_par, gpu)
fRcg = np.zeros([Ns,N,N],dtype="float32")+1e-3
fRcg = lpmethods.cg(lp,fRcg, tomo, num_iter['cg'], reg_par, gpu)
fRem = np.zeros([Ns,N,N],dtype="float32")+1e-3
fRem = lpmethods.em(lp,fRem, tomo, num_iter['em'], reg_par, gpu)
fRtv = np.zeros([Ns,N,N],dtype="float32")+1e-3
fRtv = lpmethods.tv(lp,fRtv, tomo, num_iter['tv'], reg_par, gpu)

plt.subplot(2, 3, 1)
plt.imshow(fRfbp[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 2)
plt.imshow(fRgrad[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 3)
plt.imshow(fRcg[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 4)
plt.imshow(fRem[-1, :, :])
plt.colorbar()
plt.subplot(2, 3, 5)
plt.imshow(fRtv[-1, :, :])
plt.colorbar()
plt.show()






