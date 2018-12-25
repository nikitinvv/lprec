from lprec import lpTransform
from lprec import lpmethods

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import struct


def test_lpmethods():
    N = 512
    Nproj = int(3*N/4)
    Ns = 8
    filter_type = 'None'
    cor = N/2
    interp_type = 'cubic'
    gpu = 0

    num_iter = {'fbp': 0,
                'grad': 64,
                'cg': 16,
                'em': 64,
                'tv': 256,
                }
    reg_par = {'fbp': 0,
               'grad': -1,
               'cg': 0,
               'em': 0.01,
               'tv': 0.001,
               }

    fid = open('tests/data/f', 'rb')
    f = np.float32(np.reshape(struct.unpack(
        N*N*'f', fid.read(N*N*4)), [1, N, N]))
    fa = np.zeros([Ns, N, N], dtype=np.float32)
    for k in range(0, Ns):
        fa[k, :, :] = f

    # class lprec
    lp = lpTransform.lpTransform(N, Nproj, Ns, filter_type, cor, interp_type)
    lp.precompute(1)
    lp.initcmem(1, gpu)

    # init data
    fag = cp.array(fa)
    Rag = cp.zeros([Ns, Nproj, N], dtype="float32")
    lp.fwdp(Rag, fag, gpu)
    Ra = Rag

    # rec
    fRfbp = np.zeros([Ns, N, N], dtype="float32")
    fRfbp = lpmethods.fbp(lp, fRfbp, Ra, num_iter['fbp'], reg_par['fbp'], gpu)
    fRgrad = np.zeros([Ns, N, N], dtype="float32")
    fRgrad = lpmethods.grad(
        lp, fRgrad, Ra, num_iter['grad'], reg_par['grad'], gpu)
    fRcg = np.zeros([Ns, N, N], dtype="float32")
    fRcg = lpmethods.cg(lp, fRcg, Ra, num_iter['cg'], reg_par['cg'], gpu)
    fRem = np.zeros([Ns, N, N], dtype="float32")+1e-3
    fRem = lpmethods.em(lp, fRem, Ra, num_iter['em'], reg_par['em'], gpu)
    fRtv = np.zeros([Ns, N, N], dtype="float32")
    fRtv = lpmethods.tv(lp, fRtv, Ra, num_iter['tv'], reg_par['tv'], gpu)

    norm0 = np.linalg.norm(np.float64(fRfbp))
    norm1 = np.linalg.norm(np.float64(fRgrad))
    norm2 = np.linalg.norm(np.float64(fRcg))
    norm3 = np.linalg.norm(np.float64(fRem))
    norm4 = np.linalg.norm(np.float64(fRtv))
    # plt.subplot(2, 3, 1)
    # plt.imshow(fRfbp[-1, :, :])
    # plt.colorbar()
    # plt.subplot(2, 3, 2)
    # plt.imshow(fRgrad[-1, :, :])
    # plt.colorbar()
    # plt.subplot(2, 3, 3)
    # plt.imshow(fRcg[-1, :, :])
    # plt.colorbar()
    # plt.subplot(2, 3, 4)
    # plt.imshow(fRem[-1, :, :])
    # plt.colorbar()
    # plt.subplot(2, 3, 5)
    # plt.imshow(fRtv[-1, :, :])
    # plt.colorbar()
    # plt.show()
    print([norm0, norm1, norm2, norm3, norm4])

    return [norm0, norm1, norm2, norm3, norm4]


if __name__ == '__main__':
    test_lpmethods()
