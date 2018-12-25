from lprec import lpTransform
from lprec import lpmethods
import matplotlib.pyplot as plt
import numpy as np
import struct


def test_imp():

    N = 2016
    Nproj = 1800
    Ns = 1
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
               'em': 0.5,
               'tv': 0.001,
               }

    lp = lpTransform.lpTransform(N, Nproj, Ns, filter_type, cor, interp_type)
    lp.precompute(1)
    lp.initcmem(1, gpu)

    fid = open('./tests/data/Rimp', 'rb')
    tomo = np.float32(np.reshape(struct.unpack(
        Nproj*N*'f', fid.read(Nproj*N*4)), [Ns, N, Nproj])).swapaxes(1, 2)

    # rec
    fRfbp = np.zeros([Ns, N, N], dtype="float32")
    fRfbp = lpmethods.fbp(
        lp, fRfbp, tomo, num_iter['fbp'], reg_par['fbp'], gpu)
    fRgrad = np.zeros([Ns, N, N], dtype="float32")
    fRgrad = lpmethods.grad(
        lp, fRgrad, tomo, num_iter['grad'], reg_par['grad'], gpu)
    fRcg = np.zeros([Ns, N, N], dtype="float32")
    fRcg = lpmethods.cg(lp, fRcg, tomo, num_iter['cg'], reg_par['cg'], gpu)
    fRem = np.zeros([Ns, N, N], dtype="float32")+1e-3
    fRem = lpmethods.em(lp, fRem, tomo, num_iter['em'], reg_par['em'], gpu)
    fRtv = np.zeros([Ns, N, N], dtype="float32")
    fRtv = lpmethods.tv(lp, fRtv, tomo, num_iter['tv'], reg_par['tv'], gpu)

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
    test_imp()
