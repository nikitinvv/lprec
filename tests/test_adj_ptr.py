from lprec import lpTransform
import numpy as np
import struct
import cupy as cp


def test_adj_ptr():
    N = 512
    Nproj = np.int(3*N/2)
    Nslices = 1
    filter_type = 'None'
    cor = N/2
    interp_type = 'cubic'
    gpu = 0

    # init random arrays
    f = np.float32(np.random.random([Nslices, N, N]))
    R = np.float32(np.random.random([Nslices, Nproj, N]))

    # copy arrays to gpu
    fg = cp.array(f)
    Rg = cp.array(R)
    # allocate gpu memory for results
    Rfg = cp.zeros([Nslices, Nproj, N], dtype="float32")
    fRg = cp.zeros([Nslices, N, N], dtype="float32")

    # class lprec
    lp = lpTransform.lpTransform(
        N, Nproj, Nslices, filter_type, cor, interp_type)
    lp.precompute(1)
    lp.initcmem(1, gpu)

    # compute with gpu pointers
    lp.fwdp(Rfg, fg, gpu)
    lp.adjp(fRg, Rg, gpu)

    # self adjoint test
    sum1 = sum(np.float64(np.ndarray.flatten(Rfg.get())*np.ndarray.flatten(Rg.get())))
    sum2 = sum(np.float64(np.ndarray.flatten(fRg.get())*np.ndarray.flatten(fg.get())))
    err0 = np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2)
    print(err0)

    return err0


if __name__ == '__main__':
    test_adj_ptr()
