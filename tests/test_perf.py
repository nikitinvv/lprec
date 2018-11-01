from lprec import lpTransform
from lprec import lpmethods

import numpy as np
import struct
import cupy as cp
from timing import tic, toc
import concurrent.futures as cf
import threading
from itertools import repeat
from functools import partial
from timing import tic,toc

def lpmultigpu(lp, lpmethod, recon, tomo, num_iter, reg_par, gpu_list, ids):
    """
    Reconstruction Nssimgpu slices simultaneously on 1 GPU
    """
    # take gpu number with respect to the current thread
    gpu = gpu_list[int(threading.current_thread().name.split("_", 1)[1])]

    # reconstruct
    print([gpu,ids])
    recon[ids] = lpmethod(lp, recon[ids], tomo[ids], num_iter, reg_par, gpu)
    return recon[ids]


def test_gpus_many_map():
    N = 2048
    Nproj = N
    Ns = 2048
    filter_type = 'None'
    cor = N//2
    interp_type = 'cubic'

    # init random array
    R = np.float32(np.sin(np.arange(0, Ns*Nproj*N)/float(Ns*Nproj)))
    R = np.reshape(R, [Ns, Nproj, N])
    # input parameters
    tomo = R
    reg_par = 0.001  # *np.max(tomo)
    num_iter = 100
    recon = np.zeros([Ns, N, N], dtype="float32")+1e-3
    method = "em"
    gpu_list = [0, 1, 2, 3]
    # list of available methods for reconstruction
    lpmethods_list = {
        'fbp': lpmethods.fbp,
        'grad': lpmethods.grad,
        'cg': lpmethods.cg,
        'tv': lpmethods.tv,
        'em': lpmethods.em
    }

    ngpus = len(gpu_list)
    # number of slices for simultaneous processing by 1 gpu
    # (depends on gpu memory size, chosen for gpus with >= 4GB memory)
    Nssimgpu = min(int(pow(2, 26)/float(N*N)), int(np.ceil(Ns/float(ngpus))))

    tic()
    # class lprec
    lp = lpTransform.lpTransform(
        N, Nproj, Nssimgpu, filter_type, cor, interp_type)
    # if not fbp, precompute for the forward transform
    lp.precompute(method != 'fbp')
    print("Init time" % toc())
    # list of slices sets for simultaneous processing b gpus
    ids_list = [None]*int(np.ceil(Ns/float(Nssimgpu)))
    for k in range(0, len(ids_list)):
        ids_list[k] = range(k*Nssimgpu, min(Ns, (k+1)*Nssimgpu))

    tic()
    # init memory for each gpu
    for igpu in range(0, ngpus):
        gpu = gpu_list[igpu]
        # if not fbp, allocate memory for the forward transform arrays
        lp.initcmem(method != 'fbp', gpu)

    # run reconstruciton on many gpus
    with cf.ThreadPoolExecutor(ngpus) as e:
        shift = 0
        for reconi in e.map(partial(lpmultigpu, lp, lpmethods_list[method], recon, tomo, num_iter, reg_par, gpu_list), ids_list):
            recon[np.arange(0, reconi.shape[0])+shift] = reconi
            shift += reconi.shape[0]
    print("Rec time %f" % toc())
    #norm = np.linalg.norm(recon)
    #print(norm)
    #return norm


if __name__ == "__main__":
    test_gpus_many_map()
