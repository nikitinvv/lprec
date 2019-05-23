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


bgpus = [] # busy gpus
def lpmultigpu(lp, lpmethod, recon, tomo, num_iter, reg_par, gpu_list, ids):
    """
    Reconstruction Nssimgpu slices simultaneously on 1 GPU
    """
    # take gpu number with respect to the current thread
    global bgpus
    lock = threading.Lock()
    lock.acquire() # will block if lock is already held
    print('lock')
    for k in range(len(gpu_list)):
        if bgpus[k]==0:
            bgpus[k] = 1
            gpu_id = k
            print(bgpus)
    print('release')
    lock.release()        
    gpu = gpu_list[gpu_id]
    #gpu = gpu_list[int(threading.current_thread().name.split("_", 1)[1])]

    # reconstruct
    recon[ids] = lpmethod(lp, recon[ids], tomo[ids], num_iter, reg_par, gpu)
    bgpus[gpu_id] = 0
    print([gpu,ids])

    return recon[ids]


def test_gpus_many_map():
    N = 512
    Nproj = np.int(3*N/2)
    Ns = 32
    filter_type = 'None'
    cor = int(N/2)-3
    interp_type = 'cubic'

    # init random array
    R = np.float32(np.sin(np.arange(0, Ns*Nproj*N)/float(Ns*Nproj)))
    R = np.reshape(R, [Ns, Nproj, N])
    # input parameters
    tomo = R
    reg_par = -1  # *np.max(tomo)
    num_iter = 100
    recon = np.zeros([Ns, N, N], dtype="float32")+1e-3
    method = "grad"
    gpu_list = [0,1]
    
    # list of available methods for reconstruction
    lpmethods_list = {
        'fbp': lpmethods.fbp,
        'grad': lpmethods.grad,
        'cg': lpmethods.cg,
        'tv': lpmethods.tv,
        'em': lpmethods.em
    }
    try:
        cp.cuda.Device(1).use()
    except:
        gpu_list = [0]        
    ngpus = len(gpu_list)
    global bgpus
    bgpus = np.zeros(ngpus)
    # number of slices for simultaneous processing by 1 gpu
    # (depends on gpu memory size, chosen for gpus with >= 4GB memory)
    Nssimgpu = min(int(pow(2, 24)/float(N*N)), int(np.ceil(Ns/float(ngpus))))

    # class lprec
    lp = lpTransform.lpTransform(
        N, Nproj, Nssimgpu, filter_type, cor, interp_type)
    # if not fbp, precompute for the forward transform
    lp.precompute(method != 'fbp')

    # list of slices sets for simultaneous processing b gpus
    ids_list = [None]*int(np.ceil(Ns/float(Nssimgpu)))
    for k in range(0, len(ids_list)):
        ids_list[k] = range(k*Nssimgpu, min(Ns, (k+1)*Nssimgpu))

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

    norm = np.linalg.norm(np.float64(recon))
    print(norm)
    return norm


if __name__ == "__main__":
    test_gpus_many_map()
