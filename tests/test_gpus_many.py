from lprec import lpTransform
from lprec import lpmethods

import numpy as np
import struct 
import cupy as cp
from timing import tic,toc
import concurrent.futures as cf


def lpmultigpu(lp,lpmethod,recon,tomo,num_iter,reg_par,gpu,Nssimgpu):
    """
    Reconstruction Nspergpu slices where Nssimgpu slices are reconstructed simultaneously on 1 GPU
    """

    for k in range(0,int(np.ceil(recon.shape[0]/Nssimgpu))):
        ids = range(k*Nssimgpu, min(recon.shape[0], (k+1)*Nssimgpu))
        recon[ids] = lpmethod(lp, recon[ids], tomo[ids], num_iter, reg_par, gpu)

    return recon

def test_gpus_many():
    N = 512
    Nproj = np.int(3*N/2)
    Ns = 32
    filter_type = 'None'
    cor = N/2
    interp_type = 'cubic'

    #init random arrays
    R = np.float32(np.sin(np.arange(0,Ns*Nproj*N)/float(Ns*Nproj)))
    R = np.reshape(R,[Ns,Nproj,N])


    #input parameters
    tomo = R
    reg_par = 0.001#*np.max(tomo)
    num_iter = 100
    recon = np.zeros([Ns,N,N],dtype="float32")+1e-3
    method = "tv"
    gpu_list=[0,1]
     # list of available methods for reconstruction
    lpmethods_list = {
                'fbp': lpmethods.fbp,
                'grad': lpmethods.grad,
                'cg': lpmethods.cg,
                'tv': lpmethods.tv,
                'em': lpmethods.em
                }

    ngpus = len(gpu_list)
    # total number of slices for processing by 1 gpu
    Nspergpu = int(np.ceil(Ns/float(ngpus)))
    # number of slices for simultaneous processing by 1 gpu 
    # (depends on gpu memory size, chosen for gpus with >= 4GB memory)
    Nssimgpu = min(int(pow(2, 24)/float(N*N)), Nspergpu)

    # class lprec
    lp = lpTransform.lpTransform(N, Nproj, Nssimgpu, filter_type, cor, interp_type)
    lp.precompute(method!='fbp')# if not fbp, precompute for the forward transform 

    # execute reconstruction on a thread per GPU
    jobs = [None]*ngpus
    with cf.ThreadPoolExecutor(ngpus) as e:
        for igpu in range(0,ngpus):
            gpu = gpu_list[igpu]
            lp.initcmem(method!='fbp',gpu)# if not fbp, allocate memory for the forward transform arrays
            ids = range(igpu*Nspergpu, min(Ns,(igpu+1)*Nspergpu))
            jobs[igpu]=e.submit(lpmultigpu, lp, lpmethods_list[method], recon[ids], tomo[ids], num_iter, reg_par, gpu, Nssimgpu)

    #collect results
    for igpu in range(0,ngpus):
       ids = range(igpu*Nspergpu, min(Ns,(igpu+1)*Nspergpu))
       recon[ids]=jobs[igpu].result()
    
    norm = np.linalg.norm(recon)
    print(norm)
    return norm


if __name__ == "__main__": 
    test_gpus_many()

