from lprec import lpTransform
import numpy as np
import struct 
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from timing import tic,toc

#@profile
def main():
    N = 1024
    Nproj = np.int(3*N/2)
    Nslices = 8
    filter_type = 'None'
    cor = N/2
    interp_type = 'cubic'

    #init random arrays
    f = np.float32(np.random.random([Nslices,N,N]))
    R = np.float32(np.random.random([Nslices,Nproj,N]))

    # class lprec
    lp = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
    lp.precompute(1)
    lp.initcmem(1)

    #init parameters for the iterative scheme
    reg_par = 1e-2
    tomo = R
    num_iter = 32
    recon0 = np.zeros([Nslices,N,N],dtype="float32")+1e-3

    # Approach 1 (with cpu - gpu transfers)
    recon = recon0
    eps = reg_par
    # R^*(ones)
    xi = lp.adj(tomo*0+1)
    # modification for avoiing division by 0
    xi = xi+eps*np.pi/2
    e = np.float32(np.max(tomo)*eps)
    tic()
    for i in range(0, num_iter):
        g = lp.fwd(recon)
        upd = lp.adj(tomo/(g+e))
        recon = recon*(upd/xi)
    print("  Approach 1 (with cpu - gpu transfers)")
    toc()
    recon1 = recon

    # Approach 2 (work with gpu pointers)
    recon = gpuarray.to_gpu(recon0)
    tomo = gpuarray.to_gpu(tomo)
    xi = gpuarray.GPUArray([Nslices,N,N],dtype="float32")
    g = gpuarray.GPUArray([Nslices,Nproj,N],dtype="float32")
    upd = gpuarray.GPUArray([Nslices,N,N],dtype="float32")

    # R^*(ones)
    lp.adjp(xi, tomo*0+1)
    # modification for avoiing division by 0
    xi = xi+eps*np.pi/2
    e = np.float32(np.max(tomo.get())*eps)

    tic()
    for i in range(0, num_iter):
        lp.fwdp(g,recon)
        #pycuda.driver.Context.synchronize() # the previous cuda call is asynchronous (use this for profiling)
        lp.adjp(upd,tomo/(g+e))
        #pycuda.driver.Context.synchronize() # the previous cuda call is asynchronous
        recon = recon*(upd/xi)
    print("  Approach 2 (work with gpu pointers)")
    toc()
    recon2 = recon

    print("  Difference in results:")
    print(np.linalg.norm(recon1-recon2.get()))

if __name__ == "__main__": main()