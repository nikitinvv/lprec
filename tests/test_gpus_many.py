from lprec import lpTransform
import numpy as np
import struct 
import cupy as cp
from timing import tic,toc
import concurrent.futures as cf
from lprec import iterative

#@profile
def main():
    N = 512
    Nproj = np.int(3*N/2)
    Nslices = 4
    filter_type = 'None'
    cor = N/2
    interp_type = 'cubic'

    #init random arrays
    f = np.float32(np.random.random([Nslices,N,N]))
    R = np.float32(np.random.random([Nslices,Nproj,N]))

    gpu_list=[4,6]
    ngpus = len(gpu_list)

    #init parameters for the iterative scheme
    reg_par = 1e-2
    tomo = R
    num_iter = 10
    recon = np.zeros([Nslices,N,N],dtype="float32")+1e-3
   
    tic()    
    Nslicespergpu = int(np.ceil(Nslices/float(ngpus)))
    Nslices0 = min(int(pow(2, 23)/float(N*N)), Nslicespergpu)
    # class lprec
    lp = lpTransform.lpTransform(N, Nproj, Nslices0, filter_type, cor, interp_type)
    lp.precompute(1)

    tic()

    # execute recon on a thread per GPU
    with cf.ThreadPoolExecutor(ngpus) as e:
        for igpu in range(0,ngpus):
            gpu = gpu_list[igpu]
            lp.initcmem(1,gpu)

            for k in range(0,int(np.ceil(Nslicespergpu/Nslices0))):
                ids = range(k*Nslices0+igpu*Nslicespergpu, min(Nslices,min(Nslicespergpu, (k+1)*Nslices0)+igpu*Nslicespergpu))
                print([gpu,ids])
                print(tomo[ids].shape)
                recon[ids] = iterative.em(lp, recon[ids], tomo[ids,:,:], num_iter, reg_par, gpu)
                #recon[ids] = e.submit(iterative.em, lp, recon[ids], tomo[ids], num_iter, reg_par, gpu).result()
    import time
    time.sleep(5)

    toc()
    print(np.linalg.norm(recon))

if __name__ == "__main__": main()
