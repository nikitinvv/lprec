from lprec import lpTransform
from lprec import iterative
import numpy as np
import struct 

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
    recon = np.zeros([Nslices,N,N],dtype="float32")+1e-3
    recon = iterative.tv(lp,recon,tomo,num_iter,reg_par)


if __name__ == "__main__": main()