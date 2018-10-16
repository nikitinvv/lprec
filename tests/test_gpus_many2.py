from lprec import lpTransform
import numpy as np
import struct 
import cupy as cp
from timing import tic,toc
import concurrent.futures as cf
from lprec import iterative

#@profile
def multigpu(lp,recon,tomo,num_iter,reg_par,gpu,Nssimgpu):
	for k in range(0,int(np.ceil(recon.shape[0]/Nssimgpu))):
		ids = range(k*Nssimgpu, min(recon.shape[0], (k+1)*Nssimgpu))
		print([gpu,ids])
		recon[ids] = iterative.tv(lp, recon[ids], tomo[ids], num_iter, reg_par, gpu)
	return recon

def main():
    N = 1024
    Nproj = np.int(3*N/2)
    Ns = 1024
    filter_type = 'None'
    cor = N/2
    interp_type = 'cubic'

    #init random arrays
    #R = np.float32(np.reshape(np.sin(np.arange(1,Ns*Nproj*N+1)/np.float32(Ns*Nproj*N)),[Ns,Nproj,N]))
    R = np.float32(np.random.random([Ns,Nproj,N]))

    gpu_list=[0,1,2,3]
    ngpus = len(gpu_list)

    #init parameters for the iterative scheme
    tomo = R
    reg_par = 0.001#*np.max(tomo)

    num_iter = 100
    recon = np.zeros([Ns,N,N],dtype="float32")+1e-3
   
    tic()    
    Nspergpu = int(np.ceil(Ns/float(ngpus)))
    Nssimgpu = min(int(pow(2, 26)/float(N*N)), Nspergpu)
    # class lprec
    lp = lpTransform.lpTransform(N, Nproj, Nssimgpu, filter_type, cor, interp_type)
    lp.precompute(1)
    toc()

    tic()

    jobs=[None]*ngpus
    # execute recon on a thread per GPU
    with cf.ThreadPoolExecutor(ngpus) as e:
        for igpu in range(0,ngpus):
            gpu = gpu_list[igpu]
            lp.initcmem(1,gpu)
            ids = range(igpu*Nspergpu, min(Ns,(igpu+1)*Nspergpu))
            jobs[igpu]=e.submit(multigpu, lp, recon[ids], tomo[ids], num_iter, reg_par, gpu, Nssimgpu)

    #collect results
    for igpu in range(0,ngpus):
       ids = range(igpu*Nspergpu, min(Ns,(igpu+1)*Nspergpu))
       recon[ids]=jobs[igpu].result()
    toc()
    recon1=recon
    print(np.linalg.norm(recon))

    gpu_list=[0]
    ngpus = len(gpu_list)

    recon = np.zeros([Ns,N,N],dtype="float32")+1e-3
   
    Nspergpu = int(np.ceil(Ns/float(ngpus)))
    Nssimgpu = min(int(pow(2, 25)/float(N*N)), Nspergpu)
    # class lprec
    lp = lpTransform.lpTransform(N, Nproj, Nssimgpu, filter_type, cor, interp_type)
    lp.precompute(1)

    tic()

    jobs=[None]*ngpus
    # execute recon on a thread per GPU
    with cf.ThreadPoolExecutor(ngpus) as e:
        for igpu in range(0,ngpus):
            gpu = gpu_list[igpu]
            lp.initcmem(1,gpu)
            ids = range(igpu*Nspergpu, min(Ns,(igpu+1)*Nspergpu))
            jobs[igpu]=e.submit(multigpu, lp, recon[ids], tomo[ids], num_iter, reg_par, gpu, Nssimgpu)

    #collect results
    for igpu in range(0,ngpus):
       ids = range(igpu*Nspergpu, min(Ns,(igpu+1)*Nspergpu))
       recon[ids]=jobs[igpu].result()
    toc()

    recon2=recon
    print(np.linalg.norm(recon))

    gpu_list=[0,1,2,3]
    ngpus = len(gpu_list)

    Nspergpu = int(np.ceil(Ns/float(ngpus)))
    Nssimgpu = min(int(pow(2, 25)/float(N*N)), Nspergpu)
 
    for igpu in range(0,ngpus):
       ids = range(igpu*Nspergpu, min(Ns,(igpu+1)*Nspergpu))
       print(ids)
       print(np.linalg.norm(recon1[ids]-recon2[ids]))

if __name__ == "__main__": main()

