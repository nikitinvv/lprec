from lprec import lpTransform
import numpy as np
import struct 
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

N = 512
Nproj = np.int(3*N/2)
Nslices = 1
filter_type = 'None'
cor = N/2
interp_type = 'cubic'

#init random arrays
f = np.float32(np.random.random([Nslices,N,N]))
R = np.float32(np.random.random([Nslices,Nproj,N]))

# copy arrays to gpu
fg = gpuarray.to_gpu(f)
Rg = gpuarray.to_gpu(R)

# allocate gpu memory for results
Rfg = gpuarray.GPUArray([Nslices,Nproj,N],dtype="float32")
fRg = gpuarray.GPUArray([Nslices,N,N],dtype="float32")

# class lprec
lp = lpTransform.lpTransform(N, Nproj, Nslices, filter_type, cor, interp_type)
lp.precompute(1)
lp.initcmem(1)

# compute in a standard way (with cpu-gpu data transfers)
Rf = lp.fwd(f)
fR = lp.adj(R)

# compute with gpu pointers
lp.fwdp(Rfg,fg)
lp.adjp(fRg,Rg)

#check the result
print("Differences for two approaches")
print(np.linalg.norm(Rfg.get()-Rf))
print(np.linalg.norm(fRg.get()-fR))


print("Adjoint test (should be < 0.01)")
#self adjoint test
sum1 = sum(np.ndarray.flatten(Rf)*np.ndarray.flatten(R))
sum2 = sum(np.ndarray.flatten(fR)*np.ndarray.flatten(f))
print(np.linalg.norm(sum1-sum2)/np.linalg.norm(sum2))

