import lprec.initsgl as initsgl
import lprec.initsfwd as initsfwd
import lprec.initsadj as initsadj
import lprec.lpRgpu as lpRgpu
import numpy as np
import cupy as cp
import time
import gc
class lpTransform:
    def __init__(self, N, Nproj, Nslices, filter_type, cor, interp_type):
        self.N = N
        self.Nslices = Nslices
        self.filter_type = filter_type
        self.cor = cor
        self.Nproj = Nproj
        self.interp_type = interp_type
        self.clphandle = [None]*16

    def precompute(self, flg):
        # precompute parameters for the lp method
        Pgl, self.glpars = initsgl.create_gl(
            self.N, self.Nproj, self.Nslices, self.cor, self.interp_type)
        if(flg): # parameters for the forward transform
            Pfwd, self.fwdparsi, self.fwdparamsf = initsfwd.create_fwd(Pgl)
        # parameters for the adjoint transform
        Padj, self.adjparsi, self.adjparamsf = initsadj.create_adj(
            Pgl, self.filter_type)
        # free GPU memory 
        Pgl = []            
        Pfwd = []            
        Padj = []                       
                             
        
    def initcmem(self, flg, gpu):
        # init memory in C (can be used by several gpus)        
        self.clphandle[gpu] = lpRgpu.lpRgpu(self.glpars.ctypes.data, gpu)

        if(flg):
            self.clphandle[gpu].initFwd(self.fwdparsi.ctypes.data, self.fwdparamsf.ctypes.data, gpu)
        self.clphandle[gpu].initAdj(self.adjparsi.ctypes.data, self.adjparamsf.ctypes.data, gpu)
    
    def fwdp(self, R, f, gpu):
        # Forward projection operator. Work with GPU pointers
        self.clphandle[gpu].execFwdManyPtr(
            R.data.ptr, f.data.ptr, f.shape[0], gpu)

    def adjp(self, f, R, gpu):
        # Forward projection operator. Work with GPU pointers
        self.clphandle[gpu].execAdjManyPtr(
            f.data.ptr, R.data.ptr, R.shape[0], gpu)
