import lprec.initsgl as initsgl
import lprec.initsfwd as initsfwd
import lprec.initsadj as initsadj
import lprec.lpRgpu as lpRgpu
import numpy as np

class lpTransform:
	def __init__(self,N,Nproj,Nslices,filter_type,cor,interp_type):
		self.N = N
		self.Nslices = Nslices
		self.filter_type = filter_type	
		self.cor = cor
		self.interp_type = interp_type
		#size after zero padding in the angle direction (for nondense sampling rate)
		self.osangles = int(max(round(3.0*N/2.0/Nproj),1))
		self.Nproj = self.osangles*Nproj
		#size after zero padding in radial direction
		self.Npad = np.int(N+abs(N/2-cor)*2)

	def precompute(self,flg):		
		#precompute parameters for the lp method
		Pgl,self.glpars = initsgl.create_gl(self.Npad,self.Nproj,self.Nslices,self.cor,self.interp_type)
		if(flg):
			Pfwd,self.fwdparsi,self.fwdparamsf = initsfwd.create_fwd(Pgl)
		Padj,self.adjparsi,self.adjparamsf = initsadj.create_adj(Pgl,self.filter_type)
	
	def initcmem(self,flg):
		#init memory in C, read data from files
		self.clphandle = lpRgpu.lpRgpu(self.glpars)
		if(flg):
			self.clphandle.initFwd(self.fwdparsi,self.fwdparamsf)
		self.clphandle.initAdj(self.adjparsi,self.adjparamsf)	

	def fwd(self,f):
		Ros = np.zeros([f.shape[0],self.Nproj,self.N],dtype = 'float32')
		self.clphandle.execFwdMany(Ros,f)
		R = Ros[:,0::self.osangles,:]
		return R

	def adj(self,R):
		Ros = np.zeros([R.shape[0],self.Nproj,self.N],dtype = 'float32')
		Ros[:,0::self.osangles,:] = R*self.osangles
		f = np.zeros([R.shape[0],self.N,self.N],dtype = 'float32')
		self.clphandle.execAdjMany(f,Ros)
		return f


	



