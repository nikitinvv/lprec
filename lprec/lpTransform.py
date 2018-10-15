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
		self.Nproj = Nproj
		self.interp_type = interp_type

	def precompute(self,flg):		
		#precompute parameters for the lp method
		Pgl,self.glpars = initsgl.create_gl(self.N,self.Nproj,self.Nslices,self.cor,self.interp_type)
		if(flg):
			Pfwd,self.fwdparsi,self.fwdparamsf = initsfwd.create_fwd(Pgl)
		Padj,self.adjparsi,self.adjparamsf = initsadj.create_adj(Pgl,self.filter_type)
	
	def initcmem(self,flg,gpu):
		#init memory in C (could be used by several gpus)
		self.clphandle = lpRgpu.lpRgpu(self.glpars,gpu)
		if(flg):
			self.clphandle.initFwd(self.fwdparsi,self.fwdparamsf,gpu)
		self.clphandle.initAdj(self.adjparsi,self.adjparamsf,gpu)	

	def fwd(self,f):
		# Forward projection operator
		R = np.zeros([f.shape[0],self.Nproj,self.N],dtype = 'float32')
		self.clphandle.execFwdMany(R,f)
		return R
	
	def adj(self,R):
		# Adjoint projection operator
		f = np.zeros([R.shape[0],self.N,self.N],dtype = 'float32')
		self.clphandle.execAdjMany(f,R)
		return f

	def fwdp(self,R,f,gpu):
		# Forward projection operator. Work with GPU pointers
		self.clphandle.execFwdManyPtr(R.data.ptr,f.data.ptr,gpu)

	def adjp(self,f,R,gpu):
		# Forward projection operator. Work with GPU pointers
		self.clphandle.execAdjManyPtr(f.data.ptr,R.data.ptr,gpu)
		 


	



