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
	
	def initcmem(self,flg):
		#init memory in C (could be used by several gpus)
		self.clphandle = lpRgpu.lpRgpu(self.glpars)
		if(flg):
			self.clphandle.initFwd(self.fwdparsi,self.fwdparamsf)
		self.clphandle.initAdj(self.adjparsi,self.adjparamsf)	

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

	def fwdp(self,Rptr,fptr):
		# Forward projection operator. Work with GPU pointers
		self.clphandle.execFwdManyPtr(Rptr,fptr)

	def adjp(self,fptr,Rptr):
		# Forward projection operator. Work with GPU pointers
		self.clphandle.execAdjManyPtr(fptr,Rptr)
		 


	



