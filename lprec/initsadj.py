from numpy import *

class Padj:
	def __init__(self,fZgpu,lp2p1,lp2p2,lp2p1w,lp2p2w,C2lp1,C2lp2,cids,lpids,wids,filter):
		self.fZgpu = fZgpu
		self.lp2p1 = lp2p1 
		self.lp2p2 = lp2p2
		self.lp2p1w = lp2p1w 
		self.lp2p2w = lp2p2w
		self.C2lp1 = C2lp1 
		self.C2lp2 = C2lp2
		self.cids = cids
		self.lpids = lpids
		self.wids = wids
		self.filter = filter

def create_adj(P,filter_type):
	#convolution function	
	fZ = fft.fftshift(fzeta_loop_weights_adj(P.Ntheta,P.Nrho,2*P.beta,P.g-log(P.am),0,4))

	#(C2lp1,C2lp2), transformed Cartesian to log-polar coordinates
	[x1,x2] = meshgrid(linspace(-1,1,P.N),linspace(-1,1,P.N)) 
	x1 = ndarray.flatten(x1) 
	x2 = ndarray.flatten(x2)
	x2 = x2*(-1) #adjust for tomopy
	cids = where(x1**2+x2**2<=1)
	C2lp1 = [None]*P.Nspan 
	C2lp2 = [None]*P.Nspan
	for k in range(0,P.Nspan):
		z1 = P.aR*( x1[cids]*cos(k*P.beta+P.beta/2)+x2[cids]*sin(k*P.beta+P.beta/2))+(1-P.aR)
		z2 = P.aR*(-x1[cids]*sin(k*P.beta+P.beta/2)+x2[cids]*cos(k*P.beta+P.beta/2))
		C2lp1[k] = arctan2(z2,z1) 
		C2lp2[k] = log(sqrt(z1**2+z2**2))  

	#(lp2p1,lp2p2), transformed log-polar to polar coordinates
	[z1,z2] = meshgrid(P.thsp,exp(P.rhosp)) 
	z1 = ndarray.flatten(z1) 
	z2 = ndarray.flatten(z2)
	z2n = z2-(1-P.aR)*cos(z1)
	z2n = z2n/P.aR
	lpids = where((z1>=-P.beta/2) & (z1<P.beta/2) &(abs(z2n)<=1))
	lp2p1 = [None]*P.Nspan 
	lp2p2 = [None]*P.Nspan
	for k in range(0,P.Nspan):
		lp2p1[k] = z1[lpids]+k*P.beta
		lp2p2[k] = z2n[lpids]

	#(lp2p1w,lp2p2w), transformed log-polar to polar coordinates (wrapping)
	#right side
	wids = ndarray.flatten(array(where(log(z2)>+P.g)))
	z2n = exp(log(z2[wids])+log(P.am)-P.g)-(1-P.aR)*cos(z1[wids])
	z2n = z2n/P.aR
	lpidsw = where((z1[wids]>=-P.beta/2) & (z1[wids]<P.beta/2) & (abs(z2n)<=1))
	#left side
	wids2 = ndarray.flatten(array(where(log(z2)<log(P.am)-P.g+(P.rhosp[1]-P.rhosp[0]))))
	z2n2 = exp(log(z2[wids2])-log(P.am)+P.g)-(1-P.aR)*cos(z1[wids2]) 
	z2n2 = z2n2/P.aR
	lpidsw2 = where((z1[wids2]>=-P.beta/2) & (z1[wids2]<P.beta/2) & (abs(z2n2)<=1))
	lp2p1w = [None]*P.Nspan 
	lp2p2w = [None]*P.Nspan
	for k in range(0,P.Nspan):
		lp2p1w[k] = z1[append(lpidsw,lpidsw2)]+k*P.beta
		lp2p2w[k] = append(z2n[lpidsw],z2n2[lpidsw2])
	#join for saving
	wids = append(wids[lpidsw],wids2[lpidsw2])

	#pids, index in polar grids after splitting by spans
	pids = [None]*P.Nspan
	for k in range(0,P.Nspan):
		pids[k] = ndarray.flatten(array(where((P.proj>=k*P.beta-P.beta/2) & (P.proj<k*P.beta+P.beta/2))))

	#first angle and length of spans
	proj0 = [None]*P.Nspan 
	projl = [None]*P.Nspan
	for k in range(0,P.Nspan):
		proj0[k] = P.proj[pids[k][0]]
		projl[k] = P.proj[pids[k][-1]]-P.proj[pids[k][0]]	

	#shift in angles
	projp = (P.Nproj-1)/(proj0[P.Nspan-1]+projl[P.Nspan-1]-proj0[0])

	#adapt for interpolation
	for k in range(0,P.Nspan):
		lp2p1[k] = (lp2p1[k]-proj0[k])/projl[k]*(size(pids[k])-1)+(proj0[k]-proj0[0])*projp
		lp2p1w[k] = (lp2p1w[k]-proj0[k])/projl[k]*(size(pids[k])-1)+(proj0[k]-proj0[0])*projp
		lp2p2[k] = (lp2p2[k]+1)/2*(P.N-1)
		lp2p2w[k] = (lp2p2w[k]+1)/2*(P.N-1)
		C2lp1[k] = (C2lp1[k]-P.thsp[0])/(P.thsp[-1]-P.thsp[0])*(P.Ntheta-1)
		C2lp2[k] = (C2lp2[k]-P.rhosp[0])/(P.rhosp[-1]-P.rhosp[0])*(P.Nrho-1)
	
	const = (P.N+1)/float32(P.N)*(P.N-1)/2/P.N
	fZgpu = fZ[:,arange(0,P.Ntheta/2+1)]*const
	if(P.interp_type=='cubic'):
		fZgpu = fZgpu/(P.B3com[:,arange(0,P.Ntheta/2+1)])

	#filter
	if (filter_type!= 'None'):
		filter = take_filter(P.N,filter_type)
	else: filter = None

	Padj0 = Padj(fZgpu,lp2p1,lp2p2,lp2p1w,lp2p2w,C2lp1,C2lp2,cids,lpids,wids,filter)
	#array representation
	parsi,parsf = savePadjpars(Padj0)
	return (Padj0,parsi,parsf)

def fzeta_loop_weights_adj(Ntheta,Nrho,betas,rhos,a,osthlarge):
	krho = arange(-Nrho/2,Nrho/2)
	Nthetalarge = osthlarge*Ntheta
	thsplarge = arange(-Nthetalarge/2,Nthetalarge/2)/float32(Nthetalarge)*betas
	fZ = array(zeros(shape = (Nrho,Nthetalarge)),dtype = complex)
	h = array(ones(Nthetalarge))
	# correcting = 1+[-3 4 -1]/24correcting(1) = 2*(correcting(1)-0.5)
	# correcting = 1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0correcting[0] = 2*(correcting[0]-0.5)
	correcting = 1+array([-216254335,679543284,-1412947389,2415881496,-3103579086,2939942400,-2023224114,984515304,-321455811,63253516,-5675265])/958003200.0
	correcting[0] = 2*(correcting[0]-0.5)
	h[0] = h[0]*(correcting[0])
	for j in range(1,size(correcting)):
		h[j] = h[j]*correcting[j]
		h[-1-j+1] = h[-1-j+1]*(correcting[j])
	for j in range(0,size(krho)):
		fcosa = pow(cos(thsplarge),(2*pi*1j*krho[j]/rhos-a))
		fZ[j,:] = fft.fftshift(fft.fft(fft.fftshift(h*fcosa)))
	fZ = fZ[:,range(Nthetalarge/2-Ntheta/2,Nthetalarge/2+Ntheta/2)]
	fZ = fZ*(thsplarge[1]-thsplarge[0])
	#put imag to 0 for the border
	fZ[0] = 0
	fZ[:,0] = 0
	return fZ

def take_filter(N,filter):
	os = 4
	d = 0.5
	Ne = os*N
	t = arange(0,Ne/2+1)/float32(Ne)

	if (filter=='ramp'):
	        wfa = Ne*0.5*wint(12,t)#.*(t/(2*d)<=1)%compute the weigths
	elif (filter=='shepp-logan'):
		wfa = Ne*0.5*wint(12,t)*sinc(t/(2*d))*(t/d<=2)
	elif (filter=='cosine'):
        	wfa = Ne*0.5*wint(12,t)*cos(pi*t/(2*d))*(t/d<=1)
	elif (filter=='cosine2'):
	        wfa = Ne*0.5*wint(12,t)*(cos(pi*t/(2*d)))**2*(t/d<=1) 
	elif (filter=='hamming'):
		wfa = Ne*0.5*wint(12,t)*(.54 + .46 * cos(pi*t/d))*(t/d<=1)
	elif (filter=='hann'):
	        wfa = Ne*0.5*wint(12,t)*(1+cos(pi*t/d)) / 2.0*(t/d<=1)
	elif (filter=='parzen'):
	        wfa = Ne*0.5*wint(12,t)*pow(1-t,3)#*(t/d<=1)

	wfa = wfa*(wfa>=0)
	wfamid = 2*wfa[0]
	tmp = wfa
	wfa = append(flipud(tmp[1:]),wfamid)
	wfa = append(wfa, tmp[1:])
	wfa = wfa[0:-1]
	wfa = float32(wfa)
	return wfa

def wint(n,t):
	N = size(t)
	s = linspace(1e-40,1,n)
	iv = linalg.inv(exp(transpose(matrix(arange(0,n)))*log(s)))#Inverse vandermonde matrix
	u = diff(multiply(exp(transpose(matrix(arange(1,n+2)))*log(s)),transpose(tile(1.0/arange(1,n+2),[n,1])))) #integration over short intervals

	W1 = iv*u[range(1,n+1),:]#x*pn(x) term
	W2 = iv*u[range(0,n),:]#const*pn(x) term

	p = 1./concatenate([range(1,n), [(n-1)]*(N-2*(n-1)-1),range(n-1,0,-1)])#Compensate for overlapping short intervals
	w = float32(array([0]*N))
	for j in range(0,N-n+1):
		W = ((t[j+n-1]-t[j])**2)*W1+(t[j+n-1]-t[j])*t[j]*W2#Change coordinates, and constant and linear parts
		
		for k in range(0,n-1):
			w[j+arange(0,n)] = w[j+arange(0,n)]+transpose(W[:,k])*p[j+k]#% Add to weights

	wn = w
	wn[range(N-40,N)] = (w[N-40])/(N-40)*arange(N-40,N)
	w = wn
	return w

def savePadjpars(P):
	Nspan = shape(P.C2lp1)[0]
	Ncids = size(P.C2lp1[0])
	Nlpids = size(P.lp2p1[0])
	Nwids = size(P.lp2p1w[0])
	fZvec = ndarray.flatten(transpose(array([real(ndarray.flatten(P.fZgpu)),ndarray.flatten(imag(P.fZgpu))])))

	parsi = []
	parsf = []
	parsi = append(parsi,Ncids)
	parsi = append(parsi,Nlpids)
	parsi = append(parsi,Nwids)
	for k in range(0,Nspan):
		parsf = append(parsf,P.C2lp1[k])
	for k in range(0,Nspan):
		parsf = append(parsf,P.C2lp2[k])
	for k in range(0,Nspan):
		parsf = append(parsf,P.lp2p1[k])
	for k in range(0,Nspan):
		parsf = append(parsf,P.lp2p2[k])
	for k in range(0,Nspan):
		parsf = append(parsf,P.lp2p1w[k])
	for k in range(0,Nspan):
		parsf = append(parsf,P.lp2p2w[k])
	parsi = append(parsi,P.cids)
	parsi = append(parsi,P.lpids)
	parsi = append(parsi,P.wids)
	parsf = append(parsf,fZvec)
	if(size(P.filter)>1):
		parsf = append(parsf,P.filter)
		parsi = append(parsi,1)
	else:
		parsi = append(parsi,0)
	
	parsi = int32(parsi)
	parsf = float32(parsf)
	return (parsi,parsf)