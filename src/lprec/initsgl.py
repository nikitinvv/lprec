import numpy as np

class Pgl:
    def __init__(self, Nspan, N, N0, Nproj, Nslices, Ntheta, Nrho, proj, s, thsp, rhosp, aR, beta, B3com, am, g, cor, osangles, interp_type):
        self.Nspan = Nspan
        self.N = N
        self.N0 = N0
        self.Nproj = Nproj
        self.Nslices = Nslices
        self.Ntheta = Ntheta
        self.Nrho = Nrho
        self.proj = proj
        self.s = s
        self.thsp = thsp
        self.rhosp = rhosp
        self.aR = aR
        self.beta = beta
        self.B3com = B3com
        self.am = am
        self.g = g
        self.cor = cor
        self.osangles = osangles
        self.interp_type = interp_type


def create_gl(N0, Nproj, Nslices, cor, interp_type):
    Nspan = 3
    beta = np.pi/Nspan

    # size after zero padding in radial direction
    N = int(np.ceil((N0+abs(N0/2-cor)*2.0)/16.0)*16)
    # size after zero padding in the angle direction (for nondense sampling rate)
    osangles = int(max(round(3.0*N/2.0/Nproj), 1))
    Nproj = osangles*Nproj
    # polar space
    proj = np.arange(0, Nproj)*np.pi/Nproj-beta/2
    s = np.linspace(-1, 1, N)

    # np.log-polar parameters
    (Nrho, Ntheta, dtheta, drho, aR, am, g) = getparameters(
        beta, proj[1]-proj[0], 2.0/(N-1), N, Nproj)

    # np.log-polar space
    thsp = np.arange(-Ntheta/2, Ntheta/2)*dtheta
    rhosp = np.arange(-Nrho, 0)*drho
    erho = np.tile(np.exp(rhosp)[...,np.newaxis], [1, Ntheta])
    
    # compensation for cubic interpolation
    B3th = splineB3(thsp, 1)
    B3th = np.fft.fft(np.fft.ifftshift(B3th))
    B3rho = splineB3(rhosp, 1)
    B3rho = (np.fft.fft(np.fft.ifftshift(B3rho)))
    B3com = np.outer(B3rho,B3th)
    
    # struct with global parameters
    P = Pgl(Nspan, N, N0, Nproj, Nslices, Ntheta, Nrho, proj, s, thsp,
            rhosp, aR, beta, B3com, am, g, cor, osangles, interp_type)
    # represent as array
    params = np.float32(np.append([P.N, P.N0, P.Ntheta, P.Nrho, P.Nspan, P.Nproj,
                             P.Nslices, P.cor, P.osangles, P.interp_type == 'cubic'], erho))
    return (P, params)


def getparameters(beta, dtheta, ds, N, Nproj):
    aR = np.sin(beta/2)/(1+np.sin(beta/2))
    am = (np.cos(beta/2)-np.sin(beta/2))/(1+np.sin(beta/2))

    # wrapnp.ping
    g = osg(aR, beta/2)
    Ntheta = N
    Nrho = 2*N
    dtheta = (2*beta)/Ntheta
    drho = (g-np.log(am))/Nrho
    return (Nrho, Ntheta, dtheta, drho, aR, am, g)


def osg(aR, theta):
    t = np.linspace(-np.pi/2, np.pi/2, 100000)
    w = aR*np.cos(t)+(1-aR)+1j*aR*np.sin(t)
    g = max(np.log(abs(w))+np.log(np.cos(theta-np.arctan2(w.imag, w.real))))
    return g


def splineB3(x2, r):
    sizex = len(x2)
    x2 = x2-(x2[-1]+x2[0])/2
    stepx = x2[1]-x2[0]
    ri = int(np.ceil(2*r))
    r = r*stepx
    x2c = x2[int(np.ceil((sizex+1)/2.0))-1]
    x = x2[int(np.ceil((sizex+1)/2.0)-ri-1):int(np.ceil((sizex+1)/2.0)+ri)]
    d = np.abs(x-x2c)/r
    B3 = x*0
    for ix in range(-ri, ri+1):
        id = ix+ri
        if d[id] < 1:  # use the first polynomial
            B3[id] = (3*d[id]**3-6*d[id]**2+4)/6
        else:
            if(d[id] < 2):
                B3[id] = (-d[id]**3+6*d[id]**2-12*d[id]+8)/6
    B3f = x2*0
    B3f[int(np.ceil((sizex+1)/2.0)-ri-1):int(np.ceil((sizex+1)/2.0)+ri)] = B3
    return B3f
