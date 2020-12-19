from numpy import *


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
    beta = pi/Nspan

    # size after zero padding in radial direction
    N = int(ceil((N0+abs(N0/2-cor)*2.0)/16.0)*16)

    # size after zero padding in the angle direction (for nondense sampling rate)
    osangles = int(max(round(3.0*N/2.0/Nproj), 1))
    Nproj = osangles*Nproj
    # polar space
    proj = arange(0, Nproj)*pi/Nproj-beta/2
    s = linspace(-1, 1, N)

    # log-polar parameters
    (Nrho, Ntheta, dtheta, drho, aR, am, g) = getparameters(
        beta, proj[1]-proj[0], 2.0/(N-1), N, size(proj))

    # log-polar space
    thsp = arange(-Ntheta/2, Ntheta/2)*dtheta
    rhosp = arange(-Nrho, 0)*drho
    erho = transpose(tile(exp(rhosp), [size(thsp), 1]))

    # compensation for cubic interpolation
    B3th = splineB3(thsp, 1)
    B3th = fft.fft(fft.ifftshift(B3th))
    B3rho = splineB3(rhosp, 1)
    B3rho = (fft.fft(fft.ifftshift(B3rho)))
    B3com = array(transpose(matrix(B3rho))*B3th)

    # struct with global parameters
    P = Pgl(Nspan, N, N0, Nproj, Nslices, Ntheta, Nrho, proj, s, thsp,
            rhosp, aR, beta, B3com, am, g, cor, osangles, interp_type)
    # represent as array
    params = float32(append([P.N, P.N0, P.Ntheta, P.Nrho, P.Nspan, P.Nproj,
                             P.Nslices, P.cor, P.osangles, P.interp_type == 'cubic'], erho))
    return (P, params)


def getparameters(beta, dtheta, ds, N, Nproj):
    aR = sin(beta/2)/(1+sin(beta/2))
    am = (cos(beta/2)-sin(beta/2))/(1+sin(beta/2))

    # wrapping
    g = osg(aR, beta/2)
    Ntheta = N
    Nrho = 2*N
    dtheta = (2*beta)/Ntheta
    drho = (g-log(am))/Nrho
    return (Nrho, Ntheta, dtheta, drho, aR, am, g)


def osg(aR, theta):
    t = linspace(-pi/2, pi/2, 100000)
    w = aR*cos(t)+(1-aR)+1j*aR*sin(t)
    g = max(log(abs(w))+log(cos(theta-arctan2(imag(w), real(w)))))
    return g


def splineB3(x2, r):
    sizex = size(x2)
    x2 = x2-(x2[-1]+x2[0])/2
    stepx = x2[1]-x2[0]
    ri = int32(ceil(2*r))
    r = r*stepx
    x2c = x2[int32(ceil((sizex+1)/2.0))-1]
    x = x2[range(int32(ceil((sizex+1)/2.0)-ri-1),
                 int32(ceil((sizex+1)/2.0)+ri))]
    d = abs(x-x2c)/r
    B3 = x*0
    for ix in range(-ri, ri+1):
        id = ix+ri
        if d[id] < 1:  # use the first polynomial
            B3[id] = (3*d[id]**3-6*d[id]**2+4)/6
        else:
            if(d[id] < 2):
                B3[id] = (-d[id]**3+6*d[id]**2-12*d[id]+8)/6
    B3f = x2*0
    B3f[range(int32(ceil((sizex+1)/2.0)-ri-1),
              int32(ceil((sizex+1)/2.0)+ri))] = B3
    return B3f
