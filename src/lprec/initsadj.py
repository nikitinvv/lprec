import cupy as cp
import numpy as np

class Padj:
    def __init__(self, fZgpu, lp2p1, lp2p2, lp2p1w, lp2p2w, C2lp1, C2lp2, cids, lpids, wids, filter):
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


def create_adj(P, filter_type):
    # convolution function
    fZ = cp.fft.fftshift(fzeta_loop_weights_adj(
        P.Ntheta, P.Nrho, 2*P.beta, P.g-np.log(P.am), 0, 4))

    # (C2lp1,C2lp2), transformed Cartesian to log-polar coordinates
    [x1, x2] = cp.meshgrid(cp.linspace(-1, 1, P.N), cp.linspace(-1, 1, P.N))
    x1 = x1.flatten()
    x2 = x2.flatten()
    x2 = x2*(-1)  # adjust for tomopy
    cids = cp.where(x1**2+x2**2 <= 1)[0]
    C2lp1 = [None]*P.Nspan
    C2lp2 = [None]*P.Nspan
    for k in range(0, P.Nspan):
        z1 = P.aR*(x1[cids]*cp.cos(k*P.beta+P.beta/2)+x2[cids]
                   * cp.sin(k*P.beta+P.beta/2))+(1-P.aR)
        z2 = P.aR*(-x1[cids]*cp.sin(k*P.beta+P.beta/2) +
                   x2[cids]*cp.cos(k*P.beta+P.beta/2))
        C2lp1[k] = cp.arctan2(z2, z1)
        C2lp2[k] = cp.log(cp.sqrt(z1**2+z2**2))
    
    # (lp2p1,lp2p2), transformed log-polar to polar coordinates
    [z1, z2] = cp.meshgrid(P.thsp, cp.exp(P.rhosp))
    z1 = z1.flatten()
    z2 = z2.flatten()
    z2n = z2-(1-P.aR)*cp.cos(z1)
    z2n = z2n/P.aR
    lpids = cp.where((z1 >= -P.beta/2) & (z1 < P.beta/2) & (abs(z2n) <= 1))[0]
    lp2p1 = [None]*P.Nspan
    lp2p2 = [None]*P.Nspan
    for k in range(P.Nspan):
        lp2p1[k] = (z1[lpids]+k*P.beta)
        lp2p2[k] = z2n[lpids]
    # (lp2p1w,lp2p2w), transformed log-polar to polar coordinates (wrapping)
    # right side
    wids = cp.where(cp.log(z2) > +P.g)[0]
    z2n = cp.exp(cp.log(z2[wids])+cp.log(P.am)-P.g)-(1-P.aR)*cp.cos(z1[wids])
    z2n = z2n/P.aR
    lpidsw = cp.where((z1[wids] >= -P.beta/2) &
                      (z1[wids] < P.beta/2) & (abs(z2n) <= 1))[0]
    # left side
    wids2 = cp.where(cp.log(z2) < cp.log(P.am)-P.g+(P.rhosp[1]-P.rhosp[0]))[0]
    z2n2 = cp.exp(cp.log(z2[wids2])-cp.log(P.am)+P.g) - \
        (1-P.aR)*cp.cos(z1[wids2])
    z2n2 = z2n2/P.aR
    lpidsw2 = cp.where((z1[wids2] >= -P.beta/2) &
                       (z1[wids2] < P.beta/2) & (abs(z2n2) <= 1))[0]
    lp2p1w = [None]*P.Nspan
    lp2p2w = [None]*P.Nspan
    for k in range(P.Nspan):
        lp2p1w[k] = (z1[cp.concatenate((lpidsw, lpidsw2))]+k*P.beta)
        lp2p2w[k] = cp.concatenate((z2n[lpidsw], z2n2[lpidsw2]))
    # join for saving
    wids = cp.concatenate((wids[lpidsw], wids2[lpidsw2]))

    # pids, index in polar grids after splitting by spans
    pids = [None]*P.Nspan
    for k in range(P.Nspan):
        pids[k] = cp.where((P.proj >= k*P.beta-P.beta/2) &
                           (P.proj < k*P.beta+P.beta/2))[0]

    # first angle and length of spans
    proj0 = [None]*P.Nspan
    projl = [None]*P.Nspan
    for k in range(P.Nspan):
        proj0[k] = P.proj[pids[k][0]]
        projl[k] = P.proj[pids[k][-1]]-P.proj[pids[k][0]]

    #shift in angles
    projp = (P.Nproj-1)/(proj0[P.Nspan-1]+projl[P.Nspan-1]-proj0[0])

    # adapt for interpolation
    for k in range(P.Nspan):
        lp2p1[k] = (lp2p1[k]-proj0[k])/projl[k] * \
            (len(pids[k])-1)+(proj0[k]-proj0[0])*projp
        lp2p1w[k] = (lp2p1w[k]-proj0[k])/projl[k] * \
            (len(pids[k])-1)+(proj0[k]-proj0[0])*projp
        lp2p2[k] = (lp2p2[k]+1)/2*(P.N-1)
        lp2p2w[k] = (lp2p2w[k]+1)/2*(P.N-1)
        C2lp1[k] = (C2lp1[k]-P.thsp[0])/(P.thsp[-1]-P.thsp[0])*(P.Ntheta-1)
        C2lp2[k] = (C2lp2[k]-P.rhosp[0])/(P.rhosp[-1]-P.rhosp[0])*(P.Nrho-1)

    const = (P.N+1)*(P.N-1)/P.N**2/2*np.sqrt(P.osangles*P.Nproj/P.N/2)
    fZgpu = fZ[:, :P.Ntheta//2+1]*const
    if(P.interp_type == 'cubic'):
        fZgpu = fZgpu/(P.B3com[:, :P.Ntheta//2+1])

    # filter
    if (filter_type != 'None'):
        filter = take_filter(P.N, filter_type)
    else:
        filter = None

    Padj0 = Padj(fZgpu, lp2p1, lp2p2, lp2p1w, lp2p2w,
                 C2lp1, C2lp2, cids, lpids, wids, filter)
    # array representation
    parsi, parsf = savePadjpars(Padj0)
    return (Padj0, parsi, parsf)


def fzeta_loop_weights_adj(Ntheta, Nrho, betas, rhos, a, osthlarge):
    krho = cp.arange(-Nrho/2, Nrho/2, dtype='float32')
    Nthetalarge = osthlarge*Ntheta
    thsplarge = cp.arange(-Nthetalarge/2, Nthetalarge/2,
                          dtype='float32') / Nthetalarge*betas
    fZ = cp.zeros([Nrho, Nthetalarge], dtype='complex64')
    h = cp.ones(Nthetalarge, dtype='float32')
    # correcting = 1+[-3 4 -1]/24correcting(1) = 2*(correcting(1)-0.5)
    # correcting = 1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0correcting[0] = 2*(correcting[0]-0.5)
    correcting = 1+cp.array([-216254335, 679543284, -1412947389, 2415881496, -3103579086,
                             2939942400, -2023224114, 984515304, -321455811, 63253516, -5675265])/958003200.0
    correcting[0] = 2*(correcting[0]-0.5)
    h[0] = h[0]*(correcting[0])
    for j in range(1, len(correcting)):
        h[j] = h[j]*correcting[j]
        h[-1-j+1] = h[-1-j+1]*(correcting[j])
    for j in range(len(krho)):
        fcosa = pow(cp.cos(thsplarge), (2*cp.pi*1j*krho[j]/rhos-a))
        fZ[j, :] = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(h*fcosa)))
    fZ = fZ[:, Nthetalarge//2-Ntheta//2:Nthetalarge//2+Ntheta//2]
    fZ = fZ*(thsplarge[1]-thsplarge[0])
    # put imag to 0 for the border
    fZ[0] = 0
    fZ[:, 0] = 0
    return fZ


def take_filter(N, filter):
    os = 4
    d = 0.5
    Ne = os*N
    t = cp.arange(0, Ne/2+1)/Ne

    if (filter == 'ramp'):
        wfa = Ne*0.5*wint(12, t)  # .*(t/(2*d)<=1)%compute the weigths
    elif (filter == 'shepp-logan'):
        wfa = Ne*0.5*wint(12, t)*cp.sinc(t/(2*d))*(t/d <= 2)
    elif (filter == 'cosine'):
        wfa = Ne*0.5*wint(12, t)*cp.cos(cp.pi*t/(2*d))*(t/d <= 1)
    elif (filter == 'cosine2'):
        wfa = Ne*0.5*wint(12, t)*(cp.cos(cp.pi*t/(2*d)))**2*(t/d <= 1)
    elif (filter == 'hamming'):
        wfa = Ne*0.5*wint(12, t)*(.54 + .46 * cp.cos(cp.pi*t/d))*(t/d <= 1)
    elif (filter == 'hann'):
        wfa = Ne*0.5*wint(12, t)*(1+np.cos(cp.pi*t/d)) / 2.0*(t/d <= 1)
    elif (filter == 'parzen'):
        wfa = Ne*0.5*wint(12, t)*pow(1-t/d, 3)*(t/d <= 1)

    wfa = wfa*(wfa >= 0)
    wfamid = cp.array([2*wfa[0]])
    tmp = wfa
    wfa = cp.concatenate((cp.flipud(tmp[1:]), wfamid))
    wfa = cp.concatenate((wfa, tmp[1:]))
    wfa = wfa[:-1].astype('float32')
    return wfa


def wint(n, t):

    N = len(t)
    s = cp.linspace(1e-40, 1, n)
    # Inverse vandermonde matrix
    tmp1 = cp.arange(n)
    tmp2 = cp.arange(1, n+2)
    iv = cp.linalg.inv(cp.exp(cp.outer(tmp1, cp.log(s))))    
    u = cp.diff(cp.exp(cp.outer(tmp2,cp.log(s)))*cp.tile(1.0/tmp2[...,cp.newaxis], [1, n]))  # integration over short intervals                                                                
    W1 = cp.matmul(iv,u[1:n+1, :])# x*pn(x) term
    W2 = cp.matmul(iv,u[0:n, :])# const*pn(x) term

    # Compensate for overlapping short intervals
    tmp1 = cp.arange(1,n)
    tmp2 = (n-1)*cp.ones((N-2*(n-1)-1))
    tmp3 = cp.arange(n-1, 0, -1)
    p = 1/cp.concatenate((tmp1,tmp2,tmp3))
    w = cp.zeros(N)
    for j in range(N-n+1):
        # Change coordinates, and constant and linear parts
        W = ((t[j+n-1]-t[j])**2)*W1+(t[j+n-1]-t[j])*t[j]*W2

        for k in range(n-1):
            w[j:j+n] = w[j:j+n] + p[j+k]*W[:, k]

    wn = w
    wn[-40:] = (w[-40])/(N-40)*cp.arange(N-40, N)
    return wn

def savePadjpars(P):
    Nspan = len(P.C2lp1)
    Ncids = len(P.C2lp1[0])
    Nlpids = len(P.lp2p1[0])
    Nwids = len(P.lp2p1w[0])
    fZvec = cp.zeros([P.fZgpu.size*2], dtype='float32')
    fZvec[::2] = P.fZgpu.real.flatten()
    fZvec[1::2] = P.fZgpu.imag.flatten()

    parsi = []
    parsf = []
    parsi.append([Ncids])
    parsi.append([Nlpids])
    parsi.append([Nwids])
    for k in range(Nspan):
        parsf.append(P.C2lp1[k].get())
    for k in range(Nspan):
        parsf.append(P.C2lp2[k].get())
    for k in range(Nspan):
        parsf.append(P.lp2p1[k].get())
    for k in range(Nspan):
        parsf.append(P.lp2p2[k].get())
    for k in range(Nspan):
        parsf.append(P.lp2p1w[k].get())
    for k in range(Nspan):
        parsf.append(P.lp2p2w[k].get())
    parsi.append(P.cids.get())
    parsi.append(P.lpids.get())
    parsi.append(P.wids.get())
    parsf.append(fZvec.get())
    if(np.size(P.filter) > 1):
        parsf.append(P.filter.get())
        parsi.append([1])
    else:
        parsi.append([0])

    parsi = np.concatenate(parsi).astype('int32')
    parsf = np.concatenate(parsf).astype('float32')
    
    return (parsi, parsf)
