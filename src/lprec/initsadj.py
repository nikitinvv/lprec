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
    fZ = np.fft.fftshift(fzeta_loop_weights_adj(
        P.Ntheta, P.Nrho, 2*P.beta, P.g-np.log(P.am), 0, 4))

    # (C2lp1,C2lp2), transformed Cartesian to log-polar coordinates
    [x1, x2] = np.meshgrid(np.linspace(-1, 1, P.N), np.linspace(-1, 1, P.N))
    x1 = x1.flatten()
    x2 = x2.flatten()
    x2 = x2*(-1)  # adjust for tomopy
    cids = np.where(x1**2+x2**2 <= 1)[0]
    C2lp1 = [None]*P.Nspan
    C2lp2 = [None]*P.Nspan
    for k in range(0, P.Nspan):
        z1 = P.aR*(x1[cids]*np.cos(k*P.beta+P.beta/2)+x2[cids]
                   * np.sin(k*P.beta+P.beta/2))+(1-P.aR)
        z2 = P.aR*(-x1[cids]*np.sin(k*P.beta+P.beta/2) +
                   x2[cids]*np.cos(k*P.beta+P.beta/2))
        C2lp1[k] = np.arctan2(z2, z1)
        C2lp2[k] = np.log(np.sqrt(z1**2+z2**2))

    # (lp2p1,lp2p2), transformed log-polar to polar coordinates
    [z1, z2] = np.meshgrid(P.thsp, np.exp(P.rhosp))
    z1 = z1.flatten()
    z2 = z2.flatten()
    z2n = z2-(1-P.aR)*np.cos(z1)
    z2n = z2n/P.aR
    lpids = np.where((z1 >= -P.beta/2) & (z1 < P.beta/2) & (abs(z2n) <= 1))[0]
    lp2p1 = [None]*P.Nspan
    lp2p2 = [None]*P.Nspan
    for k in range(P.Nspan):
        lp2p1[k] = z1[lpids]+k*P.beta
        lp2p2[k] = z2n[lpids]

    # (lp2p1w,lp2p2w), transformed log-polar to polar coordinates (wrapping)
    # right side
    wids = np.where(np.log(z2) > +P.g)[0]
    z2n = np.exp(np.log(z2[wids])+np.log(P.am)-P.g)-(1-P.aR)*np.cos(z1[wids])
    z2n = z2n/P.aR
    lpidsw = np.where((z1[wids] >= -P.beta/2) &
                      (z1[wids] < P.beta/2) & (abs(z2n) <= 1))[0]
    # left side
    wids2 = np.where(np.log(z2) < np.log(P.am)-P.g+(P.rhosp[1]-P.rhosp[0]))[0]
    z2n2 = np.exp(np.log(z2[wids2])-np.log(P.am)+P.g) - \
        (1-P.aR)*np.cos(z1[wids2])
    z2n2 = z2n2/P.aR
    lpidsw2 = np.where((z1[wids2] >= -P.beta/2) &
                       (z1[wids2] < P.beta/2) & (abs(z2n2) <= 1))[0]
    lp2p1w = [None]*P.Nspan
    lp2p2w = [None]*P.Nspan
    for k in range(P.Nspan):
        lp2p1w[k] = z1[np.append(lpidsw, lpidsw2)]+k*P.beta
        lp2p2w[k] = np.append(z2n[lpidsw], z2n2[lpidsw2])
    # join for saving
    wids = np.append(wids[lpidsw], wids2[lpidsw2])

    # pids, index in polar grids after splitting by spans
    pids = [None]*P.Nspan
    for k in range(P.Nspan):
        pids[k] = np.where((P.proj >= k*P.beta-P.beta/2) &
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
    correcting = 1+np.array([-216254335, 679543284, -1412947389, 2415881496, -3103579086,
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
    return fZ.get()


def take_filter(N, filter):
    os = 4
    d = 0.5
    Ne = os*N
    t = np.arange(0, Ne/2+1, dtype='float32')/Ne

    if (filter == 'ramp'):
        wfa = Ne*0.5*wint(12, t)  # .*(t/(2*d)<=1)%compute the weigths
    elif (filter == 'shepp-logan'):
        wfa = Ne*0.5*wint(12, t)*np.sinc(t/(2*d))*(t/d <= 2)
    elif (filter == 'cosine'):
        wfa = Ne*0.5*wint(12, t)*np.cos(np.pi*t/(2*d))*(t/d <= 1)
    elif (filter == 'cosine2'):
        wfa = Ne*0.5*wint(12, t)*(np.cos(np.pi*t/(2*d)))**2*(t/d <= 1)
    elif (filter == 'hamming'):
        wfa = Ne*0.5*wint(12, t)*(.54 + .46 * np.cos(np.pi*t/d))*(t/d <= 1)
    elif (filter == 'hann'):
        wfa = Ne*0.5*wint(12, t)*(1+np.cos(np.pi*t/d)) / 2.0*(t/d <= 1)
    elif (filter == 'parzen'):
        wfa = Ne*0.5*wint(12, t)*pow(1-t/d, 3)*(t/d <= 1)

    wfa = wfa*(wfa >= 0)
    wfamid = 2*wfa[0]
    tmp = wfa
    wfa = np.append(np.flipud(tmp[1:]), wfamid)
    wfa = np.append(wfa, tmp[1:])
    wfa = np.float32(wfa[:-1])
    return wfa


def wint(n, t):
    N = len(t)
    s = np.linspace(1e-40, 1, n)
    # Inverse vandermonde matrix
    iv = np.linalg.inv(np.exp(np.outer(np.arange(0, n), np.log(s))))
    u = np.diff(np.multiply(np.exp(np.transpose(np.matrix(np.arange(1, n+2)))*np.log(s)),
                            np.transpose(np.tile(1.0/np.arange(1, n+2), [n, 1]))))  # integration over short intervals
    W1 = iv*u[1:n+1, :]  # x*pn(x) term
    W2 = iv*u[0:n, :]  # const*pn(x) term

    # Compensate for overlapping short intervals
    p = 1/np.concatenate([range(1, n), [(n-1)] *
                          (N-2*(n-1)-1), range(n-1, 0, -1)])
    w = np.zeros(N)
    for j in range(0, N-n+1):
        # Change coordinates, and constant and linear parts
        W = ((t[j+n-1]-t[j])**2)*W1+(t[j+n-1]-t[j])*t[j]*W2

        for k in range(0, n-1):
            w[j:j+n] = w[j:j+n] + np.outer(p[j+k], W[:, k])

    wn = w
    wn[-40:] = (w[-40])/(N-40)*np.arange(N-40, N)
    return wn


def savePadjpars(P):
    Nspan = len(P.C2lp1)
    Ncids = len(P.C2lp1[0])
    Nlpids = len(P.lp2p1[0])
    Nwids = len(P.lp2p1w[0])
    fZvec = np.zeros([P.fZgpu.size*2], dtype='float32')
    fZvec[::2] = P.fZgpu.real.flatten()
    fZvec[1::2] = P.fZgpu.imag.flatten()

    parsi = []
    parsf = []
    parsi = np.append(parsi, Ncids)
    parsi = np.append(parsi, Nlpids)
    parsi = np.append(parsi, Nwids)
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.C2lp1[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.C2lp2[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.lp2p1[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.lp2p2[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.lp2p1w[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.lp2p2w[k])
    parsi = np.append(parsi, P.cids)
    parsi = np.append(parsi, P.lpids)
    parsi = np.append(parsi, P.wids)
    parsf = np.append(parsf, fZvec)
    if(np.size(P.filter) > 1):
        parsf = np.append(parsf, P.filter)
        parsi = np.append(parsi, 1)
    else:
        parsi = np.append(parsi, 0)

    parsi = np.int32(parsi)
    parsf = np.float32(parsf)
    return (parsi, parsf)
