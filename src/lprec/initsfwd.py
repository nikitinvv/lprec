import numpy as np
import cupy as cp

class Pfwd:
    def __init__(self, fZgpu, lp2C1, lp2C2, p2lp1, p2lp2, cids, pids):
        self.fZgpu = fZgpu
        self.lp2C1 = lp2C1
        self.lp2C2 = lp2C2
        self.p2lp1 = p2lp1
        self.p2lp2 = p2lp2
        self.cids = cids
        self.pids = pids


def create_fwd(P):
    # convolution function
    fZ = np.fft.fftshift(fzeta_loop_weights(
        P.Ntheta, P.Nrho, 2*P.beta, P.g-np.log(P.am), 0, 4))

    # (lp2C1,lp2C2), transformed log-polar to Cartesian coordinates
    tmp1 = np.outer(np.exp(P.rhosp), np.cos(P.thsp)).flatten()
    tmp2 = np.outer(np.exp(P.rhosp), np.sin(P.thsp)).flatten()
    lp2C1 = [None]*P.Nspan
    lp2C2 = [None]*P.Nspan
    for k in range(0, P.Nspan):
        lp2C1[k] = ((tmp1-(1-P.aR))*np.cos(k*P.beta+P.beta/2) -
                    tmp2*np.sin(k*P.beta+P.beta/2))/P.aR
        lp2C2[k] = ((tmp1-(1-P.aR))*np.sin(k*P.beta+P.beta/2) +
                    tmp2*np.cos(k*P.beta+P.beta/2))/P.aR
        lp2C2[k] *= (-1)  # adjust for Tomopy
        cids = np.where((lp2C1[k]**2+lp2C2[k]**2) <= 1)[0]
        lp2C1[k] = lp2C1[k][cids]
        lp2C2[k] = lp2C2[k][cids]

    # pids, index in polar grids after splitting by spans
    pids = [None]*P.Nspan
    [s0, th0] = np.meshgrid(P.s, P.proj)
    th0 = th0.flatten()
    s0 = s0.flatten()
    for k in range(0, P.Nspan):
        pids[k] = np.where((th0 >= k*P.beta-P.beta/2) &
                           (th0 < k*P.beta+P.beta/2))[0]

    # (p2lp1,p2lp2), transformed polar to log-polar coordinates
    p2lp1 = [None]*P.Nspan
    p2lp2 = [None]*P.Nspan
    for k in range(0, P.Nspan):
        th00 = th0[pids[k]]-k*P.beta
        s00 = s0[pids[k]]
        p2lp1[k] = th00
        p2lp2[k] = np.log(s00*P.aR+(1-P.aR)*np.cos(th00))

    # adapt for gpu interp
    for k in range(0, P.Nspan):
        lp2C1[k] = (lp2C1[k]+1)/2*(P.N-1)
        lp2C2[k] = (lp2C2[k]+1)/2*(P.N-1)
        p2lp1[k] = (p2lp1[k]-P.thsp[0])/(P.thsp[-1]-P.thsp[0])*(P.Ntheta-1)
        p2lp2[k] = (p2lp2[k]-P.rhosp[0])/(P.rhosp[-1]-P.rhosp[0])*(P.Nrho-1)

    const = np.sqrt(P.N*P.osangles/P.Nproj)*np.pi/4 / \
        P.aR/np.sqrt(2)  # adjust constant
    fZgpu = fZ[:, :P.Ntheta//2+1]*const
    if(P.interp_type == 'cubic'):
        fZgpu = fZgpu/(P.B3com[:, :P.Ntheta//2+1])

    Pfwd0 = Pfwd(fZgpu, lp2C1, lp2C2, p2lp1, p2lp2, cids, pids)
    # array representation
    parsi, parsf = savePfwdpars(Pfwd0)
    return Pfwd0, parsi, parsf


def fzeta_loop_weights(Ntheta, Nrho, betas, rhos, a, osthlarge):
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
        fcosa = pow(cp.cos(thsplarge), (-2*cp.pi*1j*krho[j]/rhos-1-a))
        fZ[j, :] = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(h*fcosa)))
    fZ = fZ[:, Nthetalarge//2-Ntheta//2:Nthetalarge//2+Ntheta//2]
    fZ = fZ*(thsplarge[1]-thsplarge[0])
    # put imag to 0 for the border
    fZ[0] = 0
    fZ[:, 0] = 0
    return fZ.get()


def savePfwdpars(P):
    Nspan = len(P.pids)
    Npids = [None]*Nspan
    for k in range(0, Nspan):
        Npids[k] = len(P.pids[k])
    Ncids = len(P.lp2C1[0])
    fZvec = np.zeros([P.fZgpu.size*2], dtype='float32')
    fZvec[::2] = P.fZgpu.real.flatten()
    fZvec[1::2] = P.fZgpu.imag.flatten()

    parsi = []
    parsf = []

    parsi = np.append(parsi, Npids)
    for k in range(0, Nspan):
        parsi = np.append(parsi, P.pids[k])
    parsi = np.append(parsi, Ncids)
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.lp2C1[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.lp2C2[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.p2lp1[k])
    for k in range(0, Nspan):
        parsf = np.append(parsf, P.p2lp2[k])
    parsi = np.append(parsi, P.cids)
    parsf = np.append(parsf, fZvec)
    parsi = np.int32(parsi)
    parsf = np.float32(parsf)
    return (parsi, parsf)
