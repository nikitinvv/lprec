from numpy import *


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
    fZ = fft.fftshift(fzeta_loop_weights(
        P.Ntheta, P.Nrho, 2*P.beta, P.g-log(P.am), 0, 4))

    # (lp2C1,lp2C2), transformed log-polar to Cartesian coordinates
    texprho = transpose(matrix(exp(P.rhosp)))
    lp2C1 = [None]*P.Nspan
    lp2C2 = [None]*P.Nspan
    for k in range(0, P.Nspan):
        lp2C1[k] = ndarray.flatten(array((texprho*cos(P.thsp)-(1-P.aR))*cos(
            (k)*P.beta+P.beta/2)-texprho*sin(P.thsp)*sin((k)*P.beta+P.beta/2))/P.aR)
        lp2C2[k] = ndarray.flatten(array((texprho*cos(P.thsp)-(1-P.aR))*sin(
            (k)*P.beta+P.beta/2)+texprho*sin(P.thsp)*cos((k)*P.beta+P.beta/2))/P.aR)
        lp2C2[k] = lp2C2[k]*(-1)  # adjust for Tomopy
        cids = where((lp2C1[k]**2+lp2C2[k]**2) <= 1)
        lp2C1[k] = lp2C1[k][cids]
        lp2C2[k] = lp2C2[k][cids]

    # pids, index in polar grids after splitting by spans
    pids = [None]*P.Nspan
    [s0, th0] = meshgrid(P.s, P.proj)
    th0 = ndarray.flatten(th0)
    s0 = ndarray.flatten(s0)
    for k in range(0, P.Nspan):
        pids[k] = ndarray.flatten(
            array(where((th0 >= k*P.beta-P.beta/2) & (th0 < k*P.beta+P.beta/2))))

    # (p2lp1,p2lp2), transformed polar to log-polar coordinates
    p2lp1 = [None]*P.Nspan
    p2lp2 = [None]*P.Nspan
    for k in range(0, P.Nspan):
        th00 = th0[pids[k]]-k*P.beta
        s00 = s0[pids[k]]
        p2lp1[k] = th00
        p2lp2[k] = log(s00*P.aR+(1-P.aR)*cos(th00))

    # adapt for gpu interp
    for k in range(0, P.Nspan):
        lp2C1[k] = (lp2C1[k]+1)/2*(P.N-1)
        lp2C2[k] = (lp2C2[k]+1)/2*(P.N-1)
        p2lp1[k] = (p2lp1[k]-P.thsp[0])/(P.thsp[-1]-P.thsp[0])*(P.Ntheta-1)
        p2lp2[k] = (p2lp2[k]-P.rhosp[0])/(P.rhosp[-1]-P.rhosp[0])*(P.Nrho-1)

    const = sqrt(P.N)*pi/P.Nproj/4/P.aR*sqrt(P.osangles) * \
        sqrt(P.Nproj)/sqrt(2)  # adjust constant
    fZgpu = fZ[:, arange(0, int(P.Ntheta/2)+1)]*const
    if(P.interp_type == 'cubic'):
        fZgpu = fZgpu/(P.B3com[:, arange(0, int(P.Ntheta/2)+1)])

    Pfwd0 = Pfwd(fZgpu, lp2C1, lp2C2, p2lp1, p2lp2, cids, pids)
    # array representation
    parsi, parsf = savePfwdpars(Pfwd0)
    return Pfwd0, parsi, parsf


def fzeta_loop_weights(Ntheta, Nrho, betas, rhos, a, osthlarge):
    krho = arange(-Nrho/2, Nrho/2)
    Nthetalarge = osthlarge*Ntheta
    thsplarge = arange(-Nthetalarge/2, Nthetalarge/2) / \
        float32(Nthetalarge)*betas
    fZ = array(zeros(shape=(Nrho, Nthetalarge)), dtype=complex)
    h = array(ones(Nthetalarge))
    # correcting = 1+[-3 4 -1]/24correcting(1) = 2*(correcting(1)-0.5)
    # correcting = 1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0correcting[0] = 2*(correcting[0]-0.5)
    correcting = 1+array([-216254335, 679543284, -1412947389, 2415881496, -3103579086,
                          2939942400, -2023224114, 984515304, -321455811, 63253516, -5675265])/958003200.0
    correcting[0] = 2*(correcting[0]-0.5)
    h[0] = h[0]*(correcting[0])
    for j in range(1, size(correcting)):
        h[j] = h[j]*correcting[j]
        h[-1-j+1] = h[-1-j+1]*(correcting[j])
    for j in range(0, size(krho)):
        fcosa = pow(cos(thsplarge), (-2*pi*1j*krho[j]/rhos-1-a))
        fZ[j, :] = fft.fftshift(fft.fft(fft.fftshift(h*fcosa)))
    fZ = fZ[:, range(int(Nthetalarge/2-Ntheta/2), int(Nthetalarge/2+Ntheta/2))]
    fZ = fZ*(thsplarge[1]-thsplarge[0])
    # put imag to 0 for the border
    fZ[0] = 0
    fZ[:, 0] = 0
    return fZ


def savePfwdpars(P):
    Nspan = shape(P.pids)[0]
    Npids = [None]*Nspan
    for k in range(0, Nspan):
        Npids[k] = size(P.pids[k])
    Ncids = size(P.lp2C1[0])
    fZvec = ndarray.flatten(transpose(
        array([real(ndarray.flatten(P.fZgpu)), ndarray.flatten(imag(P.fZgpu))])))
    parsi = []
    parsf = []

    parsi = append(parsi, Npids)
    for k in range(0, Nspan):
        parsi = append(parsi, P.pids[k])
    parsi = append(parsi, Ncids)
    for k in range(0, Nspan):
        parsf = append(parsf, P.lp2C1[k])
    for k in range(0, Nspan):
        parsf = append(parsf, P.lp2C2[k])
    for k in range(0, Nspan):
        parsf = append(parsf, P.p2lp1[k])
    for k in range(0, Nspan):
        parsf = append(parsf, P.p2lp2[k])
    parsi = append(parsi, P.cids)
    parsf = append(parsf, fZvec)
    parsi = int32(parsi)
    parsf = float32(parsf)
    return (parsi, parsf)
