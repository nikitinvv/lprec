import cupy as cp
import numpy as np

def em(lp,recon0,tomo0,num_iter,reg_par):
    """
    Reconstruction with the Expectation Maximization algorithm for denoising
    with parameter reg_par manually chosen for avoiding division by 0.
    Maximization of the likelihood function L(tomo,rho) 
    """    
    #Allocating necessary gpu arrays
    recon = cp.array(recon0)
    tomo = cp.array(tomo0)
    xi = cp.zeros(recon.shape,dtype="float32")
    g = cp.zeros(tomo0.shape,dtype="float32")
    upd = cp.zeros(recon.shape,dtype="float32")
    
    #Constructing iterative scheme
    eps = reg_par
    # R^*(ones)
    lp.adjp(xi, tomo*0+1)
    # modification for avoing division by 0
    xi = xi+eps*np.pi/2
    e = cp.float32(np.max(tomo.get())*eps)

    for i in range(0, num_iter):
        lp.fwdp(g,recon)
        lp.adjp(upd,tomo/(g+e))
        recon = recon*(upd/xi)

    #copy result from gpu
    recon0 = recon.get()

def tv(lp, recon0, tomo0, num_iter, reg_par):
    """
    Reconstruction with the total variation method
    with the regularization parameter reg_par.
    Solving the problem 1/2||R(recon)-tomo||^2_2 + reg_par*TV(recon) -> min
    """
    #Allocating necessary gpu arrays
    recon = cp.array(recon0)
    tomo = cp.array(tomo0)
    g = cp.zeros(tomo0.shape,dtype="float32")
    p = cp.zeros(recon0.shape,dtype="float32")


    lam = reg_par
    c = 0.35 

    upd = recon
    prox0x = recon*0
    prox0y = recon*0
    div0 = recon*0
    prox1 = tomo*0

    for i in range(0, num_iter):
        # forward step
        # compute proximal prox0

        prox0x[:, :, :-1] += c*(recon[:, :, 1:]-recon[:, :, :-1])
        prox0y[:, :-1, :] += c*(recon[:, 1:, :]-recon[:, :-1, :])
        nprox = cp.maximum(1, cp.sqrt(prox0x*prox0x+prox0y*prox0y)/lam)
        prox0x = prox0x/nprox
        prox0y = prox0y/nprox
        # compute proximal prox1
        lp.fwdp(g,recon)
        prox1 = (prox1+c*g-c*tomo)/(1+c)

        # backward step
        recon = upd
        div0[:, :, 1:] = (prox0x[:, :, 1:]-prox0x[:, :, :-1])
        div0[:, :, 0] = prox0x[:, :, 0]
        div0[:, 1:, :] += (prox0y[:, 1:, :]-prox0y[:, :-1, :])
        div0[:, 0, :] += prox0y[:, 0, :]
        lp.adjp(p,prox1)
        upd = upd-c*p+c*div0

        # update recon
        recon = 2*upd - recon

    return recon