import cupy as cp
import numpy as np

def grad(lp, init_recon, tomo0, num_iter, reg_par,gpu):
    """
    Reconstruction with the gradient descent method
    with the regularization parameter reg_par,
    if reg_par<0, then reg_par is computed on each iteration
    Solving the problem 1/2||R(recon)-tomo||^2_2 -> min
    """

    cp.cuda.Device(gpu).use()
    #Allocating necessary gpu arrays
    recon = cp.array(init_recon)
    tomo = cp.array(tomo0)
    g = cp.zeros(tomo.shape,dtype="float32")

    recon0 = recon
    grad = recon*0
    grad0 = recon*0

    for i in range(0, num_iter):
        lp.fwdp(g,recon,gpu)
        lp.adjp(grad,2*(g-tomo),gpu)
        if(reg_par < 0):
            if(i == 0):
                lam = 1e-3*cp.ones(tomo.shape[0],dtype="float32")
            else:
                lam = cp.sum(cp.sum(((recon-recon0)*(grad-grad0)), 1), 1) / \
                    cp.sum(cp.sum(((grad-grad0)*(grad-grad0)), 1), 1)
        else:
            lam = reg_par*cp.ones(tomo.shape[0],dtype="float32")
        lam = cp.array(lam)
        recon0 = recon.copy()
        grad0 = grad.copy()

        recon = recon - cp.array(cp.reshape(lam, [tomo.shape[0], 1, 1]))*grad

    return recon.get()

def cg(lp, init_recon, tomo0, num_iter, reg_par,gpu):
    """
    Reconstruction with the conjugate gradient method
    Solving the problem 1/2||R(recon)-tomo||^2_2 -> min
    """
    cp.cuda.Device(gpu).use()
    #Allocating necessary gpu arrays
    recon = cp.array(init_recon)
    tomo = cp.array(tomo0)
    b = cp.zeros(recon.shape,dtype="float32")
    f = cp.zeros(recon.shape,dtype="float32")
    g = cp.zeros(tomo.shape,dtype="float32")

    #Right side R^*(tomo)
    lp.adjp(b,tomo)

    #residual r = b - R^*(R(recon))
    lp.fwdp(g,recon); lp.adjp(f,g)
    r = b-f
    p = r.copy()
    rsold = cp.sum(r*r)
    #cg iterations
    for i in range(0,num_iter):
        lp.fwdp(g,p,gpu); lp.adjp(f,g,gpu)
        alpha = rsold/cp.sum(p*f)
        recon = recon+alpha*p 
        r = r-alpha*f
        rsnew = cp.sum(r*r)
        p = r+(rsnew/rsold)*p
        rsold = rsnew

    return recon.get()

def em(lp,init_recon,tomo0,num_iter,reg_par,gpu):
    """
    Reconstruction with the Expectation Maximization algorithm for denoising
    with parameter reg_par manually chosen for avoiding division by 0.
    Maximization of the likelihood function L(tomo,rho) 

    """   

    cp.cuda.Device(gpu).use()

    #Allocating necessary gpu arrays
    recon = cp.array(init_recon)
    tomo = cp.array(tomo0)
    xi = cp.zeros(recon.shape,dtype="float32")
    g = cp.zeros(tomo0.shape,dtype="float32")
    upd = cp.zeros(recon.shape,dtype="float32")
    #Constructing iterative scheme
    eps = reg_par
    # R^*(ones)
    lp.adjp(xi, tomo*0+1,gpu)
    xi = xi+1e-5
    # modification for avoing division by 0
    e = cp.float32(eps)
    #grad iteratins
    for i in range(0, num_iter):
        lp.fwdp(g,recon,gpu)
        lp.adjp(upd,tomo/(g+e),gpu)
        recon = recon*(upd/xi)

    return recon.get()

def tv(lp, init_recon, tomo0, num_iter, reg_par, gpu):
    """
    Reconstruction with the total variation method
    with the regularization parameter reg_par.
    Solving the problem 1/2||R(recon)-tomo||^2_2 + reg_par*TV(recon) -> min
    """
    cp.cuda.Device(gpu).use()
    #Allocating necessary gpu arrays
    recon = cp.array(init_recon)
    tomo = cp.array(tomo0)
    g = cp.zeros(tomo.shape,dtype="float32")
    p = cp.zeros(recon.shape,dtype="float32")

    lam = reg_par
    c = 0.35  # 1/power_method(lp,tomo,num_iter)

    recon0 = recon
    prox0x = recon*0
    prox0y = recon*0
    div0 = recon*0
    prox1 = tomo*0

    for i in range(0, num_iter):
        # forward step
        # compute proximal prox0
        prox0x[:, :, :-1] += c*(recon[:, :, 1:]-recon[:, :, :-1])
        prox0y[:, :-1, :] += c*(recon[:, 1:, :]-recon[:, :-1, :])
        nprox = cp.array(cp.maximum(1, (cp.sqrt(prox0x*prox0x+prox0y*prox0y)/lam)))
        prox0x = prox0x/nprox
        prox0y = prox0y/nprox
        # compute proximal prox1
        lp.fwdp(g,recon,gpu)
        prox1 = (prox1+c*g-c*tomo)/(1+c)

        # backward step
        recon = recon0
        div0[:, :, 1:] = (prox0x[:, :, 1:]-prox0x[:, :, :-1])
        div0[:, :, 0] = prox0x[:, :, 0]
        div0[:, 1:, :] += (prox0y[:, 1:, :]-prox0y[:, :-1, :])
        div0[:, 0, :] += prox0y[:, 0, :]
        lp.adjp(p,prox1,gpu)
        recon0 = recon0-c*p+c*div0

        # update recon
        recon = 2*recon0 - recon

    return recon.get()

