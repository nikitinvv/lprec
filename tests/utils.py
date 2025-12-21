import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tifffile
import os
from concurrent.futures import wait
from functools import wraps
import time
import psutil
from matplotlib_scalebar.scalebar import ScaleBar

def mshow(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        im = axs.imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_complex(a,show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(14,6))
        im = axs[0].imshow(a.real, cmap="gray", **args)
        scalebar = ScaleBar(0.015, "um", length_fraction=0.25, font_properties={
            "family": "serif",
        },  # For more information, see the cell below
        location="lower right")
        axs[0].add_artist(scalebar)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(a.imag, cmap="gray", **args)
        scalebar = ScaleBar(0.015, "um", length_fraction=0.25, font_properties={
            "family": "serif",
        },  # For more information, see the cell below        
        location="lower right")
        axs[1].add_artist(scalebar)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

def mshow3(a,b,c,show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        if isinstance(b, cp.ndarray):
            b = b.get()
        if isinstance(c, cp.ndarray):
            c = c.get()
            
        fig, axs = plt.subplots(1, 3, figsize=(18,5))
        im = axs[0].imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(b, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[2].imshow(c, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_polar(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(18,6))
        im = axs[0].imshow(np.abs(a), cmap="gray", **args)
        axs[0].set_title("abs")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(np.angle(a), cmap="gray", **args)
        axs[1].set_title("phase")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

def mplot_positions(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        plt.plot(a[:,1],a[:,0],'.')
        plt.grid()
        plt.axis('square')
        plt.show()
    
def reprod(a, b):
    return a.real * b.real + a.imag * b.imag


def redot(a, b, axis=None):
    res = cp.sum(reprod(a, b), axis=axis)
    return res

def chunking_flg(pars):
    return  any(isinstance(a, np.ndarray) for a in pars)
    

def write_tiff(a, name, **args):
    if isinstance(a, cp.ndarray):
        a = a.get()
    os.makedirs(os.path.dirname(name), exist_ok=True)
    tifffile.imwrite(name+'.tiff', a)

def read_tiff(name):
    a = tifffile.imread(name)[:]
    return a

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        process = psutil.Process(os.getpid())
        # if (end_time - start_time>0.1):
        #     print(f"{func.__name__}: {end_time - start_time:.4f} sec, {process.memory_info().rss / (1024**3):.2f} GB", flush=True)
        return result
    return wrapper

def downsample(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[..., ::2,:]+res[..., 1::2,:])
        res = 0.5*(res[..., :, ::2]+res[..., :, 1::2])
    return res