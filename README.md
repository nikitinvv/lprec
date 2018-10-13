# LpRadon
Log-polar based method for tomography reconstruction.

## Installation
export CUDAHOME=/usr/local/cuda-8.0
python setup.py install

## Dependency 
cupy - for GPU acceleration of iterative schemes. See (https://cupy.chainer.org/). For installation use

conda install -c anaconda cupy


## Tests
See tests/

## Wrapper in tomopy
See tomopy/tomopy/recon/wrappers.py file for a wrapper to the lprec library. Also see tomopy/doc/demo/lprec.ipynb jupyter notebook for functionality demonstration. The notebook shows examples of reconstruction by FBP, gradient-descent, conjugate gradient, TV, and EM methods.   

## Iterative schemes

lprec/iterative.py module contains iterative schemes implemented with using the log-polar based method. Iterative schemes are written in python with using cupy module for GPU acceleration of linear algebra operations. Access to gpu data inside the lprec library works via pointers to gpu memory.
