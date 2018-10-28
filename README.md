# LpRadon
Log-polar based method for tomography reconstruction.

## Installation
export CUDAHOME=<path to cuda>

python setup.py install

## Dependency 
cupy - for GPU acceleration of linear algebra operations in iterative schemes. See (https://cupy.chainer.org/). For installation use

conda install -c anaconda cupy


## Tests
See tests/

Run ./tests/run_test.py to perform all unit tests

## Wrapper in tomopy
See tomopy/tomopy/recon/wrappers.py file for a wrapper to the lprec library. Also see tomopy/doc/demo/lprec.ipynb jupyter notebook for functionality demonstration. The notebook shows examples of reconstruction by FBP, gradient-descent, conjugate gradient, TV, and EM methods.   

## FBP and Iterative schemes

lprec/lpmethods.py module contains FBP reconstruciton function and iterative schemes implemented with using the log-polar based method. Iterative schemes are written in python with using cupy module for GPU acceleration of linear algebra operations. Access to gpu data inside the lprec library works via pointers to gpu memory.

## Contribution
For adding your own iterative methods based on forward and backward projection operators use module lpmethods.py


