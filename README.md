# LpRadon
Log-polar based method for tomography reconstruction.

## Installation
python setup.py install

## Dependencies
cupy, scikit-build

## Tests
See tests/

Run ./tests/run_test.py to perform all unit tests

## Wrapper in tomopy
See tomopy/tomopy/recon/wrappers.py file for a wrapper to the lprec library. Also see tomopy/doc/demo/lprec.ipynb jupyter notebook for functionality demonstration. The notebook shows examples of reconstruction by FBP, gradient-descent, conjugate gradient, TV, and EM methods.   

## FBP and Iterative schemes

lprec/lpmethods.py module contains FBP reconstruciton function and iterative schemes implemented with using the log-polar based method. Iterative schemes are written in python with using cupy module for GPU acceleration of linear algebra operations. Access to gpu data inside the lprec library works via pointers to gpu memory.


