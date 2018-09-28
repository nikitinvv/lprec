# LpRadon
Log-polar based method for tomography reconstruciton

## Installation
export CUDAHOME=/usr/local/cuda-8.0
python setup.py install

## Tests
See tests/

## Wrapper in tomopy
import tomopy

obj = tomopy.shepp3d() # Generate an object.

ang = tomopy.angles(180) # Generate uniformly spaced tilt angles.

sim = tomopy.project(obj, ang) # Calculate projections.

rec = tomopy.recon(sim, ang, algorithm=tomopy.lprec,
      lpmethod='fbp', filter_name='parzen', interp_type='cubic', ncore=1)

import pylab

pylab.imshow(rec[64], cmap='gray')

pylab.show()
