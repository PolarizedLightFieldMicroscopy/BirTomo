from random import Random
import numpy as np

rng = Random()

def get_ellipsoid(vox):
    if True:
        from random import Random
        Delta_n = rng.uniform(-1,1)
        a = (rng.uniform(-5,5), rng.uniform(-5,5), rng.uniform(-5,5))
        opticAxis = a / np.linalg.norm(a)
    else:
        Delta_n = 1
        opticAxis = np.array([1,0,0])
    return Delta_n, opticAxis