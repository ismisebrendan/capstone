from object import Spectrum
import numpy as np

Niter = 20
vels = np.linspace(0, 1000, Niter)

spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=1, AoN_min=10)

spec.get_data()

for i in range(Niter):
    spec.overwrite(4, vels[i])
    spec.simulation(plotting=True)
    
    print(spec.peak_params[0])
    print(spec.fit_params[0])
    