import numpy as np
import sys
import os
import matplotlib.pyplot as plt

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum

spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=1000)

# Import data
spec.get_data()

# Run simulation
spec.simulation(plotting=False)

spec.plot_results(line=6, param='sig', xlim=None)
spec.plot_results(line=6, param='flux', xlim=None)