import numpy as np
import sys
import os
import matplotlib.pyplot as plt

%matplotlib qt

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum

spec_a = Spectrum('lines_in.txt', 'fitting.txt', Nsim=10)

# Run simulation
spec_a.simulation(plotting=False)

# Save the data
spec_a.output(overwrite=False, matrices=True, outfile='spectra_data_out.pickle')

# Import the data
spec_a = Spectrum('lines_in.txt', 'fitting.txt')
data = spec_a.read_pickle('spectra_data_out.pickle')
spec_a.read_pickle('spectra_data_out.pickle')
data_in = spec_a.pickle_in

spec_a.overwrite_all(data_in)


spec_a.plot_results(line=4, param='flux', interactive=True)


spec_a.heatmap_brightest(param='flux', line=0, value='median', text=False, step=0.5, interactive=True)

spec_a.scatter_size(param='flux', line=0, value='median', step=0.5, interactive=True)