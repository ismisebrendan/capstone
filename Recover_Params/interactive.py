import numpy as np
import sys
import os
import matplotlib.pyplot as plt

%matplotlib qt

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum

spec_a = Spectrum('lines_in.txt', 'fitting.txt', Nsim=10)

# Import data
spec_a.get_data()

# Run simulation
spec_a.simulation(plotting=False)

# Save the data
spec_a.output(overwrite=True, matrices=True, outfile='spectra_data_out.pickle')

# Import the data
spec_in = Spectrum('lines_in.txt', 'fitting.txt')
data = spec_in.read_pickle('spectra_data_out.pickle')
spec_in.read_pickle('spectra_data_out.pickle')
data_in = spec_in.pickle_in
spec_in.overwrite_all(data_in)


spec_in.plot_results(line=4, param='sig', interactive=True)