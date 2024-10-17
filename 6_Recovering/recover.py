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

# Save the data
spec.output(overwrite=False)

# Import the data
spec_in = Spectrum('lines_in.txt', 'fitting.txt')
data = spec_in.read_pickle('peak_data_out.pickle')
spec_in.read_pickle('peak_data_out.pickle')
data_in = spec_in.pickle_in
spec_in.overwrite_all(data_in)




# Plot heatmaps
for i in range(8):
    spec_in.heatmap_brightest(param='sig', line=i)
    spec_in.heatmap_brightest(param='vel', line=i)
    spec_in.heatmap_brightest(param='flux', line=i)

for i in range(8):
    spec_in.heatmap_sum(param='sig', line=i, text=False)
    spec_in.heatmap_sum(param='vel', line=i, text=False)
    spec_in.heatmap_sum(param='flux', line=i, text=False)