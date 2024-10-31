import numpy as np
import sys
import os
import matplotlib.pyplot as plt
%matplotlib inline

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum

spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=1000)

# Import data
spec.get_data()

# # Run simulation
# spec.simulation(plotting=False)

# # Save the data
# spec.output(overwrite=False, outfile='peak_data_out_missing.pickle')

# With missing lines
spec.simulation_false(plotting=False)
    
spec.output(overwrite=False, outfile='peak_data_out_missing.pickle', matrices=False)

# Import the data
spec_in = Spectrum('lines_in.txt', 'fitting.txt')
data = spec_in.read_pickle('peak_data_out_missing.pickle')
spec_in.read_pickle('peak_data_out_missing.pickle')
data_in = spec_in.pickle_in
spec_in.overwrite_all(data_in)

for i in range(8):
    for j in ['sig', 'vel', 'flux']:
        spec_in.scatter_size(param=j, line=i, step=0.5)
    
transparency = False

# Plot heatmaps
for i in range(8):
    for j in ['sig', 'vel', 'flux']:
        spec_in.heatmap_brightest(param=j, line=i, transparency=transparency, step=1)

for i in range(8):
    for j in ['sig', 'vel', 'flux']:
        spec_in.heatmap_sum(param=j, line=i, transparency=transparency, step=1, text=False)