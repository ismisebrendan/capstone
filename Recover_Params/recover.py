import numpy as np
import sys
import os
import matplotlib.pyplot as plt
%matplotlib inline

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum

# spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=10)

# # Run simulation
# spec.simulation(plotting=False)

# # Save the data
# spec.output(overwrite=False, outfile='spectra_data_out.pickle', matrices=True)

# # With missing lines
# spec.simulation_false(plotting=False)

# spec.output(overwrite=False, outfile='peak_data_out_missing.pickle', matrices=False)


# Import the data
spec_in = Spectrum('lines_in.txt', 'fitting.txt')
spec_in.read_pickle('spectra_data_out_drive.pickle')
data_in = spec_in.pickle_in
spec_in.overwrite_all(data_in)


for i in [(4, 0), (1, 2), (5, 3), (7, 6)]:
    for j in ['sig', 'vel', 'flux']:
        for val in ['std', 'median']:
            spec_in.scatter_size(param=j, line=i[1], step=1, value=val, brightest=i[0])




# for i in range(8):
#     for j in ['sig', 'vel', 'flux']:
#         for val in ['std', 'median']:
#             spec_in.scatter_size(param=j, line=i, step=1, value=val)
    
# transparency = False

# # Plot heatmaps
# for i in range(8):
#     for j in ['sig', 'vel', 'flux']:
#         for val in ['std', 'median']:
#             spec_in.heatmap_brightest(param=j, line=i, transparency=transparency, step=1, value=val)

# for i in range(8):
#     for j in ['sig', 'vel', 'flux']:
#         for val in ['std', 'median']:
#             spec_in.heatmap_sum(param=j, line=i, transparency=transparency, step=1, text=False, value=val)