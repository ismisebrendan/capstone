from spectrum_obj import Spectrum
import numpy as np
import pickle
from datetime import datetime

# Input file
file_in = 'lines_in' # .txt

# No. times to repeat
Niter = 100 # No. of different input parameters
Nsim = 100 # No. of times to repeat for each parameter

# Output data
data_out = []

# Which parameter is varied
variant = 'vel'

if variant == 'vel':
    param = 3
elif variant == 'sig':
    param = 4

# Range of values
min_val = 1
max_val = 1000
vals = np.linspace(min_val, max_val, Niter)

# Generate Spectrum object
spec = Spectrum(f'input_files/{file_in}.txt', 'fitting.txt', Nsim=Nsim)

# Import data
spec.get_data()

# Loop
for i in range(Niter):
    print(f'Simulation {i+1}')
    spec.overwrite(param, vals[i])
    spec.simulation(plotting=False)
    
    # Plot the results
    spec.plot_results(peak=2, param=variant)
    spec.plot_results_err(peak=2, param=variant)
    
    # Save the results
    data_out.append(spec.dump())

# Save the data
data_out.append(vals)
data_out.append(Niter)

with open('output_info.txt') as f:
    data_out.append(f.readlines())

date = datetime.now()
filename = f'output_files/{file_in}_{variant}_{Niter}-{date.strftime("%Y%m%dT%H%M%S")}.pickle'
outfile = open(filename, 'wb')
pickle.dump(data_out, outfile)
print(f'Saving {filename}')
outfile.close()