# 
# Generate a spectral line for the ANN to learn from
# 

import numpy as np
import sys
import os

spec_path = os.path.abspath('../..') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum


Nsim = 2000

# Generate Spectrum object
spec = Spectrum('../one_line.txt', '../fitting.txt', Nsim=Nsim)

spec.get_data()

# Generate actual amplitudes
print('Generating spectral data')
y_vals = spec.generate()

data_out = np.transpose(np.concatenate((y_vals.T, np.ones((1, Nsim)))))

print('Generating false positives')
spec.AoN_min = 0
spec.AoN_max = 0
y_vals = spec.generate()

# Save data
empty_data = np.transpose(np.concatenate((y_vals.T, np.zeros((1, Nsim)))))
data_out = np.concatenate((data_out, empty_data))

# np.random.shuffle(data_out)


files_present = [f for f in os.listdir() if os.path.isfile(os.path.join("./one_line", f)) and f[:7] == "spectra" and f[-4:] == ".txt"]
file_out = f'one_line/spectra{len(files_present)}.txt'


np.savetxt(file_out, data_out)

