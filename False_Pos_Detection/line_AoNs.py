import os
import sys
import numpy as np
import matplotlib.pyplot as plt

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum


spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=100, AoN_min=5, AoN_max=5)





spec.get_data()

spec.simulation(plotting=False)

# Histogram
AoN_out = spec.AoNs_out

for i in range(1,spec.peaks_no):
    hbin_out = np.linspace(min(AoN_out[i]), max(AoN_out[i]), 11)
    # hbin_in = np.linspace(min(spec.AoNs), max(spec.AoNs), 11)
    
    plt.hist(AoN_out[i], bins=hbin_out, density=False, histtype='step', label=f'A/N out, line {i}')
    
    # avg_in = np.mean(plt.hist(spec.AoNs, bins=hbin_in, density=False, histtype='step', label='A/N in')[0])
    
    # plt.axhline(avg, linestyle=':', color='k', label='mean of A/N out')
    # plt.axhline(avg_in, linestyle='-.', color='k', label='mean of A/N in')

plt.legend()