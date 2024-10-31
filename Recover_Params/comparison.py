import numpy as np
import sys
import os
import matplotlib.pyplot as plt

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum

# Parameters
Nsim = 1000
param = 'sig'
errorbar = False
line = 4

spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=Nsim)
spec_missing = Spectrum('lines_in.txt', 'fitting.txt', Nsim=Nsim)


# Import data
spec.get_data()
spec_missing.get_data()

# Run simulation
spec.simulation(plotting=False)

# Save the data
spec.output(overwrite=False, outfile='peak_data_out_missing.pickle')

# With missing lines
spec_missing.simulation_false(plotting=False)

spec_missing.output(overwrite=False, outfile='peak_data_out_missing.pickle', matrices=False)

# Plotting
if param == 'sig':
    array = (spec.sig_out - spec.sig_in) / spec.sig_in
    array_missing = (spec_missing.sig_out - spec_missing.sig_in) / spec_missing.sig_in
    unc = spec.sig_unc_out / spec.sig_in
    unc_missing = spec_missing.sig_unc_out / spec_missing.sig_in
elif param == 'vel':
    array = (spec.vels_out - spec.vels_in) / spec.vels_in
    array_missing = (spec_missing.vels_out - spec_missing.vels_in) / spec_missing.vels_in
    unc = spec.vels_unc_out / spec.vels_in
    unc_missing = spec_missing.vels_unc_out / spec_missing.vels_in
elif param == 'A':
    array = (spec.As_out - spec.As_in) / spec.As_in
    array_missing = (spec_missing.As_out - spec_missing.As_in) / spec_missing.As_in
    unc = spec.As_unc_out / spec.As_in
    unc_missing = spec_missing.As_unc_out / spec_missing.As_in
elif param == 'flux':
    array = (spec.f_out - spec.f_in) / spec.f_in
    array_missing = (spec_missing.f_out - spec_missing.f_in) / spec_missing.f_in
    unc = spec.f_unc_out / spec.f_in
    unc_missing = spec_missing.f_unc_out / spec_missing.f_in

fig, ax = plt.subplots()

label = f'({param}_out - {param}_in)/{param}_in'
plt.title(rf'{label} against A/N of peak {line} for Nsim = {spec.Nsim}'+f'\nv_in = {spec.peak_params[0][3]}, sig_in = {spec.peak_params[0][4]}')
plt.axhline(0, color='lightgrey')
plt.scatter(spec.AoNs_out[line], array[line], s=0.5, zorder=2.5, label='No missing mines')
plt.scatter(spec_missing.AoNs_out[line], array[line], s=0.5, zorder=2.5, label='Missing mines')
if errorbar == True:
    plt.errorbar(spec.AoNs_out[line], array[line], unc[line], fmt='none', zorder=2.5)
    plt.errorbar(spec_missing.AoNs_out[line], array[line], unc[line], fmt='none', zorder=2.5)
plt.xlabel('A/N')
plt.ylabel(label)
plt.legend()
plt.xlim([-0.2, 10])
plt.ylim([-5, 5])
plt.show()


