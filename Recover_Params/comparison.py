import numpy as np
import sys
import os
import matplotlib.pyplot as plt

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum

# Parameters
Nsim = 10
param = 'sig'
errorbar = False
line = 4

spec = Spectrum('lines_in.txt', 'fitting.txt', Nsim=Nsim)
spec_missing = Spectrum('lines_in.txt', 'fitting.txt', Nsim=Nsim)

# Run simulation
spec.simulation(plotting=False)

# Save the data
spec.output(overwrite=False, outfile='peak_data_out.pickle')

# With missing lines
spec_missing.simulation_false(plotting=False)

spec_missing.output(overwrite=False, outfile='peak_data_out_missing.pickle', matrices=False)





spec = Spectrum('lines_in.txt', 'fitting.txt')
data = spec.read_pickle('peak_data_out.pickle')
spec.read_pickle('peak_data_out.pickle')
data_in = spec.pickle_in
spec.overwrite_all(data_in)


spec_missing = Spectrum('lines_in.txt', 'fitting.txt')
data = spec_missing.read_pickle('peak_data_out_missing.pickle')
spec_missing.read_pickle('peak_data_out_missing.pickle')
data_in = spec_missing.pickle_in
spec_missing.overwrite_all(data_in)



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
plt.scatter(spec.AoNs_out[line], array[line], s=0.5, zorder=2.5, label='No missing lines')
plt.scatter(spec_missing.AoNs_out[line], array_missing[line], s=0.5, zorder=2.5, label='Missing lines')
if errorbar == True:
    plt.errorbar(spec.AoNs_out[line], array[line], unc[line], fmt='none', zorder=2.5)
    plt.errorbar(spec_missing.AoNs_out[line], array_missing[line], unc_missing[line], fmt='none', zorder=2.5)
plt.xlabel('A/N')
plt.ylabel(label)
plt.legend()
plt.xlim([-0.2, 10])
plt.ylim([-5, 5])
plt.show()


# Density plots 
step = 1/16


# Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
# Where AoN <= 12, gives array of all 2D indices that have this. Remove any index that doesn't appear 8 times (peaks_no times)
ind = np.where(np.unique(np.argwhere(spec.AoNs_out < 12)[:,1], return_counts=True)[1] == spec.peaks_no)[0]
ind_missing = np.where(np.unique(np.argwhere(spec_missing.AoNs_out < 12)[:,1], return_counts=True)[1] == spec_missing.peaks_no)[0]
        
interest_array = array[line][ind]
interest_array_missing = array_missing[line][ind_missing]
interest_AoN = spec.AoNs_out[line][ind]
interest_AoN_missing = spec_missing.AoNs_out[line][ind_missing]

# Also keep d[param]/[param] between -5 and 5
ind = np.where(np.abs(interest_array) <=5)
ind_missing = np.where(np.abs(interest_array_missing) <=5)

interest_array = interest_array[ind]
interest_array_missing = interest_array_missing[ind_missing]
interest_AoN = interest_AoN[ind]
interest_AoN_missing = interest_AoN_missing[ind_missing]

x_vals = np.arange(np.floor(min(min(interest_AoN), min(interest_AoN_missing))), np.ceil(max(max(interest_AoN), max(interest_AoN_missing))), step)
y_vals = np.arange(np.floor(min(min(interest_array), min(interest_array_missing))), np.ceil(max(max(interest_array), max(interest_array_missing))), step)

no_points = np.empty((len(x_vals) - 1, len(y_vals) - 1))
no_points_missing = np.empty((len(x_vals) - 1, len(y_vals) - 1))

for i in range(1, len(x_vals)):
    ind_x = (interest_AoN < x_vals[i]) * (interest_AoN > x_vals[i] - step)
    ind_x_missing = (interest_AoN_missing < x_vals[i]) * (interest_AoN_missing > x_vals[i] - step)

    for j in range(1, len(y_vals)):
        ind_y = (interest_array < y_vals[j]) * (interest_array > y_vals[j] - step)
        ind_y_missing = (interest_array_missing < y_vals[j]) * (interest_array_missing > y_vals[j] - step)
        
        no_points[i-1][j-1] = len(interest_AoN[ind_y*ind_x])
        no_points_missing[i-1][j-1] = len(interest_AoN_missing[ind_y_missing*ind_x_missing])
        
label = f'({param}_out - {param}_in)/{param}_in'

fig, ax = plt.subplots(2)


fig.suptitle(rf'{label} against A/N of peak {line}')
fig.supylabel(label)
ax[0].set_title(f'No missing lines, Nsim = {spec.Nsim}')
pc = ax[0].pcolormesh(x_vals, y_vals, no_points.T, cmap='plasma', vmax=min(no_points.max(), np.mean(no_points[no_points > 0]) + 3*np.std(no_points[no_points > 0])))
fig.colorbar(pc, ax=ax[0])
ax[0].set_xlabel('A/N')

ax[1].set_title(f'Missing lines, Nsim = {spec_missing.Nsim}')
pc = ax[1].pcolormesh(x_vals, y_vals, no_points_missing.T, cmap='plasma', vmax=min(no_points_missing.max(), np.mean(no_points_missing[no_points_missing > 0]) + 3*np.std(no_points_missing[no_points_missing > 0])))
fig.colorbar(pc, ax=ax[1])
ax[1].set_xlabel('A/N')

plt.tight_layout()

plt.show()



