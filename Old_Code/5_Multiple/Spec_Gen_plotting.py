import pickle
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import PowerLawModel
from lmfit import Parameters

# File name: {file in}_{quantity varied}_{no. values of varied quantity}-{dateTtime it finished processing}.pickle
with open('output_files/Ha_SII_vel_100-20240927T094947.pickle', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

# Standard deviations of the data
stds = []

arr_in = []

# Which value is varied
variant = 'vel'
if variant == 'sig':
    # sig_in varied
    label = '(sig_out - sig_in)/sig_in'
    param_out = 10
    param_in = 9
elif variant == 'vel':
    # sig_in varied
    label = '(vel_out - vel_in)/vel_in'
    param_out = 13
    param_in = 12
    
for i in range(data[-2]):
    rel_diff = (data[i][param_out][0] - data[i][param_in][0]) / data[i][param_in][0]
    plt.title(rf'{label} against A/N of H$\alpha$ for Nsim = 1000'+f'\nfor {variant}_in={data[i][12][0][0]}')
    plt.scatter(data[i][4], rel_diff, s=1)
    plt.axhline(y=-1, color=(0.8, 0.8, 0.8, 0.5))
    plt.xlabel('A/N')
    plt.ylabel(label)
    plt.xlim(-0.2, 11)
    plt.ylim(-5, 5)
    plt.show()
    
    arr_in.append(data[i][param_in][0][0])
    stds.append(np.std(rel_diff))
    print(f'{variant}_in = {data[i][param_in][0][0]}, std of data = {stds[i]}')


# pfit = Parameters()

# mod = PowerLawModel()

# fitted = mod.fit(stds, x=arr_in)

# plt.plot(arr_in, fitted.best_fit)

plt.scatter(arr_in, stds, s=1)
plt.yscale('log')
plt.xscale('log')
plt.title(f'{variant}_in against the standard deviation of {label} ($\sigma$)')
plt.xlabel(f'{variant}_in')
plt.ylabel('$\sigma$')