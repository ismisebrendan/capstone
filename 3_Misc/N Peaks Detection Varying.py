import numpy as np
import pickle
import matplotlib.pyplot as plt
from lmfit.models import Model
from lmfit import Parameters

def background(x, bkg):
    """
    Generates the background level of the spectrum.

    Parameters
    ----------
    x : array
        The x data over which the ackground should be generated.
    bkg : float
        The level of the background spectrum.

    Returns
    -------
    array
        The background level for each of the input x values.

    """
    return x*0 + bkg

def gaussian(x, A, u, sig, sig_resolution):
    """
    Produces a Gaussian curve with a background level.
    
    Parameters
    ----------
    x : float
        The x range over which the Gausssian.
    A : float
        The amplitude of the Gaussian.
    u : float
        The position of the Gaussian peak.
    sig : float
        The sigma of the Gaussian.
    sig_resolution : float
        The resolution of the detector.

    Returns
    -------
    array
        The Gaussian.

    """
    sig_obs = np.sqrt(sig**2 + sig_resolution**2)
    return A * np.exp(-0.5*(x - u)**2 / sig_obs**2)

# Redundant
# def gaussian_plus_background(x, A, u, sig, sig_resolution, bkg):
#     return gaussian(x, A, u, sig, sig_resolution) + background(x, bkg)

# Redundant
# def gaussian_plus_noise(x, A, u, sig, sig_resolution, bkg, N):
#     noise = N * np.random.randn(len(x)) * np.sqrt(gaussian_plus_background(x, A, u, sig, sig_resolution, bkg))
#     return gaussian_plus_background(x, A, u, sig, sig_resolution, bkg) + noise

# Define peak parameters from a .txt file
peak_params = []
# peaksdata = 'peaksdata.txt'
# peaksdata = 'peaksdata_1peaks.txt'
# peaksdata = 'peaksdata_2peaks.txt'
# peaksdata = 'peaksdata_3peaks.txt'
# peaksdata = 'peaksdata_4peaks.txt'
# peaksdata = 'peaksdata_4peaks_test.txt'
peaksdata = 'peaksdata_H_OIII.txt'
# peaksdata = 'peaksdata_vary.txt'

# Is a peak free or dependant on another one
dependant = []

with open(peaksdata) as f:
    data = f.readlines()
    for i in range(len(data)):
        entry = data[i].split()
        # Convert data entries from string to float
        param = [float(p) for p in entry[1:]]
        peak_params.append(param)
        
        # If preceded by f reference itself, otherwise reference specified line
        if entry[0] == 'f':
            dependant.append(i)
        elif entry[0][0] == 'd':
            dependant.append(int(entry[0][1]))

# Initialize parameters
sig_resolution = 0.05
sig_sampling = 4.0
bkg = 100 # Background level
peaks_no = len(peak_params)

# Set up centre offsets
u_offsets = [0]
for i in range(1, peaks_no):
    u_offsets.append(peak_params[i][0] - peak_params[0][0])

# Set up amplitude ratios
A_ratio = []
for i in range(peaks_no):
    A_ratio.append(peak_params[i][2])

# Create x values
u_min = min(p[0] for p in peak_params)
u_max = max(p[0] for p in peak_params)
sig = max(p[1] for p in peak_params)
dx = sig_resolution / sig_sampling
nx = int(2 * (20*sig/dx) + 1)
x = np.linspace(-20 * sig + u_min, 20 * sig + u_max, nx)

# Initialize model
model = background(x, bkg)
gaussian_models = []
mod = Model(background, prefix='bkg_')
# Loop through for number of peaks
for i, (u, sig, relative_A, cL, cU) in enumerate(peak_params):
    gauss = Model(gaussian,prefix=f'g{i+1}_')
    mod += gauss
    gaussian_models.append(gauss)

# Run Nsim simulations between A/N = A/N_min and A/N = A/N_max
Nsim = 10
AoN_min = 0
AoN_max = 10
AoNs = np.random.random(Nsim) * (AoN_max - AoN_min) + AoN_min
Nsim_per_AoN = 1
poor_fits = 0

# Initialize input arrays
As_in = np.empty(Nsim)
us_in = np.empty(Nsim)
sigs_in = np.empty(Nsim)
# Initialize output arrays
As_out = np.empty((peaks_no, Nsim))
us_out = np.empty((peaks_no, Nsim))
sigs_out = np.empty((peaks_no, Nsim))
AoNs_out = np.empty((peaks_no, Nsim))

# Simulation
for index, AoN in enumerate(AoNs):
    A = np.sqrt(bkg) * AoN
    for _ in range(Nsim_per_AoN):
        # Generate Gaussian + Noise data
        model = background(x, bkg)
        # Do separately for free and dependant lines
        amplitudes = []
        for (u, sig, relative_A, cL, cU), ratio, i in zip(peak_params, A_ratio, range(peaks_no)):
            if dependant[i] == i:
                amplitudes.append(ratio)
                model += gaussian(x, A * ratio, u, sig, sig_resolution)
            else:
                amplitudes.append(np.nan)
                
        for (u, sig, relative_A, cL, cU), ratio, i in zip(peak_params, A_ratio, range(peaks_no)):
            if np.isnan(amplitudes[i]):
                ratio = ratio * amplitudes[dependant[i]]
                model += gaussian(x, A * ratio, u, sig, sig_resolution)

        noise = np.random.randn(len(x)) * np.sqrt(model)
        y = model + noise

        # Fit with LMfit
        pfit = Parameters()
    
        # The background level
        pfit.add('bkg_bkg', value=bkg, vary=True)
        
        # Setting up parameters for the first peak (g1)
        # If the line is free its amplitude can be defined now
        if dependant[0] == 0:
            pfit.add('g1_A', value=np.max(y) - np.median(y), min=0.0)
        pfit.add('g1_sig', value=sig)

        pfit.add('g1_u', value=peak_params[0][0])
        pfit.add(name='g1_sig_resolution', value=sig_resolution, vary=False)
        
        # Looping through number of peaks to generate fits for the remaining peaks
        for i in range(1, peaks_no):
            # If the line is free its amplitude can be defined now
            if dependant[i] == i:
                pfit.add(f'g{i+1}_A', value=np.max(y) - np.median(y), min=peak_params[i][3])
            pfit.add(f'g{i+1}_sig', expr='g1_sig') # value=sig) # 

                
            pfit.add(f'g{i+1}_u', expr=f'g1_u + {u_offsets[i]}')
            pfit.add(f'g{i+1}_sig_resolution', value=sig_resolution, vary=False)

        
        # Loop again for amplitudes and sigmas of dependant peaks - have to do afterwards as could be dependant on a line appearing after it in the file
        for i in range(peaks_no):
            if dependant[i] != i:
                # Is it a fixed ratio or a range of values
                if peak_params[i][3] == peak_params[i][4]:
                    pfit.add(f'g{i+1}_A', expr=f'g{dependant[i]+1}_A * {A_ratio[i]}')
                else:
                    pfit.add(f'delta{i+1}', value = (peak_params[i][3] + peak_params[i][4]) / 2, min = peak_params[i][3], max = peak_params[i][4], vary = True)
                    pfit.add(f'g{i+1}_A', expr=f'delta{i+1} * g{dependant[i]+1}_A')
                # pfit.add(f'g{i+1}_sig', expr=f'g{dependant[i]+1}_sig')
                    
        fit = mod.fit(y, pfit, x=x)

        # fill in arrays for input and output Gaussian parameter and A/N
        As_in[index] = A

        for peak in range(peaks_no):
            As_out[peak][index] = np.abs(fit.params[f'g{peak+1}_A'].value)
        us_in[index] = u
        for peak in range(peaks_no):
            us_out[peak][index] = fit.params[f'g{peak+1}_u'].value
        sigs_in[index] = sig
        for peak in range(peaks_no):
            sigs_out[peak][index] = np.abs(fit.params[f'g{peak+1}_sig'].value)
            
        
        # Show each single emission-line model and fit
        # Comment out to hide fits if running a large number of simulations (e.g. ~1000)
        labels = ['input model + noise', 'input model', 'fitted model']
        plt.plot(x, y, 'k-')
        plt.plot(x, model, 'c-')
        plt.plot(x, fit.best_fit, 'r-')
        plt.legend(labels)
        plt.grid()
        plt.show()
        
        
        # Uncomment to show poor fits
        # if sigs_out[peak][index] < 0.2:
        #     # poor_fits += 1
        #     labels = ['input model + noise', 'input model', 'fitted model']
        #     plt.title('Poor fit')
        #     plt.plot(x, y, 'k-')
        #     plt.plot(x, model, 'c-')
        #     plt.plot(x, fit.best_fit, 'r-')
        #     plt.legend(labels)
        #     plt.grid()
        #     plt.show()
        #     print(sigs_out[peak][index])
        




# Get measured A/N values
for peak in range(peaks_no):
    AoNs_out[peak] = As_out[peak]/np.sqrt(bkg)
AoNs_out_max = np.empty(Nsim)
AoNs_out_avg = np.empty(Nsim)
for index, AoNs in enumerate(zip(*AoNs_out)):
    AoNs_out_max[index] = max(AoNs)
    AoNs_out_avg[index] = np.mean(AoNs)

# Get in and out flux values
Fs_in = As_in*np.sqrt(2*np.pi)*sigs_in
Fs_out = As_out*np.sqrt(2*np.pi)*sigs_out

# Compute in-out parameter differences
dAs = As_out - As_in
dus = us_out - us_in
dsigs = sigs_out - sigs_in
dFs = Fs_out - Fs_in

# Serialize the data in a file
data = [As_in, As_out, AoNs_out, us_in, us_out, sigs_in, sigs_out, Fs_in, Fs_out, dAs, dus, dsigs, dFs, AoNs_out_avg, AoNs_out_max, peaks_no]
filename = 'npeaksdata_n=5'
outfile = open(filename, 'wb')
pickle.dump(data,outfile)
outfile.close()

# Plot dsigs/sigs_in vs. AoNs
array = dsigs / sigs_in
i=(np.abs(AoNs_out_max-5) <0.5)
print(np.std(array[0][i]))
label = '(sig_out - sig_in)/sig_in'
plt.title(f'{label} against A/N for Nsim = {Nsim}')
plt.scatter(AoNs_out_max, array[0], s=0.5)
plt.scatter(AoNs_out_max[i], array[0][i], s=0.5)
plt.xlabel('A/N')
plt.ylabel('%s' % label)
std = np.std(array)
plt.xlim(-0.2, 11)
plt.ylim(-5, 5)
plt.show()

# print(f'{poor_fits} poor fits, {poor_fits/Nsim * 100}% of fits')