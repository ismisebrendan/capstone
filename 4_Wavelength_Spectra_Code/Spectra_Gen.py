import numpy as np
import pickle
import matplotlib.pyplot as plt
from lmfit.models import Model
from lmfit import Parameters

# The speed of light
c = 299792.458 #km/s

def background(x, bkg):
    """
    Generates the background level of the spectrum.

    Parameters
    ----------
    lam : array
        The wavelength range over which the background should be generated.
    bkg : float
        The level of the background spectrum.

    Returns
    -------
    array
        The background level for each of the input wavelength values.

    """
    return x*0 + bkg

def gaussian(x, A, lam_rf, vel, sig, sig_resolution):
    """
    Produces a Gaussian curve with a background level.
    
    Parameters
    ----------
    x : float
        The wavelength range over which the Gaussian is generated.
    A : float
        The amplitude of the Gaussian.
    lam_rf : float
        The wavelength of the peak in its rest frame in Angstrom.
    vel : float
        The relative velocity of the source in km/s.
    sig : float
        The sigma of the Gaussian in km/s.
    sig_resolution : float
        The resolution of the detector in Angstrom.

    Returns
    -------
    array
        The Gaussian.

    """
    lam_obs = lam_rf * (1 + vel/c)
    sig_intr = sig / c * lam_obs
    sig_obs = np.sqrt(sig_intr**2 + sig_resolution**2)
    return A * np.exp(-0.5*(x - lam_obs)**2 / sig_obs**2)

# Read in the data
peaksdata = 'lines_in.txt'
# peaksdata = 'peaks.txt'

peak_params = []

# Is a peak free or a doublet of another one - velocity, sigma and amplitude dependant
doublet = []

# Is the peak's velocity or profile dependant on another one
vel_dep = []
prof_dep = []

with open(peaksdata) as f:
    data = f.readlines()
    for i in range(len(data)):
        entry = data[i].split()
        # Convert data entries from string to float
        param = [float(p) for p in entry[1:5]]
        peak_params.append(param)

        # If a line reference itself, otherwise reference specified line
        if entry[5] == 'l':
            doublet.append(i)
            if entry[6] == 'f':
                # Completely free
                vel_dep.append(i)
                prof_dep.append(i)
            elif entry[6][0] == 't':
                # Moving with it and same profile
                vel_dep.append(int(entry[6][1]))
                prof_dep.append(int(entry[6][1]))
            elif entry[6][0] == 'v':
                # Multiple species moving together but profiles different
                vel_dep.append(int(entry[6][1]))
                prof_dep.append(i)
                
        elif entry[5][0] == 'd':
            doublet.append(int(entry[5][1]))
            vel_dep.append(int(entry[5][1]))
            prof_dep.append(int(entry[5][1]))

# Give the lines the same velocity and sigma as the lines they are dependant on
for i in range(len(data)):
    vel_dep[i] = vel_dep[doublet[i]]
    prof_dep[i] = prof_dep[doublet[i]]
    
    peak_params[i][2] = peak_params[vel_dep[i]][2]
    peak_params[i][3] = peak_params[prof_dep[i]][3]

# Initialize parameters
sig_resolution = 0.5
sig_sampling = 4.0
bkg = 100 # Background level
peaks_no = len(peak_params)


# Create lambda values - go from lowest wavelength - (20 * largest sigma) to lowest wavelength + (20 * largest sigma)
lam_min = min(p[0] * (1 + p[2]/c) for p in peak_params)
lam_max = max(p[0] * (1 + p[2]/c) for p in peak_params)
sig_in = max(p[3] / c * p[0] * (1 + p[3]/c) for p in peak_params)
dx = sig_resolution / sig_sampling
nx = int(2 * (20*sig_in/dx) + 1)
x = np.linspace(-20 * sig_in + lam_min, 20 * sig_in + lam_max, nx)


# Initialize model
model = background(x, bkg)
gaussian_models = []
mod = Model(background, prefix='bkg_')
# Loop through for number of peaks
for i, (lam, relative_A, vel, sig) in enumerate(peak_params):
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
As_in = np.empty((peaks_no, Nsim))
sigs_in = np.empty((peaks_no, Nsim))
vels_in = np.empty((peaks_no, Nsim))
lams_in = [p[0] for p in peak_params]
# Initialize output arrays
As_out = np.empty((peaks_no, Nsim))
sigs_out = np.empty((peaks_no, Nsim))
vels_out = np.empty((peaks_no, Nsim))
lams_out = np.empty((peaks_no, Nsim))
AoNs_out = np.empty((peaks_no, Nsim))

# Simulation
for index, AoN in enumerate(AoNs):
    A = np.sqrt(bkg) * AoN
    for _ in range(Nsim_per_AoN):
        # Generate Gaussian + Noise data
        model = background(x, bkg)
        # Do separately for free and doublet lines
        amplitudes = []
        for (lam, relative_A, vel, sig), i in zip(peak_params, range(peaks_no)):
            if doublet[i] == i:
                amplitudes.append(relative_A)
                model += gaussian(x, A * relative_A, lam, peak_params[i][2], peak_params[i][3], sig_resolution)
                
                # Store input data
                As_in[i][index] = A * relative_A
                sigs_in[i][index] = sig
                vels_in[i][index] = vel
            else:
                amplitudes.append(np.nan)    
        
        for (lam, relative_A, vel, sig), i in zip(peak_params, range(peaks_no)):
            if np.isnan(amplitudes[i]):
                model += gaussian(x, A * relative_A * amplitudes[doublet[i]], lam, peak_params[i][2], peak_params[i][3], sig_resolution)
                
                # Store input data
                As_in[i][index] = A * relative_A * amplitudes[doublet[i]]
                sigs_in[i][index] = peak_params[doublet[i]][3]
                vels_in[i][index] = peak_params[doublet[i]][2]
            
            
        noise = np.random.randn(len(x)) * np.sqrt(model)
        y = model + noise
        
        # Fit with LMfit
        pfit = Parameters()
    
        # The background level
        pfit.add('bkg_bkg', value=bkg, vary=True)
        
        # Setting up parameters for the peaks (gi)
        for i in range(peaks_no):
            # These values are fixed either physically or by the instrument
            pfit.add(f'g{i+1}_lam_rf', value=peak_params[i][0], vary=False)
            pfit.add(name=f'g{i+1}_sig_resolution', value=sig_resolution, vary=False)
            
            if doublet[i] == i:
                pfit.add(f'g{i+1}_A', value=np.max(y) - np.median(y), min=0.0)
                if vel_dep[i] == i:
                    pfit.add(f'g{i+1}_vel', value=peak_params[i][2])
                if prof_dep[i] == i:
                    pfit.add(f'g{i+1}_sig', value=peak_params[i][3])

        # Loop again for amplitudes, velocities and sigmas of doublet peaks - have to do afterwards as could be dependant on a line appearing after it in the file
        for i in range(peaks_no):
            if vel_dep[i] != i:
                pfit.add(f'g{i+1}_vel', expr=f'g{vel_dep[i]+1}_vel')
            if prof_dep[i] != i:
                pfit.add(f'g{i+1}_sig', expr=f'g{prof_dep[i]+1}_sig')
            if doublet[i] != i:
                pfit.add(f'g{i+1}_A', expr=f'{peak_params[i][1]} * g{doublet[i]+1}_A')
            
            
        fit = mod.fit(y, pfit, x=x)
        
        # Save generated data
        for peak in range(peaks_no):
            As_out[peak][index] = np.abs(fit.params[f'g{peak+1}_A'].value)
            sigs_out[peak][index] = np.abs(fit.params[f'g{peak+1}_sig'].value)
            vels_out[peak][index] = np.abs(fit.params[f'g{peak+1}_vel'].value)
            lams_out[peak][index] = np.abs(fit.params[f'g{peak+1}_lam_rf'].value)
        
        # Plot the peaks
        if index % 1 == 0:
            labels = ['input model + noise', 'input model', 'fitted model']
            plt.plot(x, y, 'k-')
            plt.plot(x, model, 'c-')
            plt.plot(x, fit.best_fit, 'r-')
            plt.legend(labels)
            plt.grid()
            plt.show()


# Get measured A/N values
for peak in range(peaks_no):
    AoNs_out[peak] = As_out[peak]/np.sqrt(bkg)
    
AoNs_out_max = np.empty(Nsim)
AoNs_out_avg = np.empty(Nsim)
for index, AoNs in enumerate(zip(*AoNs_out)):
    AoNs_out_max[index] = max(AoNs)
    AoNs_out_avg[index] = np.mean(AoNs)


# Plots of results
# dAs = As_out - As_in
# dsigs = sigs_out - sigs_in
# dvels = vels_out - vels_in

# array = dsigs / sigs_in
# print(np.std(array[0][i]))
# label = '(sig_out - sig_in)/sig_in'
# for i in range(peaks_no):
#     plt.title(f'{label} against A/N for Nsim = {Nsim} for peak {i+1}')
#     plt.scatter(AoNs_out[i], array[i], s=0.5)
#     plt.axhline(y=-1, color='black', linestyle=':')
#     plt.xlabel('A/N')
#     plt.ylabel('%s' % label)
#     plt.xlim(-0.2, 11)
#     plt.ylim(-5, 5)
#     plt.show()

# array = dvels / vels_in
# label = '(vel_out - vel_in)/vel_in'
# print(np.std(array[0][i]))
# for i in range(peaks_no):
#     plt.title(f'{label} against A/N for Nsim = {Nsim} for peak {i+1}')
#     plt.scatter(AoNs_out[i], array[i], s=0.5)
#     plt.axhline(y=-1, color='black', linestyle=':')
#     plt.xlabel('A/N')
#     plt.ylabel('%s' % label)
#     plt.xlim(-0.2, 11)
#     plt.ylim(-5, 5)
#     plt.show()
    
# # Serialize the data in a file
# data = ['As_in, As_out, AoNs_out, lams_in, lams_out, sigs_in, sigs_out, vels_in, vels_out, dAs, dsigs, dvels, AoNs_out_avg, AoNs_out_max, peaks_no', As_in, As_out, AoNs_out, lams_in, lams_out, sigs_in, sigs_out, vels_in, vels_out, dAs, dsigs, dvels, AoNs_out_avg, AoNs_out_max, peaks_no]
# filename = 'peak_data_out'
# outfile = open(filename, 'wb')
# pickle.dump(data,outfile)
# outfile.close()