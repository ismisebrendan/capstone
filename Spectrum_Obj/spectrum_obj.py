import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model
from lmfit import Parameters
import pickle
from matplotlib.colors import TwoSlopeNorm
import sys
import os

# Point to the funcs file
func_path = os.path.abspath('../') + '/Funcs'
sys.path.insert(0, func_path)
from funcs import gaussian, background, flux


# The speed of light
c = 299792.458 #km/s

class Spectrum():
    """
    Generate synthetic spectra and fit these peaks according to the input files.

    Parameters
    ----------
    peaksdata : str
        The input filename with peak generating data.
    fitdata : str
        The input filename with peak fitting data.
    sig_resolution : float, default=0.5
        The resolution of the hypothetical spectroscopic system.
    sig_sampling : float, default=4.0
        The sampling resolution for fitting the Gaussians.
    bkg : float, default=100
        The background level.
    Nsim : int, default=1000
        The number of simulations to run.
    AoN_min : float, default=0
        The minimum amplitude/nose ratio.
    AoN_max : float, default=10
        The maximum amplitude/noise ratio.
    target_type : str, default=None
        The type of body that the spectra are from.
    
    """
    
    def __init__(self, peaksdata, fitdata, sig_resolution=0.5, sig_sampling=4.0, bkg=100, Nsim=1000, AoN_min=0, AoN_max=10, target_type=None):
        self.peaksdata = peaksdata
        self.fitdata = fitdata
        self.sig_resolution = sig_resolution
        self.sig_sampling = sig_sampling
        self.bkg = bkg
        self.Nsim = Nsim
        self.AoN_min = AoN_min
        self.AoN_max = AoN_max
        self.target_type = target_type
        
        self.peak_params = []
        self.fit_params = []
        self.doublet = []
        self.vel_dep = []
        self.prof_dep = []
        self.peaks_no = 0
        
        # For output
        self.data = np.array([])
        self.data_info = '0  As_in\n1  As_out\n2  As_unc_out\n3  AoNs\n4  AoNs_out\n5  AoNs_unc_out\n6  lams_in\n7  lams_out\n8  lams_unc_out\n9  sigs_in\n10 sigs_out\n11 sigs_unc_out\n12 vels_in\n13 vels_out\n14 vels_unc_out\n15 peak_params\n16 peaks_no\n17 Nsim\n18 This information'
        
    def print_info(self):
        """
        Print the information about the object.
        
        """
        
        print('Spectrum object, generates and fits synthetic spectra from files.')
        print(f'Type of object: {self.target_type}')
        print('----------------')
        print('Input parameters')
        print(f'\t- sig_resolution: {self.sig_resolution}')
        print(f'\t- sig_sampling: {self.sig_sampling}')
        print(f'\t- background level: {self.bkg}')
        print(f'\t- NSim: {self.Nsim}')
        print(f'\t- AoN range: [{self.AoN_min}, {self.AoN_max}')
        print(f'\t- Number of peaks: {self.peaks_no}')

    def get_data(self):
        """
        Take the input files and extract the data from them.
        
        """
        
        # Plotting file
        with open(self.peaksdata) as f:
            data_in = f.readlines()
            
            for i in range(1, len(data_in)):
                entry_in = data_in[i].split()
                
                # Convert data entries from string to float
                param = [float(p) for p in entry_in[2:7]]
                self.peak_params.append(param)

                # Note if a line references itself or a specified line
                if entry_in[7] == 'l':
                    self.doublet.append(i-1)
                elif entry_in[7][0] == 'd':
                    self.doublet.append(int(entry_in[7][1:]))
        
        # How many peaks
        self.peaks_no = len(self.peak_params)

        # Fitting data
        with open(self.fitdata) as f:
            data_fit = f.readlines()
            
            for i in range(1, len(data_fit)):
                entry_fit = data_fit[i].split()
                
                # Ensure that the fitting and plotting lines appear in the same order
                for j in range(1, len(data_in)):
                    entry_in = data_in[j].split()
                    
                    if entry_fit[1] == entry_in[1]:
                        # Convert data entries from string to float
                        param = [float(p) for p in entry_fit[2:7]]
                        self.fit_params.append(param)
        
                        # Note if a line references itself or a specified line
                        if entry_fit[7] == 'l':
                            if entry_fit[8] == 'f':
                                # Completely free
                                self.vel_dep.append(j-1)
                                self.prof_dep.append(j-1)
                                
                            elif entry_fit[8][0] == 't':
                                # Moving with it and same profile
                                self.vel_dep.append(int(entry_fit[8][1:]))
                                self.prof_dep.append(int(entry_fit[8][1:]))
                                
                            elif entry_fit[8][0] == 'v':
                                # Multiple species moving together but profiles different
                                self.vel_dep.append(int(entry_fit[8][1:]))
                                self.prof_dep.append(i)
                                
                        elif entry_fit[7][0] == 'd':
                            # If dependent then moving with it and has same profile
                            self.vel_dep.append(int(entry_fit[7][1:]))
                            self.prof_dep.append(int(entry_fit[7][1:]))
        
        # Give the lines the same velocity and sigma as the lines they are dependent on
        for i in range(self.peaks_no - 1):
            self.peak_params[i][3] = self.peak_params[self.doublet[i]][3]
            self.peak_params[i][4] = self.peak_params[self.doublet[i]][4]
            self.vel_dep[i] = self.vel_dep[self.doublet[i]]
            self.prof_dep[i] = self.prof_dep[self.doublet[i]]
            
            self.fit_params[i][3] = self.fit_params[self.vel_dep[i]][3]
            self.fit_params[i][4] = self.fit_params[self.prof_dep[i]][4]

    def create_bkg(self):
        """
        Create the array for the background level to the spectrum.
        
        """
        
        # Create lambda values - go from shortest wavelength - (20 * largest sigma) to longest wavelength + (20 * largest sigma)
        lam_min = min(p[0] * (1 + p[3]/c) for p in self.peak_params)
        lam_max = max(p[0] * (1 + p[3]/c) for p in self.peak_params)
        sig_in = max(p[4] / c * p[0] * (1 + p[4]/c) for p in self.peak_params)
        dx = self.sig_resolution / self.sig_sampling
        nx = int(2 * (20*sig_in/dx) + 1)
        self.x = np.linspace(-20 * sig_in + lam_min, 20 * sig_in + lam_max, nx)

    def init_model(self):
        """
        Generates the background level of the model.
   
        """
        
        # Prerequesite functions
        self.get_data()
        self.create_bkg()
        self.gen_arrs()

        self.model = background(self.x, self.bkg)
        gaussian_models = []
        self.mod = Model(background, prefix='bkg_')

        # Loop through for the number of peaks
        for i, (lam, relative_A_l, relative_A_u, vel, sig) in enumerate(self.peak_params):
            gauss = Model(gaussian, prefix=f'g{i}_')
            self.mod += gauss
            gaussian_models.append(gauss)
    
    def gen_arrs(self):
        """
        Generate the arrays used to store input and output data.
   
        """
        
        # Initialize input arrays
        self.As_in = np.empty((self.peaks_no, self.Nsim))
        self.sigs_in = np.empty((self.peaks_no, self.Nsim))
        self.vels_in = np.empty((self.peaks_no, self.Nsim))
        self.lams_in = np.array([p[0] for p in self.peak_params])

        # Initialize output arrays
        self.As_out = np.empty((self.peaks_no, self.Nsim))
        self.As_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.sigs_out = np.empty((self.peaks_no, self.Nsim))
        self.sigs_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.vels_out = np.empty((self.peaks_no, self.Nsim))
        self.vels_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.lams_out = np.empty((self.peaks_no, self.Nsim))
        self.lams_unc_out = np.empty((self.peaks_no, self.Nsim))
        self.AoNs_out = np.empty((self.peaks_no, self.Nsim))
        self.AoNs_unc_out = np.empty((self.peaks_no, self.Nsim))
    
    def generate(self):
        """
        Generate the synthetic spectrum.

        Returns
        -------
        y_vals : array
            The generated spectra
    
        """
        
        # Initialise
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        
        y_vals = np.empty((self.Nsim, len(self.x)))

        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN

            # Generate Gaussian + Noise data 
            self.model = background(self.x, self.bkg)
            
            # Generate free and doublet lines separately
            amplitudes = []
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if self.doublet[i] == i:
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    amplitudes.append(relative_A)
                    self.model += gaussian(self.x, A * relative_A, lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A
                    self.sigs_in[i][index] = sig
                    self.vels_in[i][index] = vel
                else:
                    amplitudes.append(np.nan)    
            
            # Repeat to generate the doublet lines
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if np.isnan(amplitudes[i]):
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    self.model += gaussian(self.x, A * relative_A * amplitudes[self.doublet[i]], lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A * amplitudes[self.doublet[i]]
                    self.sigs_in[i][index] = sig
                    self.vels_in[i][index] = vel
                
            # Generate noise and add it to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            y_vals[index] = y
        
        return y_vals
    
    def simulation(self, plotting=False):
        """
        Generate and fit the synthetic spectrum.
        
        Parameters
        ----------
        plotting : bool, default=False
            Whether or not to plot the graphs.
     
        """
     
        # Initialise
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN
            
            # Generate Gaussian + Noise data 
            self.model = background(self.x, self.bkg)
            
            # Generate free and doublet lines separately
            amplitudes = []
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if self.doublet[i] == i:
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    amplitudes.append(relative_A)
                    self.model += gaussian(self.x, A * relative_A, lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A
                    self.sigs_in[i][index] = sig
                    self.vels_in[i][index] = vel
                else:
                    amplitudes.append(np.nan)    
            
            # Repeat to generate the doublet lines
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if np.isnan(amplitudes[i]):
                    # If given a range of values choose a random value
                    relative_A = np.random.uniform(relative_A_l, relative_A_u)
                    self.model += gaussian(self.x, A * relative_A * amplitudes[self.doublet[i]], lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A * amplitudes[self.doublet[i]]
                    self.sigs_in[i][index] = sig
                    self.vels_in[i][index] = vel
                
            # Generate noise and add it to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            
            # Fit with LMfit 
            pfit = Parameters()
        
            # The background level
            pfit.add('bkg_bkg', value=self.bkg, vary=True)
            
            # Setting up parameters for the peaks (g_i)
            for i in range(self.peaks_no):
                # These values are fixed either physically or by the instrument
                pfit.add(f'g{i}_lam_rf', value=self.fit_params[i][0], vary=False)
                pfit.add(name=f'g{i}_sig_resolution', value=self.sig_resolution, vary=False)
                
                if self.doublet[i] == i:
                    # For free lines take initial guess as largest y value in the region +- 100 Angstrom from where it should be based on initial guesses
                    expec_lam = self.fit_params[i][0] *  (1 + self.fit_params[i][3]/c)
                    ind = np.where(np.abs(expec_lam - self.x) <= 100)
                    pfit.add(f'g{i}_A', value=np.max(y[ind]) - np.median(y), min=self.fit_params[i][1], max=self.fit_params[i][2])
                    
                    # If independent in terms of velocity and sigma take those as its initial estimates
                    if self.vel_dep[i] == i:
                        pfit.add(f'g{i}_vel', value=self.fit_params[i][3])
                    if self.prof_dep[i] == i:
                        pfit.add(f'g{i}_sig', value=self.fit_params[i][4])

            # Loop again for amplitudes, velocities and sigmas of dependent peaks - have to do afterwards as could be dependent on a line appearing after it in the file
            for i in range(self.peaks_no):
                if self.vel_dep[i] != i:
                    # Velocity must be that of another peak
                    pfit.add(f'g{i}_vel', expr=f'g{self.vel_dep[i]}_vel')
                if self.prof_dep[i] != i:
                    # Sigma must be that of another peak
                    pfit.add(f'g{i}_sig', expr=f'g{self.prof_dep[i]}_sig')
                if self.doublet[i] != i:
                    # The amplitude of the peak is equal to the that of the reference line times some value
                    if self.fit_params[i][1] != self.fit_params[i][2]:
                        pfit.add(f'g{i}_delta', min=self.fit_params[i][1], max=self.fit_params[i][2])
                    else:
                        pfit.add(f'g{i}_delta', expr=f'{self.fit_params[i][2]}')
                    pfit.add(f'g{i}_A', expr=f'g{i}_delta * g{self.doublet[i]}_A')
                
                
            fit = self.mod.fit(y, pfit, x=self.x)
            
            # Save generated data
            for peak in range(self.peaks_no):
                self.As_out[peak][index] = fit.params[f'g{peak}_A'].value
                self.As_unc_out[peak][index] = fit.params[f'g{peak}_A'].stderr
                self.sigs_out[peak][index] = fit.params[f'g{peak}_sig'].value
                self.sigs_unc_out[peak][index] = fit.params[f'g{peak}_sig'].stderr
                self.vels_out[peak][index] = fit.params[f'g{peak}_vel'].value
                self.vels_unc_out[peak][index] = fit.params[f'g{peak}_vel'].stderr
                self.lams_out[peak][index] = fit.params[f'g{peak}_lam_rf'].value
                self.lams_unc_out[peak][index] = fit.params[f'g{peak}_lam_rf'].stderr
                self.f_out, self.f_unc_out = flux(self.As_out, self.sigs_out, self.As_unc_out, self.sigs_unc_out)
                self.f_in, self.f_unc_in = flux(self.As_in, self.sigs_in)

            # Plotting
            if plotting == True:
                self.plot_spectrum(y, fit)
        
        # Save AoNs
        for peak in range(self.peaks_no):
            self.AoNs_out[peak] = self.As_out[peak]/np.sqrt(self.bkg)
            self.AoNs_unc_out[peak] = self.As_unc_out[peak]/np.sqrt(self.bkg)

    def simulation_false(self, plotting=False):
        """
        Generate and fit the synthetic spectrum, randomly choose some lines to remove.
        
        Parameters
        ----------
        plotting : bool, default=False
            Whether or not to plot the graphs.
        """
        
        # Initial variables
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        
        # Keep some lines, remove others
        self.keep_lines = np.random.choice([0,1], (self.peaks_no, self.Nsim))
    
        # Generate and fit spectra
        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN
            
            # Generate Gaussian + Noise data
            self.model = background(self.x, self.bkg)
            
            # Generate free and doublet lines separately
            amplitudes = []
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if self.doublet[i] == i:
                    # If given a range of values choose a random value
                    # Randomly keep or remove a line
                    relative_A = np.random.uniform(relative_A_l, relative_A_u) * self.keep_lines[i][index]
                    amplitudes.append(relative_A)
                    self.model += gaussian(self.x, A * relative_A, lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A
                    self.sigs_in[i][index] = sig
                    self.vels_in[i][index] = vel
                else:
                    amplitudes.append(np.nan)    
            
            # Repeat to generate the doublet lines
            for (lam, relative_A_l, relative_A_u, vel, sig), i in zip(self.peak_params, range(self.peaks_no)):
                if np.isnan(amplitudes[i]):
                    
                    # If a main line is removed the doublet should also be removed
                    if self.keep_lines[self.doublet[i]][index] == 0:
                        self.keep_lines[i][index] = 0    
                    
                    # If given a range of values choose a random value
                    # Randomly keep or remove a line
                    relative_A = np.random.uniform(relative_A_l, relative_A_u) * self.keep_lines[i][index]
                    self.model += gaussian(self.x, A * relative_A * amplitudes[self.doublet[i]], lam, vel, sig, self.sig_resolution)
                    
                    # Store input data
                    self.As_in[i][index] = A * relative_A * amplitudes[self.doublet[i]]
                    self.sigs_in[i][index] = sig
                    self.vels_in[i][index] = vel
                
            # Generate noise and add to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            
            # Fit with LMfit 
            pfit = Parameters()
        
            # The background level
            pfit.add('bkg_bkg', value=self.bkg, vary=True)
            
            # Setting up parameters for the peaks (g_i)
            for i in range(self.peaks_no):
                # These values are fixed either physically or by the instrument
                pfit.add(f'g{i}_lam_rf', value=self.fit_params[i][0], vary=False)
                pfit.add(name=f'g{i}_sig_resolution', value=self.sig_resolution, vary=False)
                
                if self.doublet[i] == i:
                    # For free lines take initial guess as largest y value in the region +- 100 Angstrom from where it should be based on initial guesses
                    expec_lam = self.fit_params[i][0] *  (1 + self.fit_params[i][3]/c)
                    ind = np.where(np.abs(expec_lam - self.x) <= 100)
                    pfit.add(f'g{i}_A', value=np.max(y[ind]) - np.median(y), min=self.fit_params[i][1], max=self.fit_params[i][2])
                    
                    # If independent in terms of velocity and sigma take those as its initial estimates
                    if self.vel_dep[i] == i:
                        pfit.add(f'g{i}_vel', value=self.fit_params[i][3])
                    if self.prof_dep[i] == i:
                        pfit.add(f'g{i}_sig', value=self.fit_params[i][4])

            # Loop again for amplitudes, velocities and sigmas of dependent peaks - have to do afterwards as could be dependent on a line appearing after it in the file
            for i in range(self.peaks_no):
                if self.vel_dep[i] != i:
                    # Velocity must be that of another peak
                    pfit.add(f'g{i}_vel', expr=f'g{self.vel_dep[i]}_vel')
                if self.prof_dep[i] != i:
                    # Sigma must be that of another peak
                    pfit.add(f'g{i}_sig', expr=f'g{self.prof_dep[i]}_sig')
                if self.doublet[i] != i:
                    # The amplitude of the peak is equal to the that of the reference line times some value
                    if self.fit_params[i][1] != self.fit_params[i][2]:
                        pfit.add(f'g{i}_delta', min=self.fit_params[i][1], max=self.fit_params[i][2])
                    else:
                        pfit.add(f'g{i}_delta', expr=f'{self.fit_params[i][2]}')
                    pfit.add(f'g{i}_A', expr=f'g{i}_delta * g{self.doublet[i]}_A')
                
                
            fit = self.mod.fit(y, pfit, x=self.x)
            
            # Save generated data
            for peak in range(self.peaks_no):
                self.As_out[peak][index] = fit.params[f'g{peak}_A'].value
                self.As_unc_out[peak][index] = fit.params[f'g{peak}_A'].stderr
                self.sigs_out[peak][index] = fit.params[f'g{peak}_sig'].value
                self.sigs_unc_out[peak][index] = fit.params[f'g{peak}_sig'].stderr
                self.vels_out[peak][index] = fit.params[f'g{peak}_vel'].value
                self.vels_unc_out[peak][index] = fit.params[f'g{peak}_vel'].stderr
                self.lams_out[peak][index] = fit.params[f'g{peak}_lam_rf'].value
                self.lams_unc_out[peak][index] = fit.params[f'g{peak}_lam_rf'].stderr
                self.f_out, self.f_unc_out = flux(self.As_out, self.sigs_out, self.As_unc_out, self.sigs_unc_out)
                self.f_in, self.f_unc_in = flux(self.As_in, self.sigs_in)

            # Plotting
            if plotting == True:
                self.plot_spectrum(y, fit)
        
        # Save AoNs
        for peak in range(self.peaks_no):
            self.AoNs_out[peak] = self.As_out[peak]/np.sqrt(self.bkg)
            self.AoNs_unc_out[peak] = self.As_unc_out[peak]/np.sqrt(self.bkg)

    def plot_spectrum(self, y, fit):
        """
        Plot a spectrum.

        Parameters
        ----------
        y : array
            The amplitude data.
        fit : lmfit fit
            The fit data for the model.

        """
        
        labels = ['input model + noise', 'input model', 'fitted model']
        plt.plot(self.x, y, 'k-')
        plt.plot(self.x, self.model, 'c-')
        plt.plot(self.x, fit.best_fit, 'r-')
        plt.xlabel(r'$\lambda$ ($\AA$)')
        plt.ylabel('Amplitude (arbitrary units)')
        if self.target_type != None:
            plt.title(f'Generated and fit spectrum with emission lines of {self.target_type}')
        else:
            plt.title('Generated and fit spectrum with emission lines')
        plt.legend(labels)
        plt.grid()
        plt.show()

    def dump(self):
        """
        Dump the inputted and fitted parameters to a variable

        Returns
        -------
        data : list
            The input and output values.
       
        """
        
        data = [self.As_in, self.As_out, self.As_unc_out, self.AoNs, self.AoNs_out, self.AoNs_unc_out, self.lams_in, self.lams_out, self.lams_unc_out, self.sigs_in, self.sigs_out, self.sigs_unc_out, self.vels_in, self.vels_out, self.vels_unc_out, self.peak_params, self.peaks_no, self.Nsim]
        return data

    def output(self, outfile='peak_data_out.pickle', overwrite=True):
        """
        Dump out the input and fitted parameters using pickle, can append to data files containing the same number of peaks (and ideally the same actual peaks of course).

        Parameters
        ----------
        outfile : str, default='peak_data_out.pickle'
            The name of the file to save the data to.
        overwrite : bool, default=True
            Overwrite the file or append to it.
       
        """
        
        data = self.dump()
        
        if overwrite == False:
            try:
                # Try to open the data file if it exists and append the data from this run to it
                with open(outfile, 'rb') as pickle_file:
                    in_data = pickle.load(pickle_file)
                    
                # Concatenate the data
                for i in range(15):
                    if i != 6:
                        data[i] = np.concatenate((in_data[i].T, data[i].T)).T
            
                # Increment Nsim
                data[17] = data[17] + in_data[17]
                
                # Add the info
                data.append(self.data_info)
                
                # Save the data
                with open(outfile, 'wb') as outfile:
                    pickle.dump(data, outfile)
                outfile.close()
                
            except:
                # Create the file
                outfile = open(outfile, 'wb')
                
                data.append(self.data_info)
                
                # Write the data
                pickle.dump(data, outfile)
                outfile.close()
        else:
            # Create the file
            outfile = open(outfile, 'wb')
            
            data.append(self.data_info)
        
            # Write the data
            pickle.dump(data, outfile)
            outfile.close()
    
    def read_pickle(self, filename):
        """
        Read data from a pickle file.

        Parameters
        ----------
        filename : str
            The filename to read.
    
        """
      
        with open(filename, 'rb') as pickle_file:
            self.pickle_in = pickle.load(pickle_file)
    
    def overwrite_all(self, data_in):
        """
        Overwrite all variables with data from a variable. Designed to correspond to the format of that data is output from this object.
        
        See Also
        --------
        overwrite : Overwrite a particular parameter with a new value.
     
        """
        
        # Overwrite variables
        self.As_in = data_in[0]
        self.As_out = data_in[1]
        self.As_unc_out = data_in[2]
        self.AoNs = data_in[3]
        self.AoNs_out = data_in[4]
        self.AoNs_unc_out = data_in[5]
        self.lams_in = data_in[6]
        self.lams_out = data_in[7]
        self.lams_unc_out = data_in[8]
        self.sigs_in = data_in[9]
        self.sigs_out = data_in[10]
        self.sigs_unc_out = data_in[11]
        self.vels_in = data_in[12]
        self.vels_out = data_in[13]
        self.vels_unc_out = data_in[14]
        self.peak_params = data_in[15]
        self.peaks_no = data_in[16]
        self.Nsim = data_in[17]
        
        # Calculate fluxes
        self.f_out, self.f_unc_out = flux(self.As_out, self.sigs_out, self.As_unc_out, self.sigs_unc_out)
        self.f_in, self.f_unc_in = flux(self.As_in, self.sigs_in)

    def overwrite(self, parameter, value):
        """
        Overwrite a particular parameter with a new value.

        Parameters
        ----------
        parameter : int
            The parameter to overwrite given as the index of the parameter in self.peak_params.
        value
            The (list of) values to overwrite with.
        
        Indices
        -------
        wl,	A_in_l,	A_in_u,	v_in,	sig_in,	free
        0,  1,      2,      3,     4,      5
        
        See Also
        --------
        overwrite_all : Overwrite all variables with data from a variable. Designed to correspond to the format of that data is output from this object.
        
        """

        for i in range(self.peaks_no):
            if type(value) == list:
                self.peak_params[i][parameter] = value[i]
                self.fit_params[i][parameter] = value[i]
            else:
                self.peak_params[i][parameter] = value
                self.fit_params[i][parameter] = value
                
    def plot_results(self, line=0, param='sig', xlim=[-0.2, 11], ylim=[-5, 5]):
        """
        Plot the difference between the input and output values of different components.
        
        Parameters
        ----------
        line : int, default=0
            Which line of the spectrum to plot for.
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to plot for. 
        xlim : array or None, default=[-0.2,11]
            The xlimits of the plot
        ylim : array or None, default=[-5,5]
            The ylimits of the plot
        
        See Also
        --------
        plot_results_err : Plot the difference between the input and output values of different components showing errorbars.
     
        """
        
        if param == 'sig':
            array = (self.sigs_out - self.sigs_in) / self.sigs_in
        elif param == 'vel':
            array = (self.vels_out - self.vels_in) / self.vels_in
        elif param == 'A':
            array = (self.As_out - self.As_in) / self.As_in
        elif param == 'flux':
            array = (self.f_out - self.f_in) / self.f_in
        
        close_0 = self.count_not_fit()
        
        label = f'({param}_out - {param}_in)/{param}_in'
        plt.title(rf'{label} against A/N of peak {line} for Nsim = {self.Nsim}'+f'\nv_in = {self.peak_params[0][3]}, sig_in = {self.peak_params[0][4]}')
        plt.scatter(self.AoNs_out[line], array[line], s=0.5, label='Data')
        plt.scatter(self.AoNs_out[line][close_0], array[line][close_0], s=0.5, label=r'$\sigma \leq 10$ ')
        plt.xlabel('A/N')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.show()

    def plot_results_err(self, line=0, param='sig', xlim=[-0.2, 11], ylim=[-5, 5]):
        """
        Plot the difference between the input and output values of different components.
        
        Parameters
        ----------
        line : int, default=0
            Which line of the spectrum to plot for.
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to plot for. 
        xlim : array or None, default=[-0.2,11]
            The xlimits of the plot
        ylim : array or None, default=[-5,5]
            The ylimits of the plot
                
        See Also
        --------
        plot_results : Plot the difference between the input and output values of different components without showing errorbars.
   
        """
        
        if param == 'sig':
            array = (self.sigs_out - self.sigs_in) / self.sigs_in
            unc = self.sigs_unc_out / self.sigs_in
        elif param == 'vel':
            array = (self.vels_out - self.vels_in) / self.vels_in
            unc = self.vels_unc_out / self.vels_in
        elif param == 'A':
            array = (self.As_out - self.As_in) / self.As_in
            unc = self.As_unc_out / self.As_in
        elif param == 'flux':
            array = (self.f_out - self.f_in) / self.f_in
            unc = self.f_unc_out / self.f_in
                
        close_0 = self.count_not_fit()

        label = f'({param}_out - {param}_in)/{param}_in'
        plt.title(rf'{label} against A/N of peak {line} for Nsim = {self.Nsim}'+f'\nv_in = {self.peak_params[0][3]}, sig_in = {self.peak_params[0][4]}')
        plt.scatter(self.AoNs_out[line], array[line], s=0.5)
        plt.errorbar(self.AoNs_out[line], array[line], unc[line], fmt='none', label='Data')
        plt.scatter(self.AoNs_out[line][close_0], array[line][close_0], s=0.5)
        plt.errorbar(self.AoNs_out[line][close_0], array[line][close_0], unc[line][close_0], fmt='none', label=r'$\sigma \leq 10$', color='#ff7f0e')
        plt.xlabel('A/N')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.show()
                
    def recover_data(self, peak=0, param='sig'):
        """
        Find the difference between the input and output values of different components.
        
        Parameters
        ----------
        peak : int, default=0
            Which peak of the spectrum to check for.
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to check for.
        
        Returns
        -------
        arr : array
            The values of difference between the input and output values of the selected component.
        std : float
            The standard deviation of arr.
        median : float
            The median of arr.
   
        """
        
        if param == 'sig':
            array = (self.sigs_out - self.sigs_in) / self.sigs_in
        elif param == 'vel':
            array = (self.vels_out - self.vels_in) / self.vels_in
        elif param == 'A':
            array = (self.As_out - self.As_in) / self.As_in
        elif param == 'flux':
            array = (self.f_out - self.f_in) / self.f_in

        arr = array[peak]
        std = np.std(arr)
        med = np.median(arr)
        
        return arr, std, med

    def heatmap_sum(self, param, line, brightest=4, text=True):
        """
        Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.

        Parameterss
        ----------
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to check for.
        line : int
            The line of interest.
        brightest : int, default=4
            The brightest line (default corrseponds to H alpha in the normal input structure).
        text : bool
            Whether or not to show the value as text in the plot.
        
        See Also
        --------
        heatmap_brightest : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line.
   
        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of the brightest line (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. remove any index that doesn't appear 8 times (8 lines)
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)
        
        interest_AoN = self.AoNs_out[line][ind]
        sum_AoN = np.sum(self.AoNs_out[:,ind], axis=0)[0]
        
        interest_val = self.recover_data(peak=line, param=param)[0][ind]
        
        x_vals = np.linspace(int(min(sum_AoN)), int(max(sum_AoN)) + 1, int(max(sum_AoN)) + 2, dtype=int)
        y_vals = np.linspace(int(min(interest_AoN)), int(max(interest_AoN)) + 1, int(max(interest_AoN)) + 2, dtype=int)

        stds = []
        medians = []

        for i in x_vals[1:]:
            ind_x = (sum_AoN < i) * (sum_AoN > i - 1)

            for j in y_vals[1:]:
                ind_y = (interest_AoN < j) * (interest_AoN > j-1)
                
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))

        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        # Plot the standard deviations
        pc = plt.pcolormesh(x_vals, y_vals, stds.T, cmap='inferno')
        plt.colorbar(pc)
        plt.title(f'Standard deviation of {label}')
        plt.xlabel('Sum of A/N of all lines')
        plt.ylabel(f'A/N of line {line}')
        # Text
        if text == True:
            for i in range(len(x_vals)-1):
                for j in range(len(y_vals)-1):
                    if np.isnan(stds.T[j][i]):
                        pass
                    else:
                        plt.text(x_vals[i]+0.5, y_vals[j]+0.5, np.round(stds.T[j][i], 2), ha='center', va='center', color='w', fontsize='x-small')
      
        plt.show()

         # Plot the medians
        norm = TwoSlopeNorm(vcenter=0)
        pc = plt.pcolormesh(x_vals, y_vals, medians.T, norm=norm, cmap='seismic')
        plt.colorbar(pc)
        plt.title(f'Median of {label}')
        plt.xlabel('Sum of A/N of all lines')
        plt.ylabel(f'A/N of line {line}')
        # Text
        if text == True:
            for i in range(len(x_vals)-1):
                for j in range(len(y_vals)-1):
                    if np.isnan(medians.T[j][i]):
                        pass
                    else:
                        plt.text(x_vals[i]+0.5, y_vals[j]+0.5, np.round(medians.T[j][i], 3), ha='center', va='center', color='k', fontsize='x-small')
                
        plt.show()

    def heatmap_brightest(self, param, line, brightest=4, text=True):
        """
        Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line.

        Parameters
        ----------
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to check for.
        line : int
            The line of interest.
        brightest : int, default=4
            The brightest line (default corresponds to H alpha in the normal input structure).
        text : bool
            Whether or not to show the value as text in the plot.
            
        See Also
        --------
        heatmap_sum : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.
     
        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. remove any index that doesn't appear 8 times (8 lines)
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)
        
        interest_AoN = self.AoNs_out[line][ind]
        brightest_AoN = self.AoNs_out[brightest][ind]
        
        interest_val = self.recover_data(peak=line, param=param)[0][ind]
        
        
        x_vals = np.linspace(int(min(brightest_AoN)), int(max(brightest_AoN)) + 1, int(max(brightest_AoN)) + 2, dtype=int)
        y_vals = np.linspace(int(min(interest_AoN)), int(max(interest_AoN)) + 1, int(max(interest_AoN)) + 2, dtype=int)

        stds = []
        medians = []

        for i in x_vals[1:]:
            ind_x = (brightest_AoN < i) * (brightest_AoN > i - 1)

            for j in y_vals[1:]:
                ind_y = (interest_AoN < j) * (interest_AoN > j-1)
                
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))

        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        # Plot the standard deviations
        pc = plt.pcolormesh(x_vals, y_vals, stds.T, cmap='inferno')
        plt.colorbar(pc)
        plt.title(f'Standard deviation of {label}')
        plt.xlabel(f'A/N of line {brightest}')
        plt.ylabel(f'A/N of line {line}')
        # Text
        if text == True:
            for i in range(len(x_vals)-1):
                for j in range(len(y_vals)-1):
                    if np.isnan(stds.T[j][i]):
                        pass
                    else:
                        plt.text(x_vals[i]+0.5, y_vals[j]+0.5, np.round(stds.T[j][i], 3), ha='center', va='center', color='w', fontsize='x-small')
        
        plt.show()

         # Plot the medians
        norm = TwoSlopeNorm(vcenter=0)
        pc = plt.pcolormesh(x_vals, y_vals, medians.T, norm=norm, cmap='seismic')
        plt.colorbar(pc)
        plt.title(f'Median of {label}')
        plt.xlabel(f'A/N of line {brightest}')
        plt.ylabel(f'A/N of line {line}')
        # Text
        if text == True:
            for i in range(len(x_vals)-1):
                for j in range(len(y_vals)-1):
                    if np.isnan(medians.T[j][i]):
                        pass
                    else:
                        plt.text(x_vals[i]+0.5, y_vals[j]+0.5, np.round(medians.T[j][i], 2), ha='center', va='center', color='k', fontsize='x-small')

        plt.show()
        
    def count_not_fit(self):
        """
        Count the number of lines in the data that no fit was found for them based on having a very low standard deviation.
        
        Returns
        -------
        close_0 : array
            The indices where sigs_out is close to 0
        """
        
        # Find where sig_out is close to 0
        close_0 = np.unique(np.where(np.abs(self.sigs_out) <= 10)[1])
        
        return close_0
        
        
        
        