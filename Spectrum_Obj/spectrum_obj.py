import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model
from lmfit import Parameters
import pickle
from matplotlib.colors import TwoSlopeNorm
import sys
import os

# For interactive plots, comment out if not using something that allows interactive plots.
%matplotlib qt

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
        self.data_info = '0  As_in\n1  As_out\n2  As_unc_out\n3  AoNs\n4  AoNs_out\n5  AoNs_unc_out\n6  lams_in\n7  lams_out\n8  lams_unc_out\n9  sigs_in\n10 sigs_out\n11 sigs_unc_out\n12 vels_in\n13 vels_out\n14 vels_unc_out\n15 peak_params\n16 peaks_no\n17 Nsim\n18 doublet\n19 This information'
        
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
        
        self.peak_params = []
        self.fit_params = []
        self.doublet = []
        self.vel_dep = []
        self.prof_dep = []
        self.peaks_no = 0
        
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
        
        self.get_line_ratios
    
    def get_line_ratios(self):
        """
        Get the line ratios for all the peaks

        """
        
        # Get the line ratios
        self.line_ratios = np.array(self.peak_params)[:,1]
        for i in range(self.peaks_no):
            if self.doublet[i] != i:
                self.line_ratios[i] = self.line_ratios[i] * self.line_ratios[self.doublet[i]]

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
        
        # Store data and fit
        self.spectra_mat = np.empty((self.Nsim, len(self.x)))
        self.model_mat = np.empty((self.Nsim, len(self.x)))
        self.fit_mat = np.empty((self.Nsim, len(self.x)))
    
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
            
            self.spectra_mat[index] = y
            self.model_mat[index] = self.model
            self.fit_mat[index] = fit.best_fit
                

            # Plotting
            if plotting == True:
                self.plot_spectrum(y, fit.best_fit, model)
        
        self.f_out, self.f_unc_out = flux(self.As_out, self.sigs_out, self.As_unc_out, self.sigs_unc_out)
        self.f_in, self.f_unc_in = flux(self.As_in, self.sigs_in)
        
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
                self.plot_spectrum(y, fit.best_fit, self.model)
        
        # Save AoNs
        for peak in range(self.peaks_no):
            self.AoNs_out[peak] = self.As_out[peak]/np.sqrt(self.bkg)
            self.AoNs_unc_out[peak] = self.As_unc_out[peak]/np.sqrt(self.bkg)

    def plot_spectrum(self, y, fit, model):
        """
        Plot a spectrum.

        Parameters
        ----------
        y : array
            The amplitude data.
        fit : array
            The fit data for the model.
        model : array
            The model data.

        See Also
        --------
        plot_spectrum_return : Plot a spectrum and return to param v A/N graph on right click.
        plot_spectrum_centre : Plot spectra centred on certain wavelengths.
      
        """
        
        labels = ['input model + noise', 'input model', 'fitted model']
        plt.plot(self.x, y, 'k-')
        plt.plot(self.x, model, 'c-')
        plt.plot(self.x, fit, 'r-')
        plt.xlabel(r'$\lambda$ ($\AA$)')
        plt.ylabel('Amplitude (arbitrary units)')
        if self.target_type != None:
            plt.title(f'Generated and fit spectrum with emission lines of {self.target_type}')
        else:
            plt.title('Generated and fit spectrum with emission lines')
        plt.legend(labels)
        plt.grid()
        plt.show()
    
    def plot_spectrum_return(self, y, fit, model):
        """
        Plot a spectrum and return to param v A/N graph on right click.

        Parameters
        ----------
        y : array
            The amplitude data.
        fit : array
            The fit data for the model.
        model : array
            The model data.

        See Also
        --------
        plot_spectrum : Plot a spectrum.
        plot_spectrum_centre : Plot spectra centred on certain wavelengths.
        
        """
        
        global current_plot
        current_plot = 'spectrum'
        
        fig, ax = plt.subplots()

        labels = ['input model + noise', 'input model', 'fitted model']
        plt.plot(self.x, y, 'k-')
        plt.plot(self.x, model, 'c-')
        plt.plot(self.x, fit, 'r-')
        plt.xlabel(r'$\lambda$ ($\AA$)')
        plt.ylabel('Amplitude (arbitrary units)')
        if self.target_type != None:
            plt.title(f'Generated and fit spectrum with emission lines of {self.target_type}')
        else:
            plt.title('Generated and fit spectrum with emission lines')
        plt.legend(labels)
        plt.grid()
        plt.show()
        
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

    def plot_spectrum_centre(self, y, fit, model, centre, ran):
        """
        Plot spectra centred on certain wavelengths.

        Parameters
        ----------
        y : array
            The amplitude data.
        fit : array
            The fit data for the model.
        model : array
            The model data.
        centre : array
            The wavelengths to centre the plots on.
        ran : float
            The range to plot from, centre - ran to centre + ran
        
        See Also
        --------
        plot_spectrum : Plot a spectrum.
        plot_spectrum_return : Plot a spectrum and return to param v A/N graph on right click.
        
        """
        
        global current_plot
        current_plot = 'spectrum'
        
        fig, ax = plt.subplots(1, len(centre))

        labels = ['input model + noise', 'input model', 'fitted model']
        for i in range(len(centre)):
            # Which areas to plot
            # ind = np.where(np.abs(self.x - centre[i]) <= ran)
            
            ax[i].plot(self.x, y, 'k-')
            ax[i].plot(self.x, model, 'c-')
            ax[i].plot(self.x, fit, 'r-')
            ax[i].set_xlabel(r'$\lambda$ ($\AA$)')
            ax[i].grid()
            ax[i].set_xlim([centre[i] - ran, centre[i] + ran])
            
        ax[0].set_ylabel('Amplitude (arbitrary units)')
        ax[0].legend(labels)
        if self.target_type != None:
            fig.suptitle(f'Generated and fit spectrum with emission lines of {self.target_type}')
        else:
            fig.suptitle('Generated and fit spectrum with emission lines')
        plt.show()
        
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

    def dump(self, matrices=False):
        """
        Dump the inputted and fitted parameters to a variable
        
        Parameters
        ----------
        matrices : bool, deafult=False
            Return the entire spectra, model and fit.
        
        Returns
        -------
        data : list
            The input and output values.
       
        """
        
        data = [self.As_in, self.As_out, self.As_unc_out, self.AoNs, self.AoNs_out, self.AoNs_unc_out, self.lams_in, self.lams_out, self.lams_unc_out, self.sigs_in, self.sigs_out, self.sigs_unc_out, self.vels_in, self.vels_out, self.vels_unc_out, self.peak_params, self.peaks_no, self.Nsim, self.doublet]
        
        if matrices == True:
            data.append(self.spectra_mat)
            data.append(self.model_mat)
            data.append(self.fit_mat)
            
            self.data_info = '0  As_in\n1  As_out\n2  As_unc_out\n3  AoNs\n4  AoNs_out\n5  AoNs_unc_out\n6  lams_in\n7  lams_out\n8  lams_unc_out\n9  sigs_in\n10 sigs_out\n11 sigs_unc_out\n12 vels_in\n13 vels_out\n14 vels_unc_out\n15 peak_params\n16 peaks_no\n17 Nsim\n18 doublet\n19 spectra_mat\n20 model_mat\n21 fit_mat\n22 This information'
        
        return data

    def output(self, outfile='peak_data_out.pickle', overwrite=True, matrices=False):
        """
        Dump out the input and fitted parameters using pickle, can append to data files containing the same number of peaks (and ideally the same actual peaks of course).

        Parameters
        ----------
        outfile : str, default='peak_data_out.pickle'
            The name of the file to save the data to.
        overwrite : bool, default=True
            Overwrite the file or append to it.
        matrices : bool, deafult=False
            Return the entire spectra, model and fit.
           
        """
        
        data = self.dump(matrices=matrices)
        
        if overwrite == False:
            try:
                # Try to open the data file if it exists and append the data from this run to it
                with open(outfile, 'rb') as pickle_file:
                    in_data = pickle.load(pickle_file)
                    
                # Concatenate the data
                for i in range(len(data)):
                    if i != 6 and i!= 17:
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
        self.doublet = data_in[18]
        
        if len(data_in) == 23:
            self.spectra_mat = data_in[19]
            self.model_mat = data_in[20]
            self.fit_mat = data_in[21]
        
        # Calculate fluxes
        self.f_out, self.f_unc_out = flux(self.As_out, self.sigs_out, self.As_unc_out, self.sigs_unc_out)
        self.f_in, self.f_unc_in = flux(self.As_in, self.sigs_in)
        
        # Get ratios
        self.get_line_ratios()
        
        # Get x data
        self.create_bkg()

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
    
    def on_click(self, event):
        """
        When the mouse is left double clicked on the plot show the spectrum of the closest data point, when right double clicked go back.
        
        """
        
        x_click, y_click = event.xdata, event.ydata
        
        closest_point_ind = np.argmin(np.sqrt((self.AoNs_out[line_of_interest] - x_click)**2 + (arr_of_interest[line_of_interest] - y_click)**2))
        if event.button == 1 and current_plot == 'results' and event.dblclick == True: # Left click
            plt.close()
            self.plot_spectrum_centre(self.spectra_mat[closest_point_ind], self.fit_mat[closest_point_ind], self.model_mat[closest_point_ind], [4950, 6650], 150)
        elif event.button == 3 and current_plot == 'spectrum' and event.dblclick == True: # Right click
            plt.close()
            self.plot_results(line=line_of_interest, param=param_of_interest, xlim=xlim_of_interest, ylim=ylim_of_interest)
        else:
            pass
    
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
        
        # Store line and array globally for click selecting
        global line_of_interest, arr_of_interest, param_of_interest, xlim_of_interest, ylim_of_interest
        line_of_interest = line
        arr_of_interest = array
        param_of_interest = param
        xlim_of_interest = xlim
        ylim_of_interest = ylim
        
        global current_plot
        current_plot = 'results'
        
        close_0 = self.find_not_fit(peak=line, param=param)
        
        fig, ax = plt.subplots()
        
        label = f'({param}_out - {param}_in)/{param}_in'
        plt.title(rf'{label} against A/N of peak {line} for Nsim = {self.Nsim}'+f'\nv_in = {self.peak_params[0][3]}, sig_in = {self.peak_params[0][4]}')
        plt.scatter(self.AoNs_out[line], array[line], s=0.5, label='Data')
        plt.scatter(self.AoNs_out[line][close_0], array[line][close_0], s=0.5)
        plt.xlabel('A/N')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.show()
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

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
                
        close_0 = self.find_not_fit(peak=line, param=param)

        label = f'({param}_out - {param}_in)/{param}_in'
        plt.title(rf'{label} against A/N of peak {line} for Nsim = {self.Nsim}'+f'\nv_in = {self.peak_params[0][3]}, sig_in = {self.peak_params[0][4]}')
        plt.scatter(self.AoNs_out[line], array[line], s=0.5)
        plt.errorbar(self.AoNs_out[line], array[line], unc[line], fmt='none', label='Data')
        plt.scatter(self.AoNs_out[line][close_0], array[line][close_0], s=0.5)
        plt.errorbar(self.AoNs_out[line][close_0], array[line][close_0], unc[line][close_0], fmt='none', color='#ff7f0e')
        plt.xlabel('A/N')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.show()
                
    def recover_data(self, peak=0, param='sig', ind=None):
        """
        Find the difference between the input and output values of different components.
        
        Parameters
        ----------
        peak : int, default=0
            Which peak of the spectrum to check for.
        param : {'sig', 'vel', 'A', 'flux'}, default='sig'
            Which parameter to check for.
        ind : array, defualt=None
            Which specific elements to check, if None checks all
            
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
        if ind is not None:
            arr = arr[ind][0]
        std = np.std(arr)
        med = np.median(arr)
        
        return arr, std, med

    def heatmap_sum(self, param, line, brightest=4, text=True, step=1, transparency=False):
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
        step : float, default=1
            The step size for the bins.   
        transparency : bool, default=False
            Change the transparency of the cell depending on the number of points in this range.
            
        See Also
        --------
        heatmap_brightest : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line.
        scatter_size : Generate scatter plots for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line with the size of the points depending on the number of data points in this range.
        
        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. Remove any index that doesn't appear 8 times (peaks_no times)
        # Also remove the ones that go to -1
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)[0]
        
        close_0 = self.find_not_fit(peak=line, param=param, ind=ind)
        
        interest_AoN = self.AoNs_out[line][ind]
        sum_AoN = np.sum(self.AoNs_out[:,ind], axis=0)
        interest_val = self.recover_data(peak=line, param=param)[0][ind]
    
        interest_AoN = np.delete(interest_AoN, close_0)
        sum_AoN = np.delete(sum_AoN, close_0)
        interest_val = np.delete(interest_val, close_0)
        
        x_vals = np.arange(np.floor(min(sum_AoN)), np.ceil(max(sum_AoN)), step)
        y_vals = np.arange(np.floor(min(interest_AoN)), np.ceil(max(interest_AoN)), step)

        no_points = np.empty((len(x_vals) - 1, len(y_vals) - 1))

        stds = []
        medians = []

        for i in range(1, len(x_vals)):
            ind_x = (sum_AoN < x_vals[i]) * (sum_AoN > x_vals[i] - step)

            for j in range(1, len(y_vals)):
                ind_y = (interest_AoN < y_vals[j]) * (interest_AoN > y_vals[j] - step)
                
                no_points[i-1][j-1] = len(interest_AoN[ind_y*ind_x])
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))
        
        if transparency == True:
            no_points_norm = np.log10(no_points )/ np.log10(no_points.max())
            no_points_norm = np.nan_to_num(no_points_norm, neginf=0)
        else:
            no_points_norm = np.ones_like(no_points)


        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        # Plot the standard deviations
        pc = plt.pcolormesh(x_vals, y_vals, stds.T, cmap='inferno', alpha=no_points_norm.T)
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
                        plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(stds.T[j][i], 2), ha='center', va='center', color='w', fontsize='x-small')
      
        plt.show()

         # Plot the medians
        norm = TwoSlopeNorm(vcenter=0)
        pc = plt.pcolormesh(x_vals, y_vals, medians.T, norm=norm, cmap='seismic', alpha=no_points_norm.T)
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
                        plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(medians.T[j][i], 3), ha='center', va='center', color='k', fontsize='x-small')
                
        plt.show()

    def heatmap_brightest(self, param, line, brightest=4, text=True, step=1, transparency=False):
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
        step : float, default=1
            The step size for the bins.
        transparency : bool, default=False
            Change the transparency of the cell depending on the number of points in this range.
            
        See Also
        --------
        heatmap_sum : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.
        scatter_size : Generate scatter plots for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line with the size of the points depending on the number of data points in this range.
        
        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. Remove any index that doesn't appear 8 times (peaks_no times)
        # Also remove the ones that go to -1
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)[0]
        
        close_0 = self.find_not_fit(peak=line, param=param, ind=ind)
        
        interest_AoN = self.AoNs_out[line][ind]
        brightest_AoN = self.AoNs_out[brightest][ind]
        interest_val = self.recover_data(peak=line, param=param)[0][ind]
    
        interest_AoN = np.delete(interest_AoN, close_0)
        brightest_AoN = np.delete(brightest_AoN, close_0)
        interest_val = np.delete(interest_val, close_0)
        
        x_vals = np.arange(np.floor(min(brightest_AoN)), np.ceil(max(brightest_AoN)), step)
        y_vals = np.arange(np.floor(min(interest_AoN)), np.ceil(max(interest_AoN)), step)

        stds = []
        medians = []
        
        no_points = np.empty((len(x_vals) - 1, len(y_vals) - 1))

        for i in range(1, len(x_vals)):
            ind_x = (brightest_AoN < x_vals[i]) * (brightest_AoN > x_vals[i] - step)

            for j in range(1, len(y_vals)):
                ind_y = (interest_AoN < y_vals[j]) * (interest_AoN > y_vals[j] - step)
                
                no_points[i-1][j-1] = len(interest_AoN[ind_y*ind_x])
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))
        
        if transparency == True:
            no_points_norm = np.log10(no_points )/ np.log10(no_points.max())
            no_points_norm = np.nan_to_num(no_points_norm, neginf=0)
        else:
            no_points_norm = np.ones_like(no_points)


        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        # Plot the standard deviations
        pc = plt.pcolormesh(x_vals, y_vals, stds.T, cmap='inferno', alpha=no_points_norm.T)
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
                        plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(stds.T[j][i], 3), ha='center', va='center', color='w', fontsize='x-small')
        
        plt.show()

         # Plot the medians
        norm = TwoSlopeNorm(vcenter=0)
        pc = plt.pcolormesh(x_vals, y_vals, medians.T, norm=norm, cmap='seismic', alpha=no_points_norm.T)
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
                        plt.text(x_vals[i]+step/2, y_vals[j]+step/2, np.round(medians.T[j][i], 2), ha='center', va='center', color='k', fontsize='x-small')

        plt.show()
        
    def find_not_fit(self, peak=0, param='sig', ind=None):
        """
        Count the number of lines in the data that no fit was found for them based on having a very low standard deviation.
        
        Parameters
        ----------
        peak : int, default=0
            Which peak of the spectrum to check for.
        param : {'sig', 'vel', 'A', 'flux'}, default='sig'
            Which parameter to check for.
        ind : array, defualt=None
            Which specific elements to check, if None checks all
        
        Returns
        -------
        close_0 : array
            The indices where sigs_out is close to 0
        """
        
        # Find where param is close to 0
        
        close_0 = np.unique(np.where(np.abs(self.recover_data(peak=peak, param=param, ind=ind)[0] + 1) <= 0.01)[0])
        
        return close_0
        
        
    def scatter_size(self, param, line, brightest=4, step=1):
        """
        Generate scatter plots for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the A/N of the brightest line with the size of the points depending on the number of data points in this range.

        Parameters
        ----------
        param : {'sig', 'vel', 'A', 'flux'}
            Which parameter to check for.
        line : int
            The line of interest.
        brightest : int, default=4
            The brightest line (default corresponds to H alpha in the normal input structure).
        step : float, default=1
            The step size for the bins.
            
        See Also
        --------
        heatmap_sum : Generate heatmaps for the standarad deviation and medians of the difference between input and output values for the line of interest and plot against the sum of the A/N of all lines.
     
        """
        
        label = f'({param}_out - {param}_in)/{param}_in'
        
        # Check for outliers in the output A/N of all lines (keep within AoN max + 2) and filter accordingly
        # Where AoN <= 12, gives array of all 2D indices that have this. Remove any index that doesn't appear 8 times (peaks_no times)
        # Also remove the ones that go to -1
        ind = np.where(np.unique(np.argwhere(self.AoNs_out < 12)[:,1], return_counts=True)[1] == self.peaks_no)[0]
        
        close_0 = self.find_not_fit(peak=line, param=param, ind=ind)
        
        interest_AoN = self.AoNs_out[line][ind]
        brightest_AoN = self.AoNs_out[brightest][ind]
        interest_val = self.recover_data(peak=line, param=param)[0][ind]
    
        interest_AoN = np.delete(interest_AoN, close_0)
        brightest_AoN = np.delete(brightest_AoN, close_0)
        interest_val = np.delete(interest_val, close_0)
        
        x_vals = np.arange(np.floor(min(brightest_AoN)), np.ceil(max(brightest_AoN)), step)
        y_vals = np.arange(np.floor(min(interest_AoN)), np.ceil(max(interest_AoN)), step)
        
        mesh = np.meshgrid(x_vals[:-1] + step/2, y_vals[:-1] + step/2)

        stds = []
        medians = []
        
        no_points = np.empty((len(x_vals) - 1, len(y_vals) - 1))

        for i in range(1, len(x_vals)):
            ind_x = (brightest_AoN < x_vals[i]) * (brightest_AoN > x_vals[i] - step)

            for j in range(1, len(y_vals)):
                ind_y = (interest_AoN < y_vals[j]) * (interest_AoN > y_vals[j] - step)
                
                no_points[i-1][j-1] = len(interest_AoN[ind_y*ind_x])
                stds.append(np.std(interest_val[ind_y*ind_x]))
                medians.append(np.median(interest_val[ind_y*ind_x]))

        stds = np.reshape(stds, (len(x_vals)-1, len(y_vals)-1))
        medians = np.reshape(medians, (len(x_vals)-1, len(y_vals)-1))
        
        # Plot the standard deviations
        plt.title(f'Standard deviation of {label}')
        plt.xlabel(f'A/N of line {brightest}')
        plt.ylabel(f'A/N of line {line}')
        # plt.vlines(x_vals, min(y_vals), max(y_vals), color='lightgrey')
        # plt.hlines(y_vals, min(x_vals), max(x_vals), color='lightgrey')
        plt.plot(x_vals, x_vals * self.line_ratios[line]/self.line_ratios[brightest], color='k', linestyle=':')
        plt.scatter(mesh[0], mesh[1], s=np.log10(no_points.T)*10, c=stds.T, cmap='inferno')
        plt.colorbar()
        plt.show()

        #  # Plot the medians
        norm = TwoSlopeNorm(vcenter=0)
        # pc = plt.pcolormesh(x_vals, y_vals, medians.T, norm=norm, cmap='seismic')
        # plt.colorbar(pc)
        plt.title(f'Median of {label}')
        plt.xlabel(f'A/N of line {brightest}')
        plt.ylabel(f'A/N of line {line}')
        # plt.vlines(x_vals, min(y_vals), max(y_vals), color='lightgrey')
        # plt.hlines(y_vals, min(x_vals), max(x_vals), color='lightgrey')
        plt.plot(x_vals, x_vals * self.line_ratios[line]/self.line_ratios[brightest], color='k', linestyle=':')
        plt.scatter(mesh[0], mesh[1], s=np.log10(no_points.T)*10, c=medians.T, cmap='coolwarm', norm=norm)
        plt.colorbar()
        plt.show()
        
