from funcs import gaussian, background, flux
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model
from lmfit import Parameters
import pickle

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
    """
    
    def __init__(self, peaksdata, fitdata, sig_resolution=0.5, sig_sampling=4.0, bkg=100, Nsim=1000, AoN_min=0, AoN_max=10) -> None:
        self.peaksdata = peaksdata
        self.fitdata = fitdata
        self.sig_resolution = sig_resolution
        self.sig_sampling = sig_sampling
        self.bkg = bkg
        self.Nsim = Nsim
        self.AoN_min = AoN_min
        self.AoN_max = AoN_max
        
        self.peak_params = []
        self.fit_params = []
        self.doublet = []
        self.vel_dep = []
        self.prof_dep = []
        self.peaks_no = 0
        
        self.data = np.array([])
    
    def print_info(self):
        print('Spectrum object, generates and fits synthetic spectra from files.')

    def get_data(self):
        """
        Take the input files and extract the data from them.
        """
        
        with open(self.peaksdata) as f:
            data_in = f.readlines()
            for i in range(1, len(data_in)):
                entry_in = data_in[i].split()
                # Convert data entries from string to float
                param = [float(p) for p in entry_in[2:7]]
                self.peak_params.append(param)

                # If a line reference itself, otherwise reference specified line
                if entry_in[7] == 'l':
                    self.doublet.append(i-1)
                        
                elif entry_in[7][0] == 'd':
                    self.doublet.append(int(entry_in[7][1:]))
        self.peaks_no = len(self.peak_params)

        # Fitting data
        with open(self.fitdata) as f:
            data_fit = f.readlines()
            for i in range(1, len(data_fit)):
                entry_fit = data_fit[i].split()
                # See if this is in the peak data
                for j in range(1, len(data_in)):
                    entry_in = data_in[j].split()
                    if entry_fit[1] == entry_in[1]:
                        # Convert data entries from string to float
                        param = [float(p) for p in entry_fit[2:7]]
                        self.fit_params.append(param)
        
                        # If a line reference itself, otherwise reference specified line
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
                            self.vel_dep.append(int(entry_fit[7][1:]))
                            self.prof_dep.append(int(entry_fit[7][1:]))
        
        # Give the lines the same velocity and sigma as the lines they are dependant on
        for i in range(self.peaks_no - 1):
            self.peak_params[i][3] = self.peak_params[self.doublet[i]][3]
            self.peak_params[i][4] = self.peak_params[self.doublet[i]][4]
            self.vel_dep[i] = self.vel_dep[self.doublet[i]]
            self.prof_dep[i] = self.prof_dep[self.doublet[i]]
            
            self.fit_params[i][3] = self.fit_params[self.vel_dep[i]][3]
            self.fit_params[i][4] = self.fit_params[self.prof_dep[i]][4]

    def create_bkg(self):
        """
        Create the array for the background level to the spectrum, also calls Spectrum.get_data.
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
        Generates the background level of the model
        """
        
        self.create_bkg()
        self.gen_arrs()

        self.model = background(self.x, self.bkg)
        gaussian_models = []
        self.mod = Model(background, prefix='bkg_')

        # Loop through for the number of peaks
        for i, (lam, relative_A_l, relative_A_u, vel, sig) in enumerate(self.peak_params):
            gauss = Model(gaussian, prefix=f'g{i+1}_')
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
        
        y_vals = []
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        for index, AoN in enumerate(self.AoNs):
            A = np.sqrt(self.bkg) * AoN
            ##################################
            # Generate Gaussian + Noise data #
            ##################################
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
                
            # Generate noise and add to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            y_vals.append(y)
        
        return y_vals
    
    def simulation(self, plotting=False):
        """
        Generate and fit the synthetic spectrum.
        
        Parameters
        ----------
        plotting : bool, default=False
            Whether or not to plot the graphs.
        """
        self.y_vals = []
        self.init_model()
        self.AoNs = np.random.random(self.Nsim) * (self.AoN_max - self.AoN_min) + self.AoN_min
        for index, AoN in enumerate(self.AoNs):
            print(f'Simulation {index}/{self.Nsim}')
            A = np.sqrt(self.bkg) * AoN
            ##################################
            # Generate Gaussian + Noise data #
            ##################################
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
                
            # Generate noise and add to the model
            noise = np.random.randn(len(self.x)) * np.sqrt(self.model)
            y = self.model + noise
            
            self.y_vals.append(y)
            
            ##################
            # Fit with LMfit #
            ##################
            pfit = Parameters()
        
            # The background level
            pfit.add('bkg_bkg', value=self.bkg, vary=True)
            
            # Setting up parameters for the peaks (g_i)
            for i in range(self.peaks_no):
                # These values are fixed either physically or by the instrument
                pfit.add(f'g{i+1}_lam_rf', value=self.fit_params[i][0], vary=False)
                pfit.add(name=f'g{i+1}_sig_resolution', value=self.sig_resolution, vary=False)
                
                if self.doublet[i] == i:
                    # For free lines take initial guess as largest y value in the dataset
                    pfit.add(f'g{i+1}_A', value=np.max(y) - np.median(y), min=self.fit_params[i][1], max=self.fit_params[i][2])
                    # If independent in terms of velocity and sigma take those as its initial estimates
                    if self.vel_dep[i] == i:
                        pfit.add(f'g{i+1}_vel', value=self.fit_params[i][3])
                    if self.prof_dep[i] == i:
                        pfit.add(f'g{i+1}_sig', value=self.fit_params[i][4])

            # Loop again for amplitudes, velocities and sigmas of doublet peaks - have to do afterwards as could be dependant on a line appearing after it in the file
            for i in range(self.peaks_no):
                if self.vel_dep[i] != i:
                    # Velocity must be that of another peak
                    pfit.add(f'g{i+1}_vel', expr=f'g{self.vel_dep[i]+1}_vel')
                if self.prof_dep[i] != i:
                    # Sigma must be that of another peak
                    pfit.add(f'g{i+1}_sig', expr=f'g{self.prof_dep[i]+1}_sig')
                if self.doublet[i] != i:
                    # The amplitude of the peak is equal to the that of the reference line times some value
                    if self.fit_params[i][1] != self.fit_params[i][2]:
                        pfit.add(f'g{i+1}_delta', min=self.fit_params[i][1], max=self.fit_params[i][2])
                    else:
                        pfit.add(f'g{i+1}_delta', expr=f'{self.fit_params[i][2]}')
                    pfit.add(f'g{i+1}_A', expr=f'g{i+1}_delta * g{self.doublet[i]+1}_A')
                
                
            fit = self.mod.fit(y, pfit, x=self.x)
            
            # Save generated data
            for peak in range(self.peaks_no):
                self.As_out[peak][index] = fit.params[f'g{peak+1}_A'].value
                self.As_unc_out[peak][index] = fit.params[f'g{peak+1}_A'].stderr
                self.sigs_out[peak][index] = fit.params[f'g{peak+1}_sig'].value
                self.sigs_unc_out[peak][index] = fit.params[f'g{peak+1}_sig'].stderr
                self.vels_out[peak][index] = fit.params[f'g{peak+1}_vel'].value
                self.vels_unc_out[peak][index] = fit.params[f'g{peak+1}_vel'].stderr
                self.lams_out[peak][index] = fit.params[f'g{peak+1}_lam_rf'].value
                self.lams_unc_out[peak][index] = fit.params[f'g{peak+1}_lam_rf'].stderr
                self.f_out, self.f_unc_out = flux(self.As_out, self.sigs_out, self.As_unc_out, self.sigs_unc_out)
                self.f_in, self.f_unc_in = flux(self.As_in, self.sigs_in)

            # Plotting
            if plotting == True:
                labels = ['input model + noise', 'input model', 'fitted model']
                plt.plot(self.x, y, 'k-')
                plt.plot(self.x, self.model, 'c-')
                plt.plot(self.x, fit.best_fit, 'r-')
                plt.legend(labels)
                plt.grid()
                plt.show()
        
        for peak in range(self.peaks_no):
            self.AoNs_out[peak] = self.As_out[peak]/np.sqrt(self.bkg)
            self.AoNs_unc_out[peak] = self.As_unc_out[peak]/np.sqrt(self.bkg)

    def output(self, outfile='peak_data_out.pickle'):
        """
        Dump out the input and fitted parameters using pickle.

        Parameters
        ----------
        outfile : str, default='peak_data_out.pickle'
            The name of the file to save the data to.
        """
        
        data = [self.As_in, self.As_out, self.As_unc_out, self.AoNs, self.AoNs_out, self.AoNs_unc_out, self.lams_in, self.lams_out, self.lams_unc_out, self.sigs_in, self.sigs_out, self.sigs_unc_out, self.vels_in, self.vels_out, self.vels_unc_out, self.peaks_no, self.Nsim]
        filename = 'peak_data_out.pickle'
        outfile = open(filename, 'wb')
        pickle.dump(data, outfile)
        outfile.close()

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
        """

        for i in range(self.peaks_no):
            if type(value) == list:
                self.peak_params[i][parameter] = value[i]
                self.fit_params[i][parameter] = value[i]
            else:
                self.peak_params[i][parameter] = value
                self.fit_params[i][parameter] = value
                
    def dump(self):   
        """
        Dump the data to a variable.
        
        Returns
        -------
        data : list
            All the relevant input and output data
        """
        
        data = [self.As_in, self.As_out, self.As_unc_out, self.AoNs, self.AoNs_out, self.AoNs_unc_out, self.lams_in, self.lams_out, self.lams_unc_out, self.sigs_in, self.sigs_out, self.sigs_unc_out, self.vels_in, self.vels_out, self.vels_unc_out, self.peaks_no, self.Nsim]
    
        return data
                
    def plot_results(self, peak=0, param='sig', xlim=[-0.2, 11], ylim=[-5, 5]):
        """
        Plot the difference between the input and output values of different components.
        
        Parameters
        ----------
        peak : int, default=0
            Which peak of the spectrum to plot for.
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
            
        label = f'({param}_out - {param}_in)/{param}_in'
        plt.title(rf'{label} against A/N of peak {peak+1} for Nsim = {self.Nsim}'+f'\nv_in = {self.peak_params[0][3]}, sig_in = {self.peak_params[0][4]}')
        plt.scatter(self.AoNs_out[peak], array[peak], s=0.5)
        plt.xlabel('A/N')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

    def plot_results_err(self, peak=0, param='sig', xlim=[-0.2, 11], ylim=[-5, 5]):
        """
        Plot the difference between the input and output values of different components.
        
        Parameters
        ----------
        peak : int, default=0
            Which peak of the spectrum to plot for.
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
                
        label = f'({param}_out - {param}_in)/{param}_in'
        plt.title(rf'{label} against A/N of peak {peak+1} for Nsim = {self.Nsim}'+f'\nv_in = {self.peak_params[0][3]}, sig_in = {self.peak_params[0][4]}')
        plt.scatter(self.AoNs_out[peak], array[peak], s=0.5)
        plt.errorbar(self.AoNs_out[peak], array[peak], unc[0], fmt='none')
        plt.xlabel('A/N')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
                
                
                
                
                
