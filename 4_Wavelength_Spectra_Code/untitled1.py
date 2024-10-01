# Import
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lmfit.models import Model
from lmfit import Parameters
from funcs import gaussian, background

# The speed of light
c = 299792.458 #km/s

####################
# Read in the data #
####################

peaksdata = 'lines_in.txt'
fitdata = 'fitting.txt'
# peaksdata = 'peaks.txt'

# Parameter variables
peak_params = []
fit_params = []

# Dependency variables
doublet = [] # Is a peak free or a doublet of another one - velocity, sigma and amplitude dependant
vel_dep = [] # Is the peak's velocity dependant on another one
prof_dep = [] # Is the peak's profile dependant on another one

# Plotting data
with open(peaksdata) as f:
    data_in = f.readlines()
    for i in range(1, len(data_in)):
        entry = data_in[i].split()
        # Convert data entries from string to float
        param = [float(p) for p in entry[2:7]]
        peak_params.append(param)

        # If a line reference itself, otherwise reference specified line
        if entry[7] == 'l':
            doublet.append(i-1)
                
        elif entry[7][0] == 'd':
            doublet.append(int(entry[7][1:]))
peaks_no = len(peak_params)

# Fitting data
with open(fitdata) as f:
    data_fit = f.readlines()
    for i in range(1, len(data_fit)):
        entry_fit = data_fit[i].split()
        # See if this line is in the peak data
        for j in range(1, len(data_in)):
            entry_in = data_in[j].split()
            if entry_fit[1] == entry_in[1]:
                # Convert data entries from string to float
                param = [float(p) for p in entry_fit[2:7]]
                fit_params.append(param)

                # If a line reference itself, otherwise reference specified line
                if entry[7] == 'l':
                    if entry[8] == 'f':
                        # Completely free
                        vel_dep.append(i-1)
                        prof_dep.append(i-1)
                    elif entry[8][0] == 't':
                        # Moving with it and same profile
                        vel_dep.append(int(entry[8][1:]))
                        prof_dep.append(int(entry[8][1:]))
                    elif entry[8][0] == 'v':
                        # Multiple species moving together but profiles different
                        vel_dep.append(int(entry[8][1:]))
                        prof_dep.append(i)
                        
                elif entry[7][0] == 'd':
                    vel_dep.append(int(entry[7][1:]))
                    prof_dep.append(int(entry[7][1:]))