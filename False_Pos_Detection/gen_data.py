import numpy as np
import sys
import os
import datetime

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum


rm = [False, True]

Nsim = 10000


suffixes = ['SF', 'LINER', 'PN', 'Sy']

labels = ['star_forming', 'LINER', 'planetary_nebula', 'seyfert']

for j in range(len(rm)):

    remove_some = rm[j]
    
    for i in range(len(suffixes)):
        print(datetime.datetime.now())
        print(labels[i], remove_some)
        
        # Generate Spectrum object
        spec = Spectrum(f'lines_in_{suffixes[i]}.txt', f'fitting_{suffixes[i]}.txt', Nsim=Nsim)
        
        # Import data
        print('Importing data')
        spec.get_data()
        
        # Generate actual amplitudes
        print(datetime.datetime.now())
        print("Generating full spectra")
        spec.simulation()
        AoN_out = spec.AoNs_out
        data_out = np.transpose(np.concatenate((AoN_out, np.ones((8, Nsim)))))
        
        
        if remove_some == True:
            print(datetime.datetime.now())
            print('Generating spectra with missing lines')
            spec.simulation_false()
            AoN_out_missing = spec.AoNs_out
            lines_present = spec.keep_lines
        
            # Save data
            missing_lines = np.transpose(np.concatenate((AoN_out_missing, lines_present)))
            
            data_out = np.concatenate((data_out, missing_lines))
            
            
            header = f'first {Nsim} are full spectra, next {Nsim} have random lines missing, next {Nsim} is just noise'
        else:
            header = f'first {Nsim} are full spectra, next {Nsim} is just noise'
        
        
        # Generate false positives
        print(datetime.datetime.now())
        print('Generating false positives')
        spec.AoN_min = 0
        spec.AoN_max = 0
        spec.simulation()
        AoN_out = spec.AoNs_out
        
        # Save data
        empty_data = np.transpose(np.concatenate((AoN_out, np.zeros((8, Nsim)))))
        data_out = np.concatenate((data_out, empty_data))
        
        name = labels[i]
        
        files_present = [f for f in os.listdir(name) if os.path.isfile(os.path.join(name, f)) and f[:len(name)] == name and f[-4:] == ".txt"]
        file_out = f'{name}/{name}_{len(files_present)}.txt'
        
        np.savetxt(file_out, data_out, header=header)