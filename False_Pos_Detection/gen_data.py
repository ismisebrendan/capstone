import numpy as np
import sys
import os
import datetime

spec_path = os.path.abspath('../') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum


Nsim = 1000


suffixes = ['Sy', 'SF', 'PN', 'LINER']

labels = ['seyfert', 'star_forming', 'planetary_nebula', 'LINER']


    
for i in range(len(suffixes)):
    print(datetime.datetime.now())
    print(labels[i])
    
    # Generate Spectrum object
    spec = Spectrum(f'lines_in/lines_in_{suffixes[i]}.txt', f'fitting/fitting_{suffixes[i]}.txt', Nsim=Nsim)
    
    # Import data
    print('Importing data')
    spec.get_data()
    
    # Generate actual amplitudes
    print(datetime.datetime.now())
    print("Generating full spectra")
    spec.simulation()
    AoN_out = spec.AoNs_out
    data_out = np.transpose(np.concatenate((AoN_out, np.ones((8, Nsim)))))
    
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
    
    # Check that the folder exists
    try:
        os.mkdir(name)
    except:
        pass
    
    files_present = [f for f in os.listdir(name) if os.path.isfile(os.path.join(name, f)) and f[:len(name)] == name and f[-4:] == ".txt" and 'all' not in f]
    if len(files_present) < 10:
        file_out = f'{name}/{name}_0{len(files_present)}.txt'
    else:
        file_out = f'{name}/{name}_{len(files_present)}.txt'
    
    np.savetxt(file_out, data_out, header=header)