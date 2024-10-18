# 
# Generate amplitude data for the ANN to learn from
# 

import sys
import os

spec_path = os.path.abspath('../..') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum
import numpy as np
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and fit synthetic spectra using the Spectrum object for both actual amplitudes and false positives for use when training an ANN.')
    parser.add_argument('-n', '--Nsim', type=int, help='The number of real spectra to generate.', default=1000)
    parser.add_argument('-f', '--false', type=int, help='The number of false positives to generate (default is the same as Nsim).', default=None)
    args = parser.parse_args()
    
    Nsim = args.Nsim
    if args.false == None:
        false = Nsim
    else:
        false = args.false

    # Generate Spectrum object
    spec = Spectrum('../lines_in.txt', '../fitting.txt', Nsim=Nsim)

    # Import data
    print('Importing data')
    spec.get_data()

    # Generate actual amplitudes
    print('Generating spectral data')
    spec.simulation()
    AoN_out = spec.AoNs_out

    # Save data
    data_out = np.transpose(np.concatenate((AoN_out, [spec.AoNs.T/10])))
    
    # Generate false positives
    print('Generating false positives')
    spec.AoN_min = 0
    spec.AoN_max = 0
    spec.Nsim = false
    spec.simulation()
    AoN_out = spec.AoNs_out

    # Save data
    empty_data = np.transpose(np.concatenate((AoN_out, np.zeros((1, false)))))
    data_out = np.concatenate((data_out, empty_data))
    
    np.random.shuffle(data_out)

    files_present = [f for f in os.listdir() if os.path.isfile(os.path.join(".", f)) and f[:7] == "amps" and f[-4:] == ".txt"]
    file_out = f'amps{len(files_present)}.txt'
    
    np.savetxt(file_out, data_out)