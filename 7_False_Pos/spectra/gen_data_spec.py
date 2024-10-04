# 
# Generate spectral data for the ANN to learn from
# 

import numpy as np
import argparse
import sys
import os

spec_path = os.path.abspath('../..') + '/Spectrum_Obj'
sys.path.insert(0, spec_path)

from spectrum_obj import Spectrum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and fit synthetic spectra using the Spectrum object for both actual amplitudes and false positives for use when training an ANN.')
    parser.add_argument('-n', '--Nsim', type=int, help='The number of real spectra to generate.', default=1000)
    parser.add_argument('-f', '--false', type=int, help='The number of false positives to generate (default is the same as Nsim).', default=None)
    parser.add_argument('-o', '--file_out', type=str, help='The filename to save this data to.', default=None)
    args = parser.parse_args()
    
    Nsim = args.Nsim
    if args.false == None:
        false = Nsim
    else:
        false = args.false

    # Generate Spectrum object
    spec = Spectrum('../lines_in.txt', '../fitting.txt', Nsim=Nsim, AoN_min=4)

    # Import data
    print('Importing data')
    spec.get_data()

    # Generate actual amplitudes
    print('Generating spectral data')
    y_vals = spec.generate()

    # Save data
    data_out = np.transpose(np.concatenate((y_vals.T, np.ones((1, Nsim)))))

    # Generate false positives
    print('Generating false positives')
    spec.AoN_min = 0
    spec.AoN_max = 0
    spec.Nsim = false
    y_vals = spec.generate()

    # Save data
    empty_data = np.transpose(np.concatenate((y_vals.T, np.zeros((1, false)))))
    data_out = np.concatenate((data_out, empty_data))
    
    # np.random.shuffle(data_out)

    if args.file_out == None:
        files_present = [f for f in os.listdir() if os.path.isfile(os.path.join(".", f)) and f[:7] == "spectra" and f[-4:] == ".txt"]
        file_out = f'spectra{len(files_present)}.txt'
    else:
        file_out = args.file_out
    
    np.savetxt(file_out, data_out)